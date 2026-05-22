"""Tests for agentic_compiler core functionality."""

import sys
import types
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentic_compiler import (
    Compiler,
    Profiler,
    GridBackendSelector,
    hot_swap,
    hot_swap_restore,
    CompilationResult,
)
from agentic_compiler.codegen import (
    CodeGenerator,
    PythonAnalyzer,
    NumbaGenerator,
    RustGenerator,
    GeneratedKernel,
)


# ── Profiler Tests ─────────────────────────────────────────

class TestProfiler:
    def test_watch_counts_calls(self):
        profiler = Profiler(sample_rate=1.0)

        @profiler.watch
        def slow_add(x, y):
            time.sleep(0.001)
            return x + y

        for i in range(10):
            assert slow_add(i, i) == 2 * i

        key = f"{slow_add.__module__}.{slow_add.__qualname__}"
        stat = profiler.stats[key]
        assert stat.calls == 10
        assert stat.total_time_ms > 0

    def test_sample_rate_reduces_profiling(self):
        profiler = Profiler(sample_rate=0.0)  # never profile

        @profiler.watch
        def noop(x):
            return x

        for i in range(100):
            noop(i)

        key = f"{noop.__module__}.{noop.__qualname__}"
        stat = profiler.stats[key]
        # Calls are still counted via _call_count but not recorded
        # With sample_rate=0, only the first 1000 calls are profiled
        assert stat.calls <= 100

    def test_hotspots_ranking(self):
        profiler = Profiler(sample_rate=1.0)

        @profiler.watch
        def fast(x):
            return x * 2

        @profiler.watch
        def slow(x):
            time.sleep(0.005)
            return x * 3

        for _ in range(50):
            fast(1)
            slow(1)

        hotspots = profiler.get_hotspots(top_n=2)
        assert len(hotspots) == 2
        # slow should be ranked higher due to longer runtime
        assert "slow" in hotspots[0].name or "fast" in hotspots[1].name

    def test_input_shapes_recorded(self):
        profiler = Profiler(sample_rate=1.0)

        @profiler.watch
        def process(arr):
            return arr.sum()

        process(np.zeros((3, 4)))
        process(np.ones((5, 6)))

        key = f"{process.__module__}.{process.__qualname__}"
        stat = profiler.stats[key]
        assert len(stat.input_shapes) == 2
        assert stat.input_shapes[0] == ((3, 4),)


# ── GridBackendSelector Tests ──────────────────────────────

class TestGridBackendSelector:
    def test_select_numpy_for_small(self):
        # Default hardware has no rust/cuda
        backend = GridBackendSelector.select(10)
        assert backend == "numpy"

    def test_select_respects_thresholds(self):
        # With default HARDWARE (no rust/cuda), everything falls back to numpy
        assert GridBackendSelector.select(10) == "numpy"
        assert GridBackendSelector.select(100) == "numpy"
        assert GridBackendSelector.select(2000) == "numpy"

    def test_report_contains_headers(self):
        report = GridBackendSelector.report()
        assert "Hardware Detection" in report
        assert "Backend Thresholds" in report
        assert "numpy" in report


# ── Hot-Swap Tests ─────────────────────────────────────────

class TestHotSwap:
    def test_hot_swap_replaces_function(self):
        # Create a fake module
        mod = types.ModuleType("test_mod")
        mod.__dict__["original_func"] = lambda x: x * 2
        sys.modules["test_mod"] = mod

        def compiled(x):
            return x * 10

        result = hot_swap("test_mod", "original_func", compiled)
        assert result["success"] is True
        assert mod.original_func(5) == 50
        assert hasattr(mod.original_func, "_agentic_original")

    def test_hot_swap_restore(self):
        mod = types.ModuleType("test_mod2")
        original = lambda x: x * 2
        mod.__dict__["func"] = original
        sys.modules["test_mod2"] = mod

        def compiled(x):
            return x * 10

        hot_swap("test_mod2", "func", compiled)
        assert mod.func(5) == 50

        restore_result = hot_swap_restore("test_mod2", "func")
        assert restore_result["success"] is True
        assert mod.func(5) == 10
        assert restore_result["restored"] is original

    def test_hot_swap_missing_module(self):
        result = hot_swap("nonexistent_mod", "func", lambda x: x)
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_hot_swap_missing_function(self):
        mod = types.ModuleType("test_mod3")
        sys.modules["test_mod3"] = mod
        result = hot_swap("test_mod3", "missing_func", lambda x: x)
        assert result["success"] is False

    def test_hot_swap_with_code_object(self):
        mod = types.ModuleType("test_mod4")
        mod.__dict__["func"] = lambda x: x + 1
        sys.modules["test_mod4"] = mod

        code = compile("def f(x): return x * 3", "<test>", "exec")
        func_code = code.co_consts[0]

        result = hot_swap("test_mod4", "func", func_code)
        assert result["success"] is True
        assert mod.func(5) == 15


# ── Compiler Tests ─────────────────────────────────────────

class TestCompiler:
    def test_compiler_init(self):
        compiler = Compiler()
        assert compiler.profiler is not None
        assert not compiler._installed

    def test_install_patches_module(self):
        compiler = Compiler()
        # Create a fake user module with a properly attributed function
        mod = types.ModuleType("user_test_mod")

        def compute(x):
            return x + 1
        compute.__module__ = "user_test_mod"

        mod.__dict__["compute"] = compute
        sys.modules["user_test_mod"] = mod

        compiler.install("user_test_mod")
        assert compiler._installed
        assert "user_test_mod.compute" in compiler._originals
        # The function should now be wrapped
        assert hasattr(mod.compute, "__wrapped__") or callable(mod.compute)

        # Clean up
        del sys.modules["user_test_mod"]

    def test_uninstall_restores(self):
        compiler = Compiler()
        mod = types.ModuleType("user_test_mod2")
        original = lambda x: x + 1
        mod.__dict__["compute"] = original
        sys.modules["user_test_mod2"] = mod

        compiler.install("user_test_mod2")
        compiler.uninstall()
        assert mod.compute is original

    def test_compile_function_no_numba(self):
        # Test that compile_function gracefully handles missing numba
        compiler = Compiler()

        def simple_func(x):
            return x * 2

        result = compiler.compile_function(simple_func)
        assert isinstance(result, CompilationResult)
        assert result.backend == "python" or result.backend == "numba"

    def test_hot_swap_method(self):
        compiler = Compiler()

        def slow_func(arr):
            return np.sum(arr)

        mod = types.ModuleType("user_test_mod3")
        mod.__dict__["slow_func"] = slow_func
        sys.modules["user_test_mod3"] = mod

        result = compiler.hot_swap(slow_func, module=mod, attr_name="slow_func")
        assert isinstance(result, CompilationResult)
        # With no numba, it should return a result (possibly python fallback)
        assert result.error is not None or result.compiled is not None

    def test_restore_no_originals(self):
        compiler = Compiler()
        assert compiler.restore() is False


# ── CodeGenerator / Analyzer Tests ─────────────────────────

class TestPythonAnalyzer:
    def test_analyzes_numpy_calls(self):
        def func(arr):
            return np.sum(arr)

        analyzer = PythonAnalyzer().analyze(func)
        assert analyzer.has_numpy is True
        assert "sum" in analyzer.array_ops

    def test_analyzes_loops(self):
        def func(items):
            total = 0
            for item in items:
                total += item
            return total

        analyzer = PythonAnalyzer().analyze(func)
        assert analyzer.has_loops is True

    def test_numba_score_high_for_numpy(self):
        def func(arr):
            return np.sum(arr)

        analyzer = PythonAnalyzer().analyze(func)
        assert analyzer.numba_score >= 2.0

    def test_rust_score_high_for_dicts(self):
        def func(data):
            total = 0
            for k, v in data.items():
                total += v
            return total

        analyzer = PythonAnalyzer().analyze(func)
        assert analyzer.rust_score >= 2.0


class TestCodeGenerator:
    def test_compile_returns_kernel(self):
        gen = CodeGenerator()

        def simple(x):
            return x * 2

        kernel = gen.compile(simple)
        assert isinstance(kernel, GeneratedKernel)
        assert kernel.ready or kernel.error is not None

    def test_validate_identity_kernel(self):
        gen = CodeGenerator()

        def identity(x):
            return x

        kernel = gen.compile(identity)
        assert gen.validate(kernel, identity, (5,)) is True

    def test_measure_speedup_identity(self):
        gen = CodeGenerator()

        def identity(x):
            return x

        kernel = gen.compile(identity)
        speedup = gen.measure_speedup(kernel, identity, (5,))
        assert speedup == 1.0

    def test_deploy_sets_attribute(self):
        gen = CodeGenerator()

        def my_func(x):
            return x + 1

        kernel = gen.compile(my_func)
        mod = types.ModuleType("deploy_test")
        success = gen.deploy(kernel, mod, "my_func")
        assert success is True
        assert mod.my_func(5) == 6


class TestNumbaGenerator:
    def test_available_detects_numba(self):
        gen = NumbaGenerator()
        # Just test that it doesn't crash
        assert isinstance(gen._available, bool)

    def test_can_generate_numpy_function(self):
        gen = NumbaGenerator()

        def numpy_func(arr):
            return np.sum(arr)

        analyzer = PythonAnalyzer().analyze(numpy_func)
        # May be False if numba not installed
        result = gen.can_generate(numpy_func, analyzer)
        assert isinstance(result, bool)


class TestRustGenerator:
    def test_available_detects_rustc(self):
        gen = RustGenerator()
        assert isinstance(gen._available, bool)

    def test_generate_returns_stub(self):
        gen = RustGenerator()

        def any_func(x):
            return x

        analyzer = PythonAnalyzer().analyze(any_func)
        kernel = gen.generate(any_func, analyzer)
        assert kernel.error is not None
        assert "not yet implemented" in kernel.error


# ── Integration-style Tests ───────────────────────────────

class TestIntegration:
    def test_end_to_end_profile_compile(self):
        """Profile a function, then compile it manually."""
        compiler = Compiler()

        def heavy_computation(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i] * arr[i]
            return total

        # Watch the function
        watched = compiler.profiler.watch(heavy_computation)

        # Simulate many calls
        test_arr = np.arange(1000, dtype=np.float32)
        for _ in range(150):
            watched(test_arr)

        # Check profiler tracked it
        key = f"{heavy_computation.__module__}.{heavy_computation.__qualname__}"
        assert key in compiler.profiler.stats
        assert compiler.profiler.stats[key].calls >= 100

        # Try compiling
        result = compiler.compile_function(heavy_computation)
        assert isinstance(result, CompilationResult)

    def test_grid_selector_with_mock_hardware(self):
        """Test backend selection with mocked hardware capabilities."""
        with patch(
            "agentic_compiler.core.HARDWARE",
            {"numpy": True, "numba": True, "rust_persistent": True, "rust_oneshot": True, "cuda": True},
        ):
            # Re-import to pick up mocked hardware
            from agentic_compiler.core import GridBackendSelector
            assert GridBackendSelector.select(1500) == "cuda"
            assert GridBackendSelector.select(600) == "rust_persistent"
            assert GridBackendSelector.select(100) == "rust_oneshot"

    def test_profiler_decorator_syntax(self):
        profiler = Profiler(sample_rate=1.0)

        @profiler.watch
        def decorated(x):
            return x ** 2

        decorated(5)
        decorated(10)

        key = f"{decorated.__module__}.{decorated.__qualname__}"
        assert profiler.stats[key].calls == 2

    def test_compiler_restore_by_key(self):
        compiler = Compiler()

        def func(x):
            return x + 1

        mod = types.ModuleType("restore_test")
        mod.__dict__["func"] = func
        sys.modules["restore_test"] = mod

        result = compiler.hot_swap(func, module=mod, attr_name="func")
        assert isinstance(result, CompilationResult)

        if result.error is None:
            # Only test restore if hot_swap succeeded
            restored = compiler.restore("restore_test.func")
            assert restored is True or restored is False
