# DEVELOPER.md — agentic-compiler

> A practical guide for anyone hacking on, extending, or debugging the agentic compiler.

---

## 1. Architecture Overview

The agentic compiler is a five-stage pipeline that watches your Python program, identifies hot paths, compiles them to a faster backend, and silently swaps them in at runtime.

```
┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────┐
│ Profiler │ ──▶ │ Analyzer │ ──▶ │ CodeGenerator  │ ──▶ │ Validator│ ──▶ │ Deployer │
│  (watch) │     │  (rank)  │     │   (compile)  │     │(A/B test)│     │(hot-swap)│
└──────────┘     └──────────┘     └──────────────┘     └──────────┘     └──────────┘
```

### Stage 1: Profiler (`core.Profiler`)

- **Decorates** user functions via `Profiler.watch()` or `Compiler.install()` monkey-patching.
- **Samples** calls (default 5%) to keep overhead low. Uses deterministic sampling for the first 1,000 calls, then probabilistic.
- **Records** per-function stats: call count, min/max/avg time, total time, and input shapes (last 20 only).
- **Ranks** functions by `optimization_potential = calls × avg_time²` — this surfaces functions that are both frequent *and* slow.

Key data structure: `FunctionStats` (defined in `core.py`).

### Stage 2: Analyzer (`codegen.PythonAnalyzer`)

- Walks the **AST** of hot functions using `ast.NodeVisitor`.
- Scores each function for Numba vs Rust suitability:
  - **Numba score** (`numba_score`): +3 for numpy calls, +2 for loops, −1 for dicts/strings.
  - **Rust score** (`rust_score`): +3 for loops, +2 for dicts, +1 for strings, −1 for heavy numpy.
- This is a **static heuristic** — no runtime profiling data other than "is this function hot enough."

### Stage 3: CodeGenerator (`codegen.CodeGenerator`)

- Selects the **best backend** based on `can_generate()` scores and hardware availability.
- Delegates to backend-specific generators:
  - `NumbaGenerator`: wraps the function in `@numba.njit(cache=True, fastmath=True)`
  - `RustGenerator`: currently a stub (v2 will use procedural generation via `syn` + `quote`)
- Falls back to an **identity kernel** (`source_language="python"`) if no backend can compile the function.

### Stage 4: Validator (`codegen.CodeGenerator.validate()`)

- Runs an **A/B correctness test**: calls both original and compiled versions with the same synthetic arguments.
- Compares outputs with `numpy.allclose` for arrays, `abs(a−b) < 1e-5` for scalars, and recursive equality for lists/tuples.
- Defaults to **5 trials** — all must pass.

### Stage 5: Deployer (`core._hot_swap()`)

- Replaces the module attribute with the compiled callable using `setattr(module, attr_name, replacement)`.
- **Preserves rollback metadata** on the replacement:
  - `_agentic_original` → original function
  - `_agentic_module` / `_agentic_name` → for self-describing restores
- Only deploys if `validated=True` and `speedup ≥ SPEEDUP_THRESHOLD` (2× by default).

---

## 2. How to Add a New Backend

Adding a new backend (e.g., **Triton**, **MLX**, or a custom C++ extension) requires three steps: implement the generator class, register it in the orchestrator, and wire up hardware detection.

### Step 1: Implement a `JitBackend` subclass

Create a new file (e.g., `agentic_compiler/triton_backend.py`) or add to `codegen.py`:

```python
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from agentic_compiler.core import JitBackend, FunctionStats
from agentic_compiler.codegen import GeneratedKernel, PythonAnalyzer

class TritonBackend(JitBackend):
    """Triton GPU kernel backend for element-wise / matmul patterns."""
    name = "triton"

    def __init__(self) -> None:
        self._available = False
        try:
            import triton
            import triton.language as tl
            self._triton = triton
            self._tl = tl
            self._available = True
        except ImportError:
            pass

    def can_compile(self, func: Callable, stats: FunctionStats) -> bool:
        if not self._available:
            return False
        analyzer = PythonAnalyzer().analyze(func)
        # Triton works best on element-wise array ops with large inputs
        return (
            stats.calls > 100
            and analyzer.has_numpy
            and analyzer.has_loops
            and not analyzer.has_dicts
        )

    def compile(self, func: Callable, stats: FunctionStats) -> CompilationResult:
        from agentic_compiler.core import CompilationResult
        import time

        t0 = time.perf_counter()
        # --- your compilation logic here ---
        # e.g. parse AST → emit Triton JIT'd kernel → return wrapper
        # -----------------------------------
        compile_time_ms = (time.perf_counter() - t0) * 1000

        return CompilationResult(
            original=func,
            compiled=compiled_wrapper,   # your callable
            backend="triton",
            compile_time_ms=compile_time_ms,
            speedup=1.0,                 # measured later by CodeGenerator
            validated=False,
            error=None,
        )
```

**Key contract:**
- `can_compile()` must be fast — it's called on every hotspot candidate.
- `compile()` must return a `CompilationResult` where `compiled` is a Python-callable (the kernel wrapper).
- If compilation fails, return `error=str(exc)` and `compiled=func` (identity fallback).

### Step 2: Register in `CodeGenerator`

In `codegen.py`, update the `CodeGenerator.__init__()` and `compile()` method:

```python
class CodeGenerator:
    def __init__(self):
        self.numba = NumbaGenerator()
        self.rust = RustGenerator()
        self.triton = TritonBackend()  # <-- add

    def compile(self, func, test_args=None):
        analyzer = self.analyze(func)

        # Try Numba first (fastest JIT turnaround)
        if self.numba.can_generate(func, analyzer):
            ...

        # Try Triton for GPU-heavy element-wise ops
        if self.triton.can_compile(func, FunctionStats(name="tmp")):
            # Note: FunctionStats is a placeholder here.
            # In practice you may want to pass real stats from Profiler.
            kernel = self.triton.compile(func, ...)
            if kernel.ready:
                return kernel

        # Fallback
        return GeneratedKernel(...)
```

> **Tip:** If your backend needs real `FunctionStats` (e.g. input shapes), pipe them through from `Compiler.compile_hotspots()` instead of creating a dummy.

### Step 3: Add hardware detection

In `core.py`, extend `detect_hardware()`:

```python
def detect_hardware():
    hw = { ..., "triton": False }
    try:
        import triton
        hw["triton"] = True
    except ImportError:
        pass
    return hw
```

Update `GridBackendSelector.THRESHOLDS` if your backend should participate in grid-aware selection:

```python
THRESHOLDS = {
    "numpy": 0,
    "rust_oneshot": 50,
    "rust_persistent": 500,
    "cuda": 1000,
    "triton": 2000,   # only worthwhile for very large tensors
}
```

### Step 4: Add tests

See Section 6 (Testing Guidelines) for a full pattern.

---

## 3. Configuration Reference

All tunable constants live at the top of `core.py` (module-level globals). There is no config file — these are compile-time constants intended for power users.

| Constant | Default | Description |
|----------|---------|-------------|
| `SAMPLE_RATE` | `0.05` | Fraction of calls to profile after the first 1,000. Lower = less overhead, slower to gather stats. |
| `COMPILE_THRESHOLD` | `100` | Minimum call count before a function is considered for compilation. |
| `SPEEDUP_THRESHOLD` | `2.0` | Minimum measured speedup (×) required to trigger a hot-swap. |
| `WARMUP_CALLS` | `10` | Number of initial calls ignored during profiling (reserved for future JIT warmup logic). |

### Backend-specific tuning

| Class / Property | Default | Description |
|------------------|---------|-------------|
| `GridBackendSelector.THRESHOLDS` | see code | Workload-size thresholds for backend selection in grid workloads. |
| `NumbaGenerator` | `cache=True, fastmath=True` | Passed to `numba.njit()`. Change if you need debug symbols or exact floating-point. |
| `RustGenerator` | n/a | Currently stubbed. Will accept `opt-level` and `target-cpu` flags in v2. |

### Environment variables (runtime)

The codebase currently does not read env vars, but you can add guards during development:

```python
# In your driver script, before compiler.install()
import agentic_compiler.core as acc
acc.SAMPLE_RATE = float(os.environ.get("AGENTIC_SAMPLE_RATE", 0.05))
acc.SPEEDUP_THRESHOLD = float(os.environ.get("AGENTIC_SPEEDUP_THRESHOLD", 2.0))
```

---

## 4. How A/B Validation Works

Validation is the safety net. It guarantees the compiled kernel is **semantically equivalent** to the original Python function before it ever touches production code.

### The `CodeGenerator.validate()` algorithm

```python
def validate(kernel, original, test_args, trials=5):
    if kernel.source_language == "python":
        return True   # identity — nothing to validate

    for _ in range(trials):
        expected = original(*test_args)
        actual   = kernel.compiled(*test_args)
        if not _outputs_equal(expected, actual):
            return False
    return True
```

### `_outputs_equal` semantics

| Type | Comparison |
|------|------------|
| `np.ndarray` (both) | `np.allclose(a, b, rtol=1e-3, atol=1e-5)` |
| numpy scalar vs Python scalar | `abs(float(a) - float(b)) < 1e-5` |
| `list` / `tuple` | Recursive length + element-wise `_outputs_equal` |
| Exact type mismatch | `False` |
| Everything else | `a == b` |

### Why it matters

- **Numba `njit` with `fastmath=True`** can reorder operations, causing tiny bit-level differences. The tolerance accounts for this.
- **Dictionary ordering** is undefined in some Python versions; recursive equality does not ignore order by default.
- **Shape mismatches** (e.g. `(N,)` vs `(N, 1)`) will fail validation even if values are close.

### Validation failures are **fatal**

If validation fails, the kernel is **never deployed**. The `CompilationResult` carries `validated=False` and the pipeline stops. There is no "deploy anyway" switch — if you need one for experimentation, set it manually:

```python
# DEV ONLY — skip validation
kernel.error = None  # mark ready
setattr(my_module, "my_func", kernel.compiled)
```

---

## 5. Troubleshooting Guide

### "No functions are being compiled"

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `compile_hotspots()` returns empty list | Call count < `COMPILE_THRESHOLD` (100) | Run more iterations, or lower threshold |
| Profiler report is empty | `sample_rate=0` or function not decorated | Ensure `compiler.install()` ran, or use `@profiler.watch` |
| Functions are in stdlib / numpy | `install()` skips modules starting with banned prefixes | Manually decorate with `@profiler.watch` |
| Numba not found | `numba` extra not installed | `pip install -e ".[numba]"` |

### "Compilation succeeds but no hot-swap happens"

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Log says "speedup 1.3× < 2.0× threshold" | Function is already fast, or Numba overhead dominates for small inputs | Increase input size; lower `SPEEDUP_THRESHOLD` |
| `validated=False` | Output mismatch between original and compiled | Check dtypes, NaNs, or dict ordering in return value |
| `kernel.error` is set | Numba couldn't compile the function (e.g. unsupported Python feature) | Rewrite function to avoid lists/dicts inside loops |

### "Hot-swap broke my program"

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Wrong results after swap | Validation passed on synthetic args, but real args differ | Improve `_generate_test_args()` to match production shapes |
| `AttributeError` after swap | Compiled kernel lacks an attribute the original had | Ensure kernel wrapper copies `__annotations__`, `__doc__`, `__module__` |
| Can't restore | `_agentic_original` was garbage collected or overwritten | Call `compiler.restore(key)` immediately; it looks up `self._originals` first |

### "Rust backend always fails"

This is expected in v1. The `RustGenerator.generate()` currently returns:

```python
error="Procedural Rust generation v2 not yet implemented. Use manual Rust kernels."
```

To use Rust today, pre-compile a `.so` and place it at `nerve/target/release/libjepa_kernel.so`. The hardware detector will pick it up.

### Numba-specific issues

| Error | Cause | Fix |
|-------|-------|-----|
| `NumbaNotImplementedError` | Unsupported Python construct (e.g. `**kwargs`, closures) | Refactor to plain args, top-level function |
| `TypingError` | Type inference failed | Add type hints, or initialize arrays with explicit dtype |
| Slow first call | LLVM compilation on first invocation | Warmup inside `generate()` is already done; increase `trials` if needed |

### Debug workflow

```python
from agentic_compiler import Compiler
compiler = Compiler()
compiler.install()

# ... run your code ...

for result in compiler.compiled.values():
    print(result.backend, result.validated, result.error, result.speedup)
```

---

## 6. Testing Guidelines for New Backends

Every backend must have tests covering **compilation**, **validation**, **speedup measurement**, and **hot-swap correctness**. Use the existing test suite in `tests/test_core.py` as a template.

### Minimum test checklist

```python
# tests/test_triton.py  (example)
import numpy as np
import pytest
from agentic_compiler.codegen import PythonAnalyzer

class TestTritonBackend:
    def test_available_detects_triton(self):
        from agentic_compiler.triton_backend import TritonBackend
        gen = TritonBackend()
        assert isinstance(gen._available, bool)

    def test_can_compile_elementwise(self):
        from agentic_compiler.triton_backend import TritonBackend
        from agentic_compiler.core import FunctionStats

        def elementwise(arr):
            return arr * 2 + 1

        backend = TritonBackend()
        stats = FunctionStats(name="test", calls=200)
        analyzer = PythonAnalyzer().analyze(elementwise)
        # This may be False if Triton isn't installed — that's OK
        result = backend.can_compile(elementwise, stats)
        assert isinstance(result, bool)

    def test_compile_returns_kernel(self):
        from agentic_compiler.triton_backend import TritonBackend
        from agentic_compiler.core import FunctionStats

        def elementwise(arr):
            return arr * 2 + 1

        backend = TritonBackend()
        stats = FunctionStats(name="test", calls=200)
        result = backend.compile(elementwise, stats)
        assert result.backend == "triton"
        # If Triton unavailable, error should explain why
        if not result.error:
            assert result.compiled is not None

    def test_hot_swap_integrity(self):
        """End-to-end: compile → validate → measure → swap → restore."""
        from agentic_compiler import Compiler
        import types, sys

        def my_kernel(arr):
            return arr * 3

        mod = types.ModuleType("triton_e2e")
        mod.__dict__["my_kernel"] = my_kernel
        sys.modules["triton_e2e"] = mod

        compiler = Compiler()
        result = compiler.hot_swap(my_kernel, module=mod, attr_name="my_kernel")

        assert isinstance(result, CompilationResult)
        if result.error is None:
            assert result.validated in (True, False)
            assert mod.my_kernel is not my_kernel or result.speedup < 2.0
            # Restore
            ok = compiler.restore("triton_e2e.my_kernel")
            assert ok is True
            assert mod.my_kernel(5) == 15   # original logic
```

### CI integration

Add your backend to `.github/workflows/ci.yml`:

```yaml
      - name: Install Triton
        run: pip install triton

      - name: Run tests with Triton
        run: pytest tests/test_triton.py -v
```

### Mocking unavailable hardware

Use `unittest.mock.patch` to test backend selection logic even when the library isn't installed:

```python
from unittest.mock import patch

@patch("agentic_compiler.core.HARDWARE", {
    "numpy": True, "numba": True, "triton": True,
    "rust_persistent": False, "rust_oneshot": False, "cuda": False,
})
def test_grid_selector_prefers_triton_for_large_tensors():
    from agentic_compiler.core import GridBackendSelector
    assert GridBackendSelector.select(5000) == "triton"
```

---

## File Layout

```
agentic_compiler/
├── __init__.py      # Public API exports
├── core.py          # Profiler, Compiler, hot-swap, hardware detection
├── codegen.py       # AST analyzer, Numba/Rust generators, validator, deployer
tests/
└── test_core.py     # Unit + integration tests
```

---

## Quick Reference: Pipeline Entry Points

| Task | Entry Point | File |
|------|-------------|------|
| Add profiler to function | `@profiler.watch` | `core.py` |
| Auto-profile all user modules | `compiler.install()` | `core.py` |
| Compile top-N hotspots | `compiler.compile_hotspots(n=5)` | `core.py` |
| Compile one function manually | `compiler.hot_swap(func, mod, name)` | `core.py` |
| Analyze AST | `PythonAnalyzer().analyze(func)` | `codegen.py` |
| Generate Numba kernel | `NumbaGenerator().generate(func, analyzer, args)` | `codegen.py` |
| Validate kernel | `CodeGenerator().validate(kernel, orig, args)` | `codegen.py` |
| Measure speedup | `CodeGenerator().measure_speedup(kernel, orig, args)` | `codegen.py` |
| Deploy (hot-swap) | `CodeGenerator().deploy(kernel, mod, name)` | `codegen.py` |
| Restore original | `compiler.restore(key)` | `core.py` |

---

*Last updated: 2026-05-23*
