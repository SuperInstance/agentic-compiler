# agentic-compiler

Runtime-adaptive compilation for Python. Automatically profiles hot functions, compiles them to optimized backends (Numba, Rust, CUDA), and hot-swaps them at runtime — no human intervention required.

## The Pitch

You write plain Python. The compiler watches. After 100+ calls it notices your function is slow, compiles it with Numba, verifies the output is identical, measures the speedup, and silently replaces the slow version — all while your program is still running. If the compiled version is ever wrong, it rolls back in one call.

## Installation

```bash
pip install agentic-compiler
```

With Numba support (recommended):
```bash
pip install agentic-compiler[numba]
```

## Quickstart

```python
from agentic_compiler import Compiler
import numpy as np

compiler = Compiler()
compiler.install()  # monkey-patch all hot paths

# Your normal code — the compiler is watching
def slow_sum(arr):
    total = 0.0
    for i in range(len(arr)):
        total += arr[i] * arr[i]
    return total

# Run it enough times to trigger compilation
for _ in range(200):
    slow_sum(np.random.randn(1000))

# The compiler will auto-compile and hot-swap if speedup > 2x
print(compiler.profiler.report())
```

### Decorator-style profiling

```python
from agentic_compiler import Profiler

profiler = Profiler(sample_rate=0.05)

@profiler.watch
def my_hot_function(x):
    return np.sum(x ** 2)
```

### Manual hot-swap

```python
from agentic_compiler import Compiler

compiler = Compiler()

result = compiler.hot_swap(my_slow_function)
print(f"Speedup: {result.speedup:.1f}x, Validated: {result.validated}")

# Rollback if something goes wrong
compiler.restore()
```

### One-shot compilation

```python
from agentic_compiler import ast_to_numba

kernel = ast_to_numba(my_function)
if kernel.ready:
    print(f"Compiled to {kernel.backend} in {kernel.compile_time_ms:.1f}ms")
```

## Grid-Aware Backend Selection

The `GridBackendSelector` automatically picks the right backend based on workload size:

| Workload Size | Backend | Why |
|--------------|---------|-----|
| n < 50 | numpy | ctypes overhead dominates |
| 50–500 | rust_oneshot | medium arrays |
| 500+ | rust_persistent | zero-copy, weights in Rust |
| 1000+ (GPU available) | cuda | maximum parallelism |

```python
from agentic_compiler import GridBackendSelector

backend = GridBackendSelector.select(n_rooms=750)
print(GridBackendSelector.report())
```

## API Reference

### `Compiler`

- `install(module_name=None)` — monkey-patch functions for profiling
- `uninstall()` — restore originals
- `compile_hotspots(top_n=5)` — auto-compile top-N hot functions
- `compile_function(func, module, attr_name)` — manual compile + swap
- `hot_swap(func, module, attr_name, backend)` — compile + swap in one call
- `restore(key=None)` — rollback to original function

### `Profiler`

- `watch(func)` — decorator to profile a function
- `get_hotspots(top_n=10)` — ranked list by optimization potential
- `report()` — human-readable profiling summary

### `CodeGenerator`

- `compile(func)` — compile to best available backend
- `validate(kernel, original, test_args)` — A/B test for correctness
- `measure_speedup(kernel, original, test_args)` — benchmark speedup
- `deploy(kernel, module, attr_name)` — hot-swap into module

### `GeneratedKernel`

- `ready` — bool, True if compilation succeeded
- `compiled` — the compiled callable
- `backend` — "numba", "rust", or "python"
- `compile_time_ms` — how long compilation took
- `speedup` — measured speedup vs original

## Architecture

```
Profiler → Analyzer → CodeGenerator → Validator → Deployer
 (watch)   (rank)    (compile)     (A/B test)  (hot-swap)
```

1. **Profiler** samples 5% of calls, tracks timing and input shapes
2. **Analyzer** scores functions for Numba vs Rust suitability via AST
3. **CodeGenerator** compiles to best backend, with fallback to identity
4. **Validator** runs A/B tests to verify correctness
5. **Deployer** hot-swaps if speedup exceeds 2× threshold

### Pipeline Details

```
                    ┌────────────┐
                    │  Profiler   │  Samples 5% of calls
                    │  (watch)    │  Tracks timing & input shapes
                    └─────┬──────┘
                          │ Ranked by optimization_potential = calls × avg_time²
                          ▼
                    ┌────────────┐
                    │  Analyzer   │  AST walk via PythonAnalyzer
                    │  (rank)     │  Scores: Numba (+numpy, +loops) vs Rust (+dicts, +strings)
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐
                    │CodeGenerator│  Tries backends in order: Numba → Rust → Python (identity)
                    │ (compile)   │  Each backend decides if it can handle the function
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐
                    │ Validator   │  A/B test: original vs compiled (5 trials)
                    │(A/B test)   │  Compares with tolerance (1e-3 rtol, 1e-5 atol)
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐
                    │  Deployer   │  setattr() hot-swap if speedup ≥ 2.0×
                    │(hot-swap)   │  Preserves _agentic_original for rollback
                    └────────────┘
```

### Backend Selection

| Backend | When Used | Requirements |
|---------|-----------|-------------|
| **Numba** | Numeric loops, numpy ops | `pip install agentic-compiler[numba]` |
| **Rust** | Dict-heavy, string processing | `libjepa_kernel.so` in `nerve/target/release/` |
| **CUDA** | Large parallel workloads (n>1000) | `libcudart.so` available |
| **Python** | Fallback — no speedup | Always available |

## Ecosystem Integration

The agentic compiler integrates with the [SuperInstance](https://github.com/SuperInstance) ecosystem:

| Component | Relationship |
|-----------|-------------|
| **SIA** (Self-Improving Agent) | Uses the compiler as its optimization engine — identifies slow skills and compiles them |
| **t-minus** | Feeds timing telemetry to the compiler, triggering compilation when operations near deadlines |
| **Self-Improving Band** | Each band member's skills are compilation targets; the band gets faster over time |
| **Band Crates** | Modular skill packages whose exported functions can be profiled and compiled |

See [`SUPERINSTANCE.md`](./SUPERINSTANCE.md) for the full vision and architecture.

### Quick Integration Example

```python
# Profile a band crate function
from agentic_compiler import Compiler
import numpy as np

compiler = Compiler()

# In production, this comes from t-minus telemetry
# identifying swarm_deliberate() as a slow path
result = compiler.hot_swap(swarm_deliberate, module=my_module, attr_name="deliberate")

if result.validated:
    print(f"Optimized: {result.speedup:.1f}× via {result.backend}")
```

See [`examples/`](./examples/) for complete integration patterns.

## Testing

```bash
pip install -e ".[numba]"
pytest
```

Tests cover profiler instrumentation, backend selection, hot-swap correctness, and Numba integration (mocked when unavailable).

## Related Repos

- **[forgemaster](https://github.com/SuperInstance/forgemaster)** — Forge orchestration (uses agentic compilation for swarm deliberation)
- **[cocapn-plato](https://github.com/SuperInstance/cocapn-plato)** — Fleet PLATO integration
- **[tensor-spline](https://github.com/SuperInstance/tensor-spline)** — Compressed neural network layers

## License

MIT
