# SUPERINSTANCE.md — Agentic Compiler Vision

> The agentic compiler is the **muscle memory** of the SuperInstance ecosystem.

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SuperInstance Ecosystem                          │
│                                                                     │
│   SIA (Self-Improving Agent)                                       │
│     ├── t-minus (countdown orchestrator)                           │
│     ├── Self-Improving Band (ensemble of specialized agents)       │
│     │     ├── Band Crates (modular agent skills)                   │
│     │     └── agentic-compiler ← YOU ARE HERE                     │
│     │           ├── Profiles hot paths                             │
│     │           ├── Compiles to Numba/Rust/CUDA                    │
│     │           └── Hot-swaps optimized versions                   │
│     └── construct-core (three-layer hardware abstraction)          │
│           ├── Layer 0: Lookup tables (ESP32)                       │
│           ├── Layer 1: Sync skills (RPi)                           │
│           └── Layer 2: Async constructs (Jetson+)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Role in the Ecosystem

The agentic compiler sits at the intersection of **observation** and **optimization**. It watches the fleet's code run, learns which paths matter, and makes them faster — automatically.

### SIA Integration

The **Self-Improving Agent** uses the compiler as its optimization engine:

1. **SIA identifies** which skills are slow (via t-minus timing data)
2. **Compiler profiles** those skills during real usage
3. **Compiler compiles** hot paths to faster backends
4. **SIA validates** that optimized skills produce identical results
5. **Compiler hot-swaps** the optimized version into the running agent

### t-minus Integration

**t-minus** is the countdown/orchestration layer. It provides:

- **Timing telemetry**: Exact latency measurements for each agent operation
- **Deadline pressure**: When an operation is near its deadline, the compiler can prioritize it
- **Trigger signals**: t-minus can request immediate compilation of a critical path

```python
# Example: t-minus feeds timing data to the compiler
from agentic_compiler import Compiler

compiler = Compiler()

# t-minus detects that deliberation is slow
compiler.hot_swap(
    func=swarm_deliberate,
    module=swarm_module,
    attr_name="deliberate",
)
```

### Self-Improving Band Integration

The **Band** is an ensemble of specialized agents. Each band member has skills that can be compiled:

```
Band Member → Skills → Hot Paths → Compiled Kernels
    ↓             ↓          ↓             ↓
  forgemaster   compile   profile    optimized skill
  cocapn-plato  compile   profile    optimized skill
  tensor-spline compile   profile    optimized skill
```

The compiler treats each band member's skills as compilation targets. Over time, the band's collective performance improves as hot paths are compiled.

### Band Crates Integration

**Band crates** are modular, publishable skill packages. The compiler integrates by:

1. **Profiling crate functions**: Identifying which exported functions are hot
2. **Compiling crate kernels**: Generating optimized versions for each hardware tier
3. **Embedding compiled artifacts**: Storing compiled kernels alongside the crate source
4. **Tier-aware deployment**: Selecting the right compiled kernel based on target hardware

```python
# Example: Compiling a band crate for deployment
from agentic_compiler import Compiler

compiler = Compiler()

# Profile the crate's core function during development
@compiler.profiler.watch
def process_sensor_grid(rooms, weights):
    # ... hot path ...
    pass

# After profiling, compile for the target tier
result = compiler.compile_function(process_sensor_grid)
print(f"Compiled to {result.backend}, {result.speedup:.1f}× speedup")
```

## Architecture in Context

```
                    ┌─────────────┐
                    │     SIA     │  ← Self-improving agent
                    │  controller │
                    └──────┬──────┘
                           │ identifies slow skills
                           ▼
                    ┌─────────────┐
                    │   t-minus   │  ← Timing & orchestration
                    │  telemetry  │
                    └──────┬──────┘
                           │ feeds timing data
                           ▼
┌──────────────────────────────────────────────────────┐
│                Agentic Compiler                       │
│                                                      │
│  Profiler ──→ Analyzer ──→ CodeGenerator ──→ Deploy │
│  (watch)      (rank)      (compile)        (swap)   │
│                                                      │
│  Backends: Numba | Rust | CUDA | Python (fallback)  │
└──────────────────────────┬───────────────────────────┘
                           │ optimized kernels
                           ▼
                    ┌─────────────┐
                    │ Band Crates │  ← Modular skills
                    │ (compiled)  │
                    └──────┬──────┘
                           │ deployed to
                           ▼
                    ┌─────────────┐
                    │   Fleet     │  ← Rooms, devices, agents
                    │  hardware   │
                    └─────────────┘
```

## The Self-Improvement Loop

The compiler closes the self-improvement loop:

1. **Observe**: Profile real-world usage patterns
2. **Learn**: Identify which functions matter most
3. **Optimize**: Compile to faster backends
4. **Validate**: A/B test for correctness
5. **Deploy**: Hot-swap optimized versions
6. **Measure**: Track actual speedup
7. **Repeat**: The fleet gets faster over time

This is the foundation of **evolutionary runtime optimization** — the fleet's compiled skills become a library of optimized, verified routines.

## Future Directions

- **Distributed compilation**: Share compiled kernels across fleet members
- **Adaptive thresholds**: Auto-tune compilation thresholds based on fleet load
- **Cross-crate optimization**: Compile across crate boundaries for whole-program optimization
- **Hardware specialization**: Generate hardware-specific kernels for each tier (ESP32, RPi, Jetson)

---

*Maintained by [SuperInstance](https://github.com/SuperInstance)*
*Part of the Self-Improving Agent ecosystem*
