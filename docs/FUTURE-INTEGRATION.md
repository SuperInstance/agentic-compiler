# Future Integration: agentic-compiler

## Current State
Runtime-adaptive compilation for Python — automatically profiles hot functions, compiles them to optimized backends (Numba, Rust, CUDA), and hot-swaps at runtime with automatic rollback. Published on PyPI. Profiles after 100+ calls, verifies identical output, measures speedup, and silently replaces slow code.

## Integration Opportunities

### With construct-core
The compiler becomes construct-core's `SELF_IMPROVEMENT` skill at Layer 2 (AsyncConstruct). It profiles which `query_owned()` calls on PiConstruct are slow, identifies hot paths via 5% statistical sampling, compiles optimized lookup tables, and installs them as Layer 0 `query_lookup()` overrides. The hardware tiering means one optimization pipeline can descend through all three layers: compile a Layer 2 async handler → Layer 1 sync skill → Layer 0 lookup table for ESP32.

### With forgemaster
The compiler's CUDA backend (currently a stub) becomes the Forgemaster's runtime optimizer. When the Forgemaster detects that a GPU simulation kernel is underperforming, the agentic-compiler generates PTX-specialized replacements and hot-swaps them — no downtime, automatic verification.

### With room-as-codespace
Each room's Codespace runs the compiler as a background daemon. As the room's ensign processes queries, the compiler profiles and optimizes in real-time. Room A might optimize its Kalman filter; Room B might optimize its music algebra. The optimizations are stored as compiled skills in the room's skill registry, available for next boot.

## Dormant Ideas Now Unlockable
The Rust and CUDA backends were stubs because there was no deployment target. Now construct-core provides the target (three hardware tiers), forgemaster provides the GPU simulation context, and room-as-codespace provides the lifecycle management. The compiler can now produce real optimized code for real hardware targets.

## Potential in Mature Systems
The agentic-compiler enables **evolutionary runtime optimization** across the fleet. Rooms compete for compute resources; the compiler ensures each room's hot paths are as fast as possible. Over time, the fleet's compiled skills become a library of optimized, verified routines — the "muscle memory" of the collective.

## Cross-Pollination Ideas
- **ptx-bench**: Benchmark results guide the compiler's CUDA optimization decisions
- **SuperInstance-Starter-Agent**: Muscle memory triggers feed into the compiler
- **tile-compiler**: Tile compilation is a specialization of agentic compilation for game strategies

## Dependencies for Next Steps
- Rust and CUDA backend implementations (currently stubs)
- construct-core SkillSpec compilation format
- Integration with forgemaster's PTX generation pipeline
