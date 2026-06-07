"""
Integration Example: agentic-compiler + SIA + t-minus

Shows how the agentic compiler connects to the Self-Improving Agent
and t-minus orchestration layer.

This is a conceptual example — SIA and t-minus are simulated.

Run: python3 examples/sia_integration.py
"""

import types
import sys
import time
import numpy as np
from agentic_compiler import Compiler, CompilationResult


# ── Simulated SIA Skill ────────────────────────────────────


def swarm_deliberate(observations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Simulated swarm deliberation — a hot path in SIA."""
    total = np.zeros_like(observations)
    for i in range(len(weights)):
        total += weights[i] * observations
    return total / len(weights)


# ── Simulated t-minus Telemetry ────────────────────────────


class TMinusTelemetry:
    """Simulated t-minus timing data."""

    def __init__(self):
        self.timings: dict[str, list[float]] = {}

    def record(self, func_name: str, elapsed_ms: float):
        if func_name not in self.timings:
            self.timings[func_name] = []
        self.timings[func_name].append(elapsed_ms)

    def get_slow_functions(self, threshold_ms: float = 1.0) -> list[str]:
        """Return functions that average above threshold."""
        slow = []
        for name, times in self.timings.items():
            avg = sum(times) / len(times)
            if avg > threshold_ms:
                slow.append(name)
        return slow


# ── Integration Pattern ────────────────────────────────────


def main():
    # 1. Create a fake module to host the skill
    mod = types.ModuleType("sia_skills")
    mod.swarm_deliberate = swarm_deliberate
    sys.modules["sia_skills"] = mod

    # 2. Set up t-minus telemetry
    tminus = TMinusTelemetry()

    # 3. Run the skill and collect timing data
    print("Collecting t-minus telemetry...")
    for _ in range(200):
        obs = np.random.randn(100).astype(np.float32)
        weights = np.random.randn(10).astype(np.float32)

        t0 = time.perf_counter()
        result = mod.swarm_deliberate(obs, weights)
        elapsed = (time.perf_counter() - t0) * 1000
        tminus.record("swarm_deliberate", elapsed)

    # 4. Check which functions are slow
    slow = tminus.get_slow_functions(threshold_ms=0.01)
    print(f"Slow functions detected by t-minus: {slow}")

    # 5. Ask the compiler to optimize
    compiler = Compiler()
    result: CompilationResult = compiler.hot_swap(
        func=swarm_deliberate,
        module=mod,
        attr_name="swarm_deliberate",
    )

    print(f"\nCompilation result:")
    print(f"  Backend: {result.backend}")
    print(f"  Speedup: {result.speedup:.1f}×")
    print(f"  Validated: {result.validated}")
    print(f"  Error: {result.error or 'none'}")

    # 6. Run again and measure improvement
    if result.validated and result.speedup > 1.0:
        print("\nPost-optimization telemetry:")
        times = []
        for _ in range(100):
            obs = np.random.randn(100).astype(np.float32)
            weights = np.random.randn(10).astype(np.float32)
            t0 = time.perf_counter()
            mod.swarm_deliberate(obs, weights)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  Average: {sum(times)/len(times):.3f}ms")

    # 7. Cleanup
    compiler.restore("sia_skills.swarm_deliberate")
    del sys.modules["sia_skills"]


if __name__ == "__main__":
    main()
