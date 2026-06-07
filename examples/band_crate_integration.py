"""
Integration Example: agentic-compiler + Band Crates

Demonstrates how to profile, compile, and hot-swap functions
from a band crate for fleet deployment.

Run: python3 examples/band_crate_integration.py
"""

import numpy as np
from agentic_compiler import Compiler, Profiler, GridBackendSelector


# ── Simulated Band Crate Function ──────────────────────────
# In a real scenario, this would be imported from a band crate.


def kalman_filter_update(measurement: np.ndarray, state: np.ndarray,
                         covariance: np.ndarray) -> tuple:
    """Simulated Kalman filter update — a common hot path in fleet rooms."""
    n = state.shape[0]
    # Simplified Kalman update
    kalman_gain = covariance @ np.eye(n) / (np.eye(n) @ covariance @ np.eye(n) + np.eye(n))
    new_state = state + kalman_gain @ (measurement - state)
    new_cov = (np.eye(n) - kalman_gain @ np.eye(n)) @ covariance
    return new_state, new_cov


def music_algebra_transform(signal: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Simulated music algebra transformation — another band crate hot path."""
    total = np.zeros_like(signal)
    for i in range(len(weights)):
        total += weights[i] * signal ** (i + 1)
    return total


# ── Integration Pattern ────────────────────────────────────


def main():
    # 1. Set up the compiler
    compiler = Compiler()
    profiler = Profiler(sample_rate=0.1)

    # 2. Watch band crate functions
    kalman_watched = profiler.watch(kalman_filter_update)
    music_watched = profiler.watch(music_algebra_transform)

    # 3. Simulate fleet usage (in production, this comes from real room queries)
    print("Simulating fleet usage...")
    for _ in range(150):
        state = np.random.randn(16).astype(np.float32)
        cov = np.eye(16, dtype=np.float32) * 0.1
        meas = np.random.randn(16).astype(np.float32)
        kalman_watched(meas, state, cov)

    for _ in range(150):
        signal = np.random.randn(100).astype(np.float32)
        weights = np.random.randn(5).astype(np.float32)
        music_watched(signal, weights)

    # 4. Check profiling results
    print("\n" + profiler.report())

    # 5. Select backend for deployment tier
    print("\n" + GridBackendSelector.report())
    backend = GridBackendSelector.select(n_rooms=750)
    print(f"\nSelected backend for 750 rooms: {backend}")

    # 6. Compile hotspots
    print("\nCompiling hotspots...")
    results = compiler.compile_hotspots(top_n=2)
    for r in results:
        status = "✅" if r.validated else "❌"
        print(f"  {status} {r.backend}: speedup={r.speedup:.1f}× "
              f"validated={r.validated} "
              f"error={r.error or 'none'}")

    print("\nDone. The compiler will continue optimizing as the fleet runs.")


if __name__ == "__main__":
    main()
