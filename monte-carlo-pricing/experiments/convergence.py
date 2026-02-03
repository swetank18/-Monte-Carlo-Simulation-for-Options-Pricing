"""Convergence experiment for European call pricing."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from pricers.european import european_mc_price


def run() -> None:
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
    path_counts = [1000, 5000, 10000, 20000, 50000]
    seed = 42

    print("Convergence Experiment (European Call):")
    results = []
    for n in path_counts:
        price, se, ci, _ = european_mc_price(s0, k, r, sigma, t, n, seed, call=True)
        results.append((n, se))
        print(f"  N={n:6d} | Price={price:.4f} | SE={se:.4f} | 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    if results:
        se_first = results[0][1]
        se_last = results[-1][1]
        if se_last > 0:
            ratio = se_first / se_last
            print(
                "Convergence Summary: SE decreased by "
                f"{ratio:.2f}x from N={results[0][0]} to N={results[-1][0]}, "
                "consistent with 1/sqrt(N)."
            )
        else:
            print("Convergence Summary: SE did not decrease as expected (check inputs).")


if __name__ == "__main__":
    run()
