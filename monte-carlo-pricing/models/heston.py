"""Heston stochastic volatility model (Euler discretization)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def simulate_heston_terminal(
    s0: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: Optional[int],
) -> np.ndarray:
    """
    Simulate terminal prices under Heston using Euler discretization.
    Variance is floored at zero to avoid negative values.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    dt = t / n_steps

    s = np.full(n_paths, s0, dtype=float)
    v = np.full(n_paths, v0, dtype=float)

    for _ in range(n_steps):
        z1 = rng.standard_normal(size=n_paths)
        z2 = rng.standard_normal(size=n_paths)
        w2 = rho * z1 + math.sqrt(1.0 - rho ** 2) * z2

        v = v + kappa * (theta - v) * dt + xi * np.sqrt(np.maximum(v, 0.0)) * math.sqrt(dt) * w2
        v = np.maximum(v, 0.0)

        s = s * np.exp((r - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0.0)) * math.sqrt(dt) * z1)

    return s
