"""GBM simulation utilities (exact solution)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def simulate_gbm_terminal(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    seed: Optional[int],
) -> np.ndarray:
    """
    Exact GBM terminal simulation:
    S_T = S_0 * exp((r - 0.5*sigma^2) * T + sigma * sqrt(T) * Z)
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    z = rng.standard_normal(size=n_paths)
    drift = (r - 0.5 * sigma ** 2) * t
    diffusion = sigma * math.sqrt(t) * z
    return s0 * np.exp(drift + diffusion)


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: Optional[int],
) -> np.ndarray:
    """
    Simulate full GBM paths using exact per-step update.
    Only needed for path-dependent options.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    dt = t / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)

    z = rng.standard_normal(size=(n_paths, n_steps))
    increments = drift + vol * z
    log_s = math.log(s0) + np.cumsum(increments, axis=1)
    return np.exp(log_s)
