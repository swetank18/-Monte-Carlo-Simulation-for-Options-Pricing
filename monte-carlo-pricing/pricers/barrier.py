"""Barrier option pricers."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from models.gbm import simulate_gbm_paths
from utils.statistics import mc_stats_from_payoffs


def up_and_out_call_mc_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: Optional[int],
    barrier: float,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Up-and-out European call priced via Monte Carlo time-stepping.
    If the path crosses the barrier, the option is knocked out.
    This is path-dependent because barrier events depend on the full trajectory.
    """
    paths = simulate_gbm_paths(s0, r, sigma, t, n_paths, n_steps, seed)
    max_s = np.max(paths, axis=1)
    knocked_out = max_s >= barrier
    s_t = paths[:, -1]
    payoffs = np.where(knocked_out, 0.0, np.maximum(s_t - k, 0.0))
    return mc_stats_from_payoffs(payoffs, r, t)
