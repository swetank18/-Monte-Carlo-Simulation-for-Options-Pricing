"""Asian option pricer (arithmetic-average call)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from models.gbm import simulate_gbm_paths
from utils.statistics import mc_stats_from_payoffs


def asian_arithmetic_call_mc_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: Optional[int],
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Arithmetic-average Asian call priced via Monte Carlo time-stepping.
    Path-dependent because payoff depends on the entire average path.
    """
    paths = simulate_gbm_paths(s0, r, sigma, t, n_paths, n_steps, seed)
    avg_price = np.mean(paths, axis=1)
    payoffs = np.maximum(avg_price - k, 0.0)
    return mc_stats_from_payoffs(payoffs, r, t)
