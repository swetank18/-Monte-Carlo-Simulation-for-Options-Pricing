"""Antithetic variates for European option pricing."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from utils.statistics import mc_stats_from_payoffs


def antithetic_european_mc_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    seed: Optional[int],
    call: bool = True,
) -> Tuple[float, float, Tuple[float, float], float]:
    """
    Antithetic variates Monte Carlo price for a European call/put.
    Returns (price, std_error, ci, variance_reduction_factor) vs naive MC
    using the same set of draws.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    half = n_paths // 2
    z = rng.standard_normal(size=half)
    z_full = np.concatenate([z, -z])
    if n_paths % 2 == 1:
        z_full = np.concatenate([z_full, rng.standard_normal(size=1)])

    drift = (r - 0.5 * sigma ** 2) * t
    diffusion = sigma * math.sqrt(t) * z_full
    s_t = s0 * np.exp(drift + diffusion)

    if call:
        payoffs = np.maximum(s_t - k, 0.0)
    else:
        payoffs = np.maximum(k - s_t, 0.0)

    # Antithetic pair average
    pair_count = half
    pair_avg = 0.5 * (payoffs[:pair_count] + payoffs[pair_count:2 * pair_count])
    if n_paths % 2 == 1:
        pair_avg = np.concatenate([pair_avg, payoffs[-1:]])

    price, std_error, ci = mc_stats_from_payoffs(pair_avg, r, t)

    # Naive variance on identical path set
    naive_price, naive_se, naive_ci = mc_stats_from_payoffs(payoffs, r, t)
    _ = (naive_price, naive_se, naive_ci)
    var_reduction = float(np.var(payoffs, ddof=1) / np.var(pair_avg, ddof=1)) if np.var(pair_avg, ddof=1) > 0 else float("inf")
    return price, std_error, ci, var_reduction
