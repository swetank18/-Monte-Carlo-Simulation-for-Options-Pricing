"""Control variate methods for European option pricing."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from analytics.black_scholes import black_scholes_delta, black_scholes_price
from utils.statistics import mc_stats_from_payoffs


def control_variate_european_mc_price(
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
    Control variate estimator using a Black-Scholes linear control.

    X = BS_price + BS_delta * (exp(-rT) * S_T - S_0)
    E[X] = BS_price

    Returns (cv_price, cv_std_error, cv_ci, variance_reduction_factor).
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    z = rng.standard_normal(size=n_paths)

    drift = (r - 0.5 * sigma ** 2) * t
    diffusion = sigma * math.sqrt(t) * z
    s_t = s0 * np.exp(drift + diffusion)

    if call:
        payoffs = np.maximum(s_t - k, 0.0)
    else:
        payoffs = np.maximum(k - s_t, 0.0)

    y = np.exp(-r * t) * payoffs
    bs_price = black_scholes_price(s0, k, r, sigma, t, call=call)
    bs_delta = black_scholes_delta(s0, k, r, sigma, t, call=call)

    x = bs_price + bs_delta * (np.exp(-r * t) * s_t - s0)

    cov = np.cov(y, x, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    b_opt = cov / var_x if var_x > 0 else 0.0

    y_cv = y - b_opt * (x - bs_price)
    cv_price = float(np.mean(y_cv))
    cv_std_error = float(np.std(y_cv, ddof=1) / math.sqrt(y_cv.size))
    cv_ci = (cv_price - 1.96 * cv_std_error, cv_price + 1.96 * cv_std_error)

    var_reduction = float(np.var(y, ddof=1) / np.var(y_cv, ddof=1)) if np.var(y_cv, ddof=1) > 0 else float("inf")
    return cv_price, cv_std_error, cv_ci, var_reduction
