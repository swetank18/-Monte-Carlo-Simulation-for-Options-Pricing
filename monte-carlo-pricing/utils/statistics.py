"""Monte Carlo statistics utilities."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def mc_stats_from_payoffs(
    payoffs: np.ndarray,
    r: float,
    t: float,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute Monte Carlo price, standard error, and 95% confidence interval.
    Std error = exp(-rT) * std(payoffs) / sqrt(N)
    CI = price +/- 1.96 * std_error
    """
    n = payoffs.size
    discounted = np.exp(-r * t) * payoffs
    price = float(np.mean(discounted))
    std_error = float(math.exp(-r * t) * np.std(payoffs, ddof=1) / math.sqrt(n))
    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error
    return price, std_error, (ci_low, ci_high)
