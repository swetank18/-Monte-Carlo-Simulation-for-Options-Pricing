"""Black-Scholes analytics for European options."""
from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    call: bool = True,
) -> float:
    """Black-Scholes price for a European call/put."""
    if t <= 0:
        intrinsic = max(0.0, s0 - k) if call else max(0.0, k - s0)
        return intrinsic
    if sigma <= 0:
        forward = s0 * math.exp(r * t)
        intrinsic = max(0.0, forward - k) if call else max(0.0, k - forward)
        return intrinsic * math.exp(-r * t)

    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    if call:
        return s0 * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)
    return k * math.exp(-r * t) * _norm_cdf(-d2) - s0 * _norm_cdf(-d1)


def black_scholes_delta(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    call: bool = True,
) -> float:
    """Black-Scholes Delta for a European call/put."""
    if t <= 0 or sigma <= 0:
        if call:
            return 1.0 if s0 > k else 0.0
        return -1.0 if s0 < k else 0.0

    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    if call:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0
