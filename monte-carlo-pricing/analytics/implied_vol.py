"""Implied volatility solver via bisection."""
from __future__ import annotations

import math
from typing import Optional

from analytics.black_scholes import black_scholes_price


def implied_volatility_bisect(
    price: float,
    s0: float,
    k: float,
    r: float,
    t: float,
    call: bool = True,
    vol_low: float = 1e-6,
    vol_high: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Optional[float]:
    """Solve for implied vol using bisection. Returns None if not bracketed."""
    if t <= 0:
        return None

    price_low = black_scholes_price(s0, k, r, vol_low, t, call=call)
    price_high = black_scholes_price(s0, k, r, vol_high, t, call=call)

    if not (price_low <= price <= price_high):
        return None

    lo, hi = vol_low, vol_high
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pmid = black_scholes_price(s0, k, r, mid, t, call=call)
        if math.isclose(pmid, price, rel_tol=0.0, abs_tol=tol):
            return mid
        if pmid < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
