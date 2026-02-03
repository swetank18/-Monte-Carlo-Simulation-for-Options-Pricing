"""
Monte Carlo simulation for European option pricing using geometric Brownian motion.
Compares Monte Carlo price to Black-Scholes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class OptionParams:
    s0: float  # spot price
    k: float   # strike
    r: float   # risk-free rate (annualized, continuous compounding)
    sigma: float  # volatility (annualized)
    t: float   # time to maturity in years


@dataclass(frozen=True)
class MCSettings:
    n_paths: int = 20000
    n_steps: int = 252
    seed: int | None = 42


def black_scholes_price(params: OptionParams, call: bool = True) -> float:
    """Black-Scholes price for European call/put."""
    s0, k, r, sigma, t = params.s0, params.k, params.r, params.sigma, params.t
    if t <= 0:
        intrinsic = max(0.0, s0 - k) if call else max(0.0, k - s0)
        return intrinsic
    if sigma <= 0:
        forward = s0 * math.exp(r * t)
        intrinsic = max(0.0, forward - k) if call else max(0.0, k - forward)
        return intrinsic * math.exp(-r * t)

    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    # CDF of standard normal
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    if call:
        return s0 * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)
    return k * math.exp(-r * t) * norm_cdf(-d2) - s0 * norm_cdf(-d1)


def monte_carlo_price(
    params: OptionParams,
    settings: MCSettings,
    call: bool = True,
) -> Tuple[float, float]:
    """
    Monte Carlo price and standard error for a European call/put.
    Returns (price, standard_error).
    """
    s0, k, r, sigma, t = params.s0, params.k, params.r, params.sigma, params.t
    n_paths, n_steps = settings.n_paths, settings.n_steps

    if settings.seed is not None:
        rng = np.random.default_rng(settings.seed)
    else:
        rng = np.random.default_rng()

    dt = t / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)

    # Simulate log returns
    z = rng.standard_normal(size=(n_paths, n_steps))
    log_returns = drift + vol * z
    log_price = math.log(s0) + np.cumsum(log_returns, axis=1)
    s_t = np.exp(log_price[:, -1])

    if call:
        payoffs = np.maximum(s_t - k, 0.0)
    else:
        payoffs = np.maximum(k - s_t, 0.0)

    discounted = np.exp(-r * t) * payoffs
    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / math.sqrt(n_paths))
    return price, std_error


def main() -> None:
    params = OptionParams(s0=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0)
    settings = MCSettings(n_paths=20000, n_steps=252, seed=42)

    for call in (True, False):
        mc_price, mc_se = monte_carlo_price(params, settings, call=call)
        bs_price = black_scholes_price(params, call=call)
        label = "Call" if call else "Put"
        print(
            f"{label}: MC={mc_price:.4f} (SE={mc_se:.4f}), "
            f"BS={bs_price:.4f}, Diff={mc_price - bs_price:+.4f}"
        )


if __name__ == "__main__":
    main()
