"""European option Monte Carlo pricers and Greeks."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from analytics.black_scholes import black_scholes_delta, black_scholes_price
from models.gbm import simulate_gbm_terminal
from models.heston import simulate_heston_terminal
from utils.statistics import mc_stats_from_payoffs


def european_mc_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    seed: Optional[int],
    call: bool = True,
) -> Tuple[float, float, Tuple[float, float], np.ndarray]:
    """Naive Monte Carlo price for a European call/put."""
    s_t = simulate_gbm_terminal(s0, r, sigma, t, n_paths, seed)
    if call:
        payoffs = np.maximum(s_t - k, 0.0)
    else:
        payoffs = np.maximum(k - s_t, 0.0)
    price, std_error, ci = mc_stats_from_payoffs(payoffs, r, t)
    return price, std_error, ci, payoffs


def european_heston_mc_price(
    s0: float,
    k: float,
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
) -> Tuple[float, float, Tuple[float, float]]:
    \"\"\"European call price under Heston via Monte Carlo (Euler).\"\"\"
    s_t = simulate_heston_terminal(s0, r, v0, kappa, theta, xi, rho, t, n_paths, n_steps, seed)
    payoffs = np.maximum(s_t - k, 0.0)
    return mc_stats_from_payoffs(payoffs, r, t)


def european_mc_delta_pathwise(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    seed: Optional[int],
    call: bool = True,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Pathwise Delta estimator for European options.
    Delta = E[ exp(-rT) * 1_{S_T > K} * (S_T / S0) ] for call
    Delta = E[ -exp(-rT) * 1_{S_T < K} * (S_T / S0) ] for put
    """
    s_t = simulate_gbm_terminal(s0, r, sigma, t, n_paths, seed)

    if call:
        indicator = (s_t > k).astype(float)
        deltas = np.exp(-r * t) * indicator * (s_t / s0)
    else:
        indicator = (s_t < k).astype(float)
        deltas = -np.exp(-r * t) * indicator * (s_t / s0)

    delta = float(np.mean(deltas))
    std_error = float(np.std(deltas, ddof=1) / math.sqrt(deltas.size))
    ci = (delta - 1.96 * std_error, delta + 1.96 * std_error)
    return delta, std_error, ci


def european_mc_greeks_fd(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    seed: Optional[int],
    call: bool = True,
    bump: float = 0.5,
) -> Tuple[float, float, float, float]:
    """
    Finite-difference Delta/Gamma using common random numbers.
    Returns (delta, delta_se, gamma, gamma_se).
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    z = rng.standard_normal(size=n_paths)

    def payoff_with_s0(s0_val: float) -> np.ndarray:
        drift = (r - 0.5 * sigma ** 2) * t
        diffusion = sigma * math.sqrt(t) * z
        s_t = s0_val * np.exp(drift + diffusion)
        if call:
            return np.maximum(s_t - k, 0.0)
        return np.maximum(k - s_t, 0.0)

    pay_up = payoff_with_s0(s0 + bump)
    pay_dn = payoff_with_s0(s0 - bump)
    pay_0 = payoff_with_s0(s0)

    disc = math.exp(-r * t)
    d_i = disc * (pay_up - pay_dn) / (2.0 * bump)
    g_i = disc * (pay_up - 2.0 * pay_0 + pay_dn) / (bump ** 2)

    delta = float(np.mean(d_i))
    gamma = float(np.mean(g_i))
    delta_se = float(np.std(d_i, ddof=1) / math.sqrt(d_i.size))
    gamma_se = float(np.std(g_i, ddof=1) / math.sqrt(g_i.size))
    return delta, delta_se, gamma, gamma_se


def european_bs_price_and_delta(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    call: bool = True,
) -> Tuple[float, float]:
    """Convenience wrapper for BS price and delta."""
    return (
        black_scholes_price(s0, k, r, sigma, t, call=call),
        black_scholes_delta(s0, k, r, sigma, t, call=call),
    )
