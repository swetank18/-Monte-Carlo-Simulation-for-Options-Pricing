"""
Monte Carlo simulation for European option pricing using geometric Brownian motion.

Risk-neutral pricing assumption: under the risk-neutral measure, the discounted
asset price is a martingale and expected discounted payoffs equal the option price.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# =========================
# Data Structures
# =========================


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
    full_paths: bool = False  # set True for path-dependent options


# =========================
# Black-Scholes Analytics
# =========================


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(s0: float, k: float, r: float, sigma: float, t: float, call: bool = True) -> float:
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


def black_scholes_delta(s0: float, k: float, r: float, sigma: float, t: float, call: bool = True) -> float:
    """Black-Scholes Delta for a European call/put."""
    if t <= 0 or sigma <= 0:
        if call:
            return 1.0 if s0 > k else 0.0
        return -1.0 if s0 < k else 0.0

    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    if call:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


# =========================
# GBM Simulation (Exact)
# =========================


def simulate_gbm_terminal(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    seed: int | None,
) -> np.ndarray:
    """
    Simulate terminal prices using the exact GBM solution:
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
    seed: int | None,
) -> np.ndarray:
    """
    Simulate full GBM paths using the exact per-step update.
    This is only needed for path-dependent options.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    dt = t / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)

    z = rng.standard_normal(size=(n_paths, n_steps))
    increments = drift + vol * z
    log_s = math.log(s0) + np.cumsum(increments, axis=1)
    return np.exp(log_s)


# =========================
# Statistics Utilities
# =========================


def mc_stats(discounted: np.ndarray, r: float, t: float, raw_payoffs: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute Monte Carlo price, standard error, and 95% confidence interval.
    Std error = exp(-rT) * std(payoffs) / sqrt(N)
    CI = price +/- 1.96 * std_error
    """
    n = discounted.size
    price = float(np.mean(discounted))
    std_error = float(math.exp(-r * t) * np.std(raw_payoffs, ddof=1) / math.sqrt(n))
    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error
    return price, std_error, (ci_low, ci_high)


# =========================
# Monte Carlo Pricers
# =========================


def monte_carlo_price(
    params: OptionParams,
    settings: MCSettings,
    call: bool = True,
) -> Tuple[float, float, Tuple[float, float], np.ndarray, np.ndarray]:
    """
    Naive Monte Carlo pricing for a European call/put.
    Returns (price, std_error, (ci_low, ci_high), s_t, raw_payoffs).
    """
    s0, k, r, sigma, t = params.s0, params.k, params.r, params.sigma, params.t

    if settings.full_paths:
        paths = simulate_gbm_paths(s0, r, sigma, t, settings.n_paths, settings.n_steps, settings.seed)
        s_t = paths[:, -1]
    else:
        s_t = simulate_gbm_terminal(s0, r, sigma, t, settings.n_paths, settings.seed)

    if call:
        payoffs = np.maximum(s_t - k, 0.0)
    else:
        payoffs = np.maximum(k - s_t, 0.0)

    discounted = np.exp(-r * t) * payoffs
    price, std_error, ci = mc_stats(discounted, r, t, payoffs)
    return price, std_error, ci, s_t, payoffs


def antithetic_monte_carlo_price(
    params: OptionParams,
    settings: MCSettings,
    call: bool = True,
) -> Tuple[float, float, Tuple[float, float], float]:
    """
    Antithetic variates Monte Carlo pricing for a European call/put.
    Returns (price, std_error, (ci_low, ci_high), variance_reduction_ratio).
    """
    s0, k, r, sigma, t = params.s0, params.k, params.r, params.sigma, params.t
    rng = np.random.default_rng(settings.seed) if settings.seed is not None else np.random.default_rng()

    n = settings.n_paths
    half = n // 2
    z = rng.standard_normal(size=half)
    z_full = np.concatenate([z, -z])
    if n % 2 == 1:
        z_full = np.concatenate([z_full, rng.standard_normal(size=1)])

    drift = (r - 0.5 * sigma ** 2) * t
    diffusion = sigma * math.sqrt(t) * z_full
    s_t = s0 * np.exp(drift + diffusion)

    if call:
        payoffs = np.maximum(s_t - k, 0.0)
    else:
        payoffs = np.maximum(k - s_t, 0.0)

    # Pairwise averaging for antithetic estimator
    pair_count = half
    pair_avg = 0.5 * (payoffs[:pair_count] + payoffs[pair_count:2 * pair_count])
    if n % 2 == 1:
        pair_avg = np.concatenate([pair_avg, payoffs[-1:]])

    discounted = np.exp(-r * t) * pair_avg
    price, std_error, ci = mc_stats(discounted, r, t, pair_avg)

    # Variance reduction vs naive MC on the same N
    naive_s_t = simulate_gbm_terminal(s0, r, sigma, t, n, settings.seed)
    if call:
        naive_payoffs = np.maximum(naive_s_t - k, 0.0)
    else:
        naive_payoffs = np.maximum(k - naive_s_t, 0.0)
    naive_discounted = np.exp(-r * t) * naive_payoffs

    var_reduction = float(np.var(naive_discounted, ddof=1) / np.var(discounted, ddof=1)) if np.var(discounted, ddof=1) > 0 else float("inf")
    return price, std_error, ci, var_reduction


def control_variate_price(
    params: OptionParams,
    settings: MCSettings,
    call: bool = True,
) -> Tuple[float, float, Tuple[float, float], float]:
    """
    Control variate estimator using a Black-Scholes linear control:

    X = BS_price + BS_delta * (exp(-rT) * S_T - S_0)
    E[X] = BS_price

    The control variate estimator is Y_cv = Y - b*(X - BS_price).
    Returns (cv_price, cv_std_error, cv_ci, variance_reduction_ratio).
    """
    s0, k, r, sigma, t = params.s0, params.k, params.r, params.sigma, params.t

    s_t = simulate_gbm_terminal(s0, r, sigma, t, settings.n_paths, settings.seed)
    if call:
        payoffs = np.maximum(s_t - k, 0.0)
    else:
        payoffs = np.maximum(k - s_t, 0.0)

    y = np.exp(-r * t) * payoffs  # target estimator
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


def monte_carlo_delta_pathwise(
    params: OptionParams,
    settings: MCSettings,
    call: bool = True,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Pathwise Delta estimator for a European call/put.
    Delta = E[ exp(-rT) * 1_{S_T > K} * (S_T / S0) ] for call
    Delta = E[ -exp(-rT) * 1_{S_T < K} * (S_T / S0) ] for put
    """
    s0, k, r, sigma, t = params.s0, params.k, params.r, params.sigma, params.t
    s_t = simulate_gbm_terminal(s0, r, sigma, t, settings.n_paths, settings.seed)

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


# =========================
# Main
# =========================


def main() -> None:
    params = OptionParams(s0=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0)
    settings = MCSettings(n_paths=20000, n_steps=252, seed=42, full_paths=False)

    for call in (True, False):
        label = "Call" if call else "Put"

        mc_price, mc_se, mc_ci, _, _ = monte_carlo_price(params, settings, call=call)
        anti_price, anti_se, anti_ci, anti_vr = antithetic_monte_carlo_price(params, settings, call=call)
        bs_price = black_scholes_price(params.s0, params.k, params.r, params.sigma, params.t, call=call)
        bs_delta = black_scholes_delta(params.s0, params.k, params.r, params.sigma, params.t, call=call)

        cv_price, cv_se, cv_ci, vr = control_variate_price(params, settings, call=call)
        mc_delta, mc_delta_se, mc_delta_ci = monte_carlo_delta_pathwise(params, settings, call=call)

        print(f"{label}:")
        print(
            f"  MC Price = {mc_price:.4f} | 95% CI [{mc_ci[0]:.4f}, {mc_ci[1]:.4f}] | SE={mc_se:.4f}"
        )
        print(
            f"  Anti Price = {anti_price:.4f} | 95% CI [{anti_ci[0]:.4f}, {anti_ci[1]:.4f}] | SE={anti_se:.4f} | VR={anti_vr:.2f}x"
        )
        print(
            f"  CV Price = {cv_price:.4f} | 95% CI [{cv_ci[0]:.4f}, {cv_ci[1]:.4f}] | SE={cv_se:.4f} | VR={vr:.2f}x"
        )
        print(
            f"  BS Price = {bs_price:.4f} | Diff (MC - BS) = {mc_price - bs_price:+.4f}"
        )
        print(
            f"  MC Delta = {mc_delta:.4f} | 95% CI [{mc_delta_ci[0]:.4f}, {mc_delta_ci[1]:.4f}] | SE={mc_delta_se:.4f}"
        )
        print(f"  BS Delta = {bs_delta:.4f}")
        print()


if __name__ == "__main__":
    main()
