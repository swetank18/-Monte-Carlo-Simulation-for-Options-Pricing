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


@dataclass(frozen=True)
class HestonParams:
    s0: float
    k: float
    r: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    t: float


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


def compare_estimators_shared_paths(
    params: OptionParams,
    settings: MCSettings,
    call: bool = True,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compare naive MC, antithetic variates, and control variates using identical paths.
    Returns tuples of (price, std_error, variance_reduction) for each estimator.
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

    # Naive MC
    y = np.exp(-r * t) * payoffs
    naive_price = float(np.mean(y))
    naive_se = float(np.std(y, ddof=1) / math.sqrt(y.size))

    # Antithetic variates (pairwise average)
    pair_count = half
    pair_avg = 0.5 * (payoffs[:pair_count] + payoffs[pair_count:2 * pair_count])
    if n % 2 == 1:
        pair_avg = np.concatenate([pair_avg, payoffs[-1:]])
    y_anti = np.exp(-r * t) * pair_avg
    anti_price = float(np.mean(y_anti))
    anti_se = float(np.std(y_anti, ddof=1) / math.sqrt(y_anti.size))
    anti_vr = float(np.var(y, ddof=1) / np.var(y_anti, ddof=1)) if np.var(y_anti, ddof=1) > 0 else float("inf")

    # Control variate (Black-Scholes linear control)
    bs_price = black_scholes_price(s0, k, r, sigma, t, call=call)
    bs_delta = black_scholes_delta(s0, k, r, sigma, t, call=call)
    x = bs_price + bs_delta * (np.exp(-r * t) * s_t - s0)

    cov = np.cov(y, x, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    b_opt = cov / var_x if var_x > 0 else 0.0
    y_cv = y - b_opt * (x - bs_price)
    cv_price = float(np.mean(y_cv))
    cv_se = float(np.std(y_cv, ddof=1) / math.sqrt(y_cv.size))
    cv_vr = float(np.var(y, ddof=1) / np.var(y_cv, ddof=1)) if np.var(y_cv, ddof=1) > 0 else float("inf")

    return (naive_price, naive_se, 1.0), (anti_price, anti_se, anti_vr), (cv_price, cv_se, cv_vr)


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


def up_and_out_call_price(
    params: OptionParams,
    settings: MCSettings,
    barrier: float,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Up-and-out European call option priced via Monte Carlo with time-stepping.
    If the path ever crosses the barrier, the option is knocked out and payoff is zero.
    This is path-dependent because the barrier depends on the entire trajectory, not just S_T.
    """
    s0, k, r, sigma, t = params.s0, params.k, params.r, params.sigma, params.t
    paths = simulate_gbm_paths(s0, r, sigma, t, settings.n_paths, settings.n_steps, settings.seed)
    max_s = np.max(paths, axis=1)
    knocked_out = max_s >= barrier
    s_t = paths[:, -1]
    payoffs = np.where(knocked_out, 0.0, np.maximum(s_t - k, 0.0))
    discounted = np.exp(-r * t) * payoffs
    price, std_error, ci = mc_stats(discounted, r, t, payoffs)
    return price, std_error, ci


def asian_arithmetic_call_price(
    params: OptionParams,
    settings: MCSettings,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Arithmetic-average Asian call priced via Monte Carlo time-stepping.
    Uses full-path simulation; Black-Scholes does not apply because the payoff
    depends on the path-average rather than only S_T.
    """
    s0, k, r, sigma, t = params.s0, params.k, params.r, params.sigma, params.t
    paths = simulate_gbm_paths(s0, r, sigma, t, settings.n_paths, settings.n_steps, settings.seed)
    avg_price = np.mean(paths, axis=1)
    payoffs = np.maximum(avg_price - k, 0.0)
    discounted = np.exp(-r * t) * payoffs
    price, std_error, ci = mc_stats(discounted, r, t, payoffs)
    return price, std_error, ci


def simulate_heston_paths(
    params: HestonParams,
    settings: MCSettings,
) -> np.ndarray:
    """
    Simulate Heston stochastic volatility paths using Euler discretization.
    Volatility follows mean-reverting square-root process with correlation to stock shocks.
    """
    n_paths, n_steps = settings.n_paths, settings.n_steps
    dt = params.t / n_steps
    rng = np.random.default_rng(settings.seed) if settings.seed is not None else np.random.default_rng()

    s = np.full(n_paths, params.s0, dtype=float)
    v = np.full(n_paths, params.v0, dtype=float)

    for _ in range(n_steps):
        z1 = rng.standard_normal(size=n_paths)
        z2 = rng.standard_normal(size=n_paths)
        w2 = params.rho * z1 + math.sqrt(1.0 - params.rho ** 2) * z2

        v = np.maximum(
            v + params.kappa * (params.theta - v) * dt + params.xi * np.sqrt(np.maximum(v, 0.0)) * math.sqrt(dt) * w2,
            0.0,
        )
        s = s * np.exp((params.r - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0.0)) * math.sqrt(dt) * z1)

    return s


def heston_call_price(
    params: HestonParams,
    settings: MCSettings,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Price a European call under Heston stochastic volatility via Monte Carlo.
    """
    s_t = simulate_heston_paths(params, settings)
    payoffs = np.maximum(s_t - params.k, 0.0)
    discounted = np.exp(-params.r * params.t) * payoffs
    price, std_error, ci = mc_stats(discounted, params.r, params.t, payoffs)
    return price, std_error, ci


def finite_difference_greeks(
    params: OptionParams,
    settings: MCSettings,
    call: bool = True,
    bump: float = 0.5,
) -> Tuple[float, float, float, float]:
    """
    Finite-difference Delta/Gamma using common random numbers for variance reduction.
    Returns (delta, delta_se, gamma, gamma_se).
    """
    s0, k, r, sigma, t = params.s0, params.k, params.r, params.sigma, params.t
    rng = np.random.default_rng(settings.seed) if settings.seed is not None else np.random.default_rng()
    z = rng.standard_normal(size=settings.n_paths)

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

    # Convergence experiment for European call
    print("Convergence Experiment (Call):")
    path_counts = [1000, 5000, 10000, 20000, 50000]
    results = []
    for n in path_counts:
        exp_settings = MCSettings(
            n_paths=n,
            n_steps=settings.n_steps,
            seed=settings.seed,
            full_paths=False,
        )
        price, se, ci, _, _ = monte_carlo_price(params, exp_settings, call=True)
        results.append((n, price, se, ci))
        print(
            f"  N={n:6d} | Price={price:.4f} | SE={se:.4f} | 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]"
        )

    # Convergence summary:
    # As N increases, the standard error should fall roughly like 1/sqrt(N),
    # and the confidence interval should tighten around the true price.
    if results:
        se_first = results[0][2]
        se_last = results[-1][2]
        if se_last > 0:
            ratio = se_first / se_last
            print(
                "Convergence Summary: SE decreased by "
                f"{ratio:.2f}x from N={results[0][0]} to N={results[-1][0]}, "
                "consistent with Monte Carlo's 1/sqrt(N) rate."
            )
        else:
            print("Convergence Summary: SE did not decrease as expected (check inputs).")

    print()
    print("Estimator Comparison (Call, Shared Paths):")
    naive_stats, anti_stats, cv_stats = compare_estimators_shared_paths(params, settings, call=True)
    print("  Method       | Price   | SE      | VR")
    print(f"  Naive MC     | {naive_stats[0]:7.4f} | {naive_stats[1]:7.4f} | {naive_stats[2]:.2f}x")
    print(f"  Antithetic   | {anti_stats[0]:7.4f} | {anti_stats[1]:7.4f} | {anti_stats[2]:.2f}x")
    print(f"  Control Var  | {cv_stats[0]:7.4f} | {cv_stats[1]:7.4f} | {cv_stats[2]:.2f}x")

    print()
    print("Asian Arithmetic-Average Call (Time-Stepping):")
    asian_settings = MCSettings(
        n_paths=settings.n_paths,
        n_steps=settings.n_steps,
        seed=settings.seed,
        full_paths=True,
    )
    asian_price, asian_se, asian_ci = asian_arithmetic_call_price(params, asian_settings)
    euro_price, euro_se, euro_ci, _, _ = monte_carlo_price(params, settings, call=True)
    print(
        f"  Asian Price = {asian_price:.4f} | 95% CI [{asian_ci[0]:.4f}, {asian_ci[1]:.4f}] | SE={asian_se:.4f}"
    )
    print(
        f"  Euro Price  = {euro_price:.4f} | 95% CI [{euro_ci[0]:.4f}, {euro_ci[1]:.4f}] | SE={euro_se:.4f}"
    )
    print(
        "  Note: Black-Scholes does not apply to arithmetic-average Asian options because the payoff\n"
        "  depends on the entire path average, not just the terminal price S_T."
    )

    print()
    print("Up-and-Out Call (Barrier, Path-Dependent):")
    barrier = 130.0
    barrier_settings = MCSettings(
        n_paths=settings.n_paths,
        n_steps=settings.n_steps,
        seed=settings.seed,
        full_paths=True,
    )
    uo_price, uo_se, uo_ci = up_and_out_call_price(params, barrier_settings, barrier=barrier)
    euro_price, euro_se, euro_ci, _, _ = monte_carlo_price(params, settings, call=True)
    print(
        f"  Barrier={barrier:.2f} | Price={uo_price:.4f} | 95% CI [{uo_ci[0]:.4f}, {uo_ci[1]:.4f}] | SE={uo_se:.4f}"
    )
    print(
        f"  Euro Price={euro_price:.4f} | 95% CI [{euro_ci[0]:.4f}, {euro_ci[1]:.4f}] | SE={euro_se:.4f}"
    )
    print(
        "  Note: Barrier options are path-dependent because a single crossing knocks the option out."
    )

    print()
    print("Heston Stochastic Volatility (Euler, European Call):")
    heston_params = HestonParams(
        s0=100.0,
        k=100.0,
        r=0.05,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        t=1.0,
    )
    heston_price, heston_se, heston_ci = heston_call_price(heston_params, settings)
    bs_price = black_scholes_price(params.s0, params.k, params.r, params.sigma, params.t, call=True)
    print(
        f"  Heston Price = {heston_price:.4f} | 95% CI [{heston_ci[0]:.4f}, {heston_ci[1]:.4f}] | SE={heston_se:.4f}"
    )
    print(
        f"  BS Price     = {bs_price:.4f} (constant volatility benchmark)"
    )
    print(
        "  Note: Heston introduces stochastic volatility and skew, so prices can differ from Black-Scholes."
    )

    print()
    print("Greeks Comparison (European Call, Monte Carlo):")
    fd_delta, fd_delta_se, fd_gamma, fd_gamma_se = finite_difference_greeks(params, settings, call=True, bump=0.5)
    pw_delta, pw_delta_se, pw_delta_ci = monte_carlo_delta_pathwise(params, settings, call=True)
    print(
        f"  Pathwise Delta = {pw_delta:.4f} | SE={pw_delta_se:.4f} | 95% CI [{pw_delta_ci[0]:.4f}, {pw_delta_ci[1]:.4f}]"
    )
    print(
        f"  FD Delta      = {fd_delta:.4f} | SE={fd_delta_se:.4f} | bump=0.5"
    )
    print(
        f"  FD Gamma      = {fd_gamma:.4f} | SE={fd_gamma_se:.4f} | bump=0.5"
    )
    print(
        "  Note: Finite differences introduce bias from the bump size but are broadly applicable.\n"
        "  Pathwise estimators typically have lower variance when payoff is differentiable."
    )


if __name__ == "__main__":
    main()
