"""Estimator comparison using identical paths."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from analytics.black_scholes import black_scholes_delta, black_scholes_price


def run() -> None:
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
    n_paths = 20000
    seed = 42

    rng = np.random.default_rng(seed)
    half = n_paths // 2
    z = rng.standard_normal(size=half)
    z_full = np.concatenate([z, -z])
    if n_paths % 2 == 1:
        z_full = np.concatenate([z_full, rng.standard_normal(size=1)])

    drift = (r - 0.5 * sigma ** 2) * t
    diffusion = sigma * math.sqrt(t) * z_full
    s_t = s0 * np.exp(drift + diffusion)

    payoffs = np.maximum(s_t - k, 0.0)
    y = np.exp(-r * t) * payoffs

    # Naive MC
    naive_price = float(np.mean(y))
    naive_se = float(np.std(y, ddof=1) / math.sqrt(y.size))

    # Antithetic MC (pairwise averaging)
    pair_count = half
    pair_avg = 0.5 * (payoffs[:pair_count] + payoffs[pair_count:2 * pair_count])
    if n_paths % 2 == 1:
        pair_avg = np.concatenate([pair_avg, payoffs[-1:]])
    y_anti = np.exp(-r * t) * pair_avg
    anti_price = float(np.mean(y_anti))
    anti_se = float(np.std(y_anti, ddof=1) / math.sqrt(y_anti.size))
    anti_vr = float(np.var(y, ddof=1) / np.var(y_anti, ddof=1)) if np.var(y_anti, ddof=1) > 0 else float("inf")

    # Control variate (Black-Scholes linear control)
    bs_price = black_scholes_price(s0, k, r, sigma, t, call=True)
    bs_delta = black_scholes_delta(s0, k, r, sigma, t, call=True)
    x = bs_price + bs_delta * (np.exp(-r * t) * s_t - s0)

    cov = np.cov(y, x, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    b_opt = cov / var_x if var_x > 0 else 0.0
    y_cv = y - b_opt * (x - bs_price)
    cv_price = float(np.mean(y_cv))
    cv_se = float(np.std(y_cv, ddof=1) / math.sqrt(y_cv.size))
    cv_vr = float(np.var(y, ddof=1) / np.var(y_cv, ddof=1)) if np.var(y_cv, ddof=1) > 0 else float("inf")

    print("Estimator Comparison (European Call, Shared Paths):")
    print("  Method       | Price   | SE      | VR")
    print(f"  Naive MC     | {naive_price:7.4f} | {naive_se:7.4f} | 1.00x")
    print(f"  Antithetic   | {anti_price:7.4f} | {anti_se:7.4f} | {anti_vr:.2f}x")
    print(f"  Control Var  | {cv_price:7.4f} | {cv_se:7.4f} | {cv_vr:.2f}x")


if __name__ == "__main__":
    run()
