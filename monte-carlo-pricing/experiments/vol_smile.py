"""Heston volatility smile experiment."""
from __future__ import annotations

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from analytics.black_scholes import black_scholes_price
from analytics.implied_vol import implied_volatility_bisect
from pricers.european import european_heston_mc_price


def run(save_path: str = "vol_smile.png") -> None:
    s0, r, t = 100.0, 0.05, 1.0
    strikes = np.arange(70, 131, 5)
    n_paths, n_steps, seed = 20000, 252, 42

    # Heston parameters
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.6
    rho = -0.7

    heston_prices = []
    heston_ivs = []

    for k in strikes:
        price, _, _ = european_heston_mc_price(
            s0=s0,
            k=float(k),
            r=r,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            t=t,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        )
        heston_prices.append(price)
        iv = implied_volatility_bisect(price, s0, float(k), r, t, call=True)
        heston_ivs.append(iv if iv is not None else float("nan"))

    flat_sigma = math.sqrt(v0)
    bs_line = [flat_sigma for _ in strikes]

    plt.figure(figsize=(8, 5))
    plt.plot(strikes, heston_ivs, label="Heston IV", marker="o")
    plt.plot(strikes, bs_line, label="Black-Scholes (flat)", linestyle="--")
    plt.title("Volatility Smile under Heston")
    plt.xlabel("Strike K")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    run()
