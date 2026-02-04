"""Run all pricing experiments and write results to CSV (summary only)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from analytics.black_scholes import black_scholes_price
from pricers.asian import asian_arithmetic_call_mc_price
from pricers.barrier import up_and_out_call_mc_price
from pricers.european import (
    european_bs_price_and_delta,
    european_heston_mc_price,
    european_mc_delta_pathwise,
    european_mc_greeks_fd,
    european_mc_price,
)
from utils.reporting import write_csv, write_greeks_csv


def run() -> None:
    # Fixed parameters and seeds
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
    n_paths, n_steps, seed = 20000, 252, 42

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    rows: list[dict[str, object]] = []

    # European (naive)
    price, se, ci, _ = european_mc_price(s0, k, r, sigma, t, n_paths, seed, call=True)
    rows.append(
        {
            "Model": "European Call (Naive MC)",
            "Price": price,
            "StdError": se,
            "CI_Low": ci[0],
            "CI_High": ci[1],
            "Notes": "GBM exact terminal",
        }
    )

    # Asian arithmetic-average call
    asian_price, asian_se, asian_ci = asian_arithmetic_call_mc_price(s0, k, r, sigma, t, n_paths, n_steps, seed)
    rows.append(
        {
            "Model": "Asian Arithmetic Call",
            "Price": asian_price,
            "StdError": asian_se,
            "CI_Low": asian_ci[0],
            "CI_High": asian_ci[1],
            "Notes": "Path-dependent average",
        }
    )

    # Up-and-out barrier call
    barrier = 130.0
    uo_price, uo_se, uo_ci = up_and_out_call_mc_price(s0, k, r, sigma, t, n_paths, n_steps, seed, barrier)
    rows.append(
        {
            "Model": "Up-and-Out Call",
            "Price": uo_price,
            "StdError": uo_se,
            "CI_Low": uo_ci[0],
            "CI_High": uo_ci[1],
            "Notes": f"Barrier={barrier:.2f}",
        }
    )

    # Heston European call
    h_price, h_se, h_ci = european_heston_mc_price(
        s0=s0,
        k=k,
        r=r,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        t=t,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    rows.append(
        {
            "Model": "Heston Call (Euler)",
            "Price": h_price,
            "StdError": h_se,
            "CI_Low": h_ci[0],
            "CI_High": h_ci[1],
            "Notes": "Stochastic vol",
        }
    )

    # Black-Scholes benchmark
    bs_price = black_scholes_price(s0, k, r, sigma, t, call=True)
    rows.append(
        {
            "Model": "Black-Scholes (Call)",
            "Price": bs_price,
            "StdError": "",
            "CI_Low": "",
            "CI_High": "",
            "Notes": "Analytical benchmark",
        }
    )

    # Greeks summary
    pw_delta, pw_se, pw_ci = european_mc_delta_pathwise(s0, k, r, sigma, t, n_paths, seed, call=True)
    fd_delta, fd_delta_se, fd_gamma, fd_gamma_se = european_mc_greeks_fd(
        s0, k, r, sigma, t, n_paths, seed, call=True, bump=0.5
    )
    _, bs_delta = european_bs_price_and_delta(s0, k, r, sigma, t, call=True)

    greeks_rows = [
        {
            "Method": "Pathwise MC",
            "Delta": pw_delta,
            "Gamma": "",
            "StdError": pw_se,
            "Notes": "Pathwise estimator (Delta only)",
        },
        {
            "Method": "Finite Diff MC",
            "Delta": fd_delta,
            "Gamma": fd_gamma,
            "StdError": fd_delta_se,
            "Notes": "Common random numbers, bump=0.5",
        },
        {
            "Method": "Black-Scholes",
            "Delta": bs_delta,
            "Gamma": "",
            "StdError": "",
            "Notes": "Analytical benchmark",
        },
    ]

    results_path = os.path.join(results_dir, "results_summary.csv")
    greeks_path = os.path.join(results_dir, "greeks_summary.csv")

    write_csv(results_path, rows)
    write_greeks_csv(greeks_path, greeks_rows)

    print("Run complete. Wrote:")
    print(f"  {results_path}")
    print(f"  {greeks_path}")


if __name__ == "__main__":
    run()
