"""CLI runner for Monte Carlo pricing demos."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(__file__)
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
from variance_reduction.antithetic import antithetic_european_mc_price
from variance_reduction.control_variate import control_variate_european_mc_price
from utils.reporting import write_csv, write_greeks_csv


def run_all(out_csv: str | None) -> None:
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
    n_paths, n_steps, seed = 20000, 252, 42
    rows: list[dict[str, object]] = []

    print("European Call (Naive MC):")
    price, se, ci, _ = european_mc_price(s0, k, r, sigma, t, n_paths, seed, call=True)
    print(f"  Price={price:.4f} | SE={se:.4f} | 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    rows.append(
        {
            "Model": "European Call (Naive MC)",
            "Price": price,
            "StdError": se,
            "CI_Low": ci[0],
            "CI_High": ci[1],
            "Notes": "GBM exact terminal simulation",
        }
    )

    print("European Call (Antithetic):")
    a_price, a_se, a_ci, a_vr = antithetic_european_mc_price(s0, k, r, sigma, t, n_paths, seed, call=True)
    print(f"  Price={a_price:.4f} | SE={a_se:.4f} | 95% CI [{a_ci[0]:.4f}, {a_ci[1]:.4f}] | VR={a_vr:.2f}x")
    rows.append(
        {
            "Model": "European Call (Antithetic)",
            "Price": a_price,
            "StdError": a_se,
            "CI_Low": a_ci[0],
            "CI_High": a_ci[1],
            "Notes": f"VR={a_vr:.2f}x",
        }
    )

    print("European Call (Control Variate):")
    c_price, c_se, c_ci, c_vr = control_variate_european_mc_price(s0, k, r, sigma, t, n_paths, seed, call=True)
    print(f"  Price={c_price:.4f} | SE={c_se:.4f} | 95% CI [{c_ci[0]:.4f}, {c_ci[1]:.4f}] | VR={c_vr:.2f}x")
    rows.append(
        {
            "Model": "European Call (Control Variate)",
            "Price": c_price,
            "StdError": c_se,
            "CI_Low": c_ci[0],
            "CI_High": c_ci[1],
            "Notes": f"VR={c_vr:.2f}x (BS control)",
        }
    )

    bs_price, bs_delta = european_bs_price_and_delta(s0, k, r, sigma, t, call=True)
    print("Black-Scholes (European Call):")
    print(f"  Price={bs_price:.4f} | Delta={bs_delta:.4f}")

    print("Asian Arithmetic-Average Call:")
    asian_price, asian_se, asian_ci = asian_arithmetic_call_mc_price(s0, k, r, sigma, t, n_paths, n_steps, seed)
    print(f"  Price={asian_price:.4f} | SE={asian_se:.4f} | 95% CI [{asian_ci[0]:.4f}, {asian_ci[1]:.4f}]")
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

    print("Up-and-Out Call:")
    barrier = 130.0
    uo_price, uo_se, uo_ci = up_and_out_call_mc_price(s0, k, r, sigma, t, n_paths, n_steps, seed, barrier)
    print(f"  Barrier={barrier:.2f} | Price={uo_price:.4f} | SE={uo_se:.4f} | 95% CI [{uo_ci[0]:.4f}, {uo_ci[1]:.4f}]")
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

    print("Heston European Call (Euler):")
    h_price, h_se, h_ci = european_heston_mc_price(
        s0=100.0,
        k=100.0,
        r=0.05,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        t=1.0,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    print(f"  Price={h_price:.4f} | SE={h_se:.4f} | 95% CI [{h_ci[0]:.4f}, {h_ci[1]:.4f}]")
    print(f"  BS Price (const vol)={black_scholes_price(s0, k, r, sigma, t, call=True):.4f}")
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

    print("Greeks (European Call):")
    pw_delta, pw_se, pw_ci = european_mc_delta_pathwise(s0, k, r, sigma, t, n_paths, seed, call=True)
    fd_delta, fd_delta_se, fd_gamma, fd_gamma_se = european_mc_greeks_fd(s0, k, r, sigma, t, n_paths, seed, call=True, bump=0.5)
    bs_price, bs_delta = european_bs_price_and_delta(s0, k, r, sigma, t, call=True)
    print(f"  Pathwise Delta={pw_delta:.4f} | SE={pw_se:.4f} | 95% CI [{pw_ci[0]:.4f}, {pw_ci[1]:.4f}]")
    print(f"  FD Delta={fd_delta:.4f} | SE={fd_delta_se:.4f} | bump=0.5")
    print(f"  FD Gamma={fd_gamma:.4f} | SE={fd_gamma_se:.4f} | bump=0.5")

    rows.append(
        {
            "Model": "Greeks (European Call)",
            "Price": "",
            "StdError": "",
            "CI_Low": "",
            "CI_High": "",
            "Notes": "Pathwise vs finite difference",
            "Delta_Pathwise": pw_delta,
            "Delta_Pathwise_SE": pw_se,
            "Delta_FD": fd_delta,
            "Delta_FD_SE": fd_delta_se,
            "Gamma_FD": fd_gamma,
            "Gamma_FD_SE": fd_gamma_se,
        }
    )

    if out_csv:
        write_csv(out_csv, rows)
        print(f"Results saved to {out_csv}")

    # Greeks summary table
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
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    greeks_path = os.path.join(results_dir, "greeks_summary.csv")
    write_greeks_csv(greeks_path, greeks_rows)
    print(f"Greeks summary saved to {greeks_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo pricing demo runner")
    parser.add_argument("--all", action="store_true", help="Run all demos (default)")
    parser.add_argument("--out", default="results.csv", help="CSV output path for results table")
    args = parser.parse_args()

    if args.all or True:
        run_all(args.out)


if __name__ == "__main__":
    main()
