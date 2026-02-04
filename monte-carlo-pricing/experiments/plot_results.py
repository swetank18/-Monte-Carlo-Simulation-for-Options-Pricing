"""Plot results from CSV files in results/ without running simulations."""
from __future__ import annotations

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _save_placeholder(path: str, title: str, message: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.title(title)
    plt.axis("off")
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_variance_reduction() -> None:
    out_path = os.path.join(RESULTS_DIR, "variance_reduction.png")
    rows = _read_csv(os.path.join(RESULTS_DIR, "results_summary.csv"))

    models = []
    prices = []
    errors = []

    for row in rows:
        model = row.get("Model", "")
        if any(k in model for k in ["Naive", "Antithetic", "Control Variate"]):
            try:
                price = float(row.get("Price", ""))
                se = float(row.get("StdError", ""))
            except ValueError:
                continue
            models.append(model)
            prices.append(price)
            errors.append(se)

    if not models:
        _save_placeholder(out_path, "Variance Reduction", "No variance reduction data found")
        return

    plt.figure(figsize=(8, 4.5))
    plt.bar(models, prices, yerr=errors, capsize=4)
    plt.title("Variance Reduction Comparison (Price with SE)")
    plt.ylabel("Price")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_convergence() -> None:
    out_path = os.path.join(RESULTS_DIR, "convergence.png")
    rows = _read_csv(os.path.join(RESULTS_DIR, "convergence.csv"))

    if not rows:
        _save_placeholder(out_path, "Convergence", "No convergence.csv found in results/")
        return

    ns = []
    prices = []
    ci_low = []
    ci_high = []

    for row in rows:
        try:
            n = int(row.get("N", ""))
            price = float(row.get("Price", ""))
            low = float(row.get("CI_Low", ""))
            high = float(row.get("CI_High", ""))
        except ValueError:
            continue
        ns.append(n)
        prices.append(price)
        ci_low.append(low)
        ci_high.append(high)

    if not ns:
        _save_placeholder(out_path, "Convergence", "convergence.csv missing required columns")
        return

    plt.figure(figsize=(8, 4.5))
    plt.plot(ns, prices, marker="o", label="Price")
    plt.fill_between(ns, ci_low, ci_high, alpha=0.2, label="95% CI")
    plt.title("Convergence of MC Price")
    plt.xlabel("Number of Paths (N)")
    plt.ylabel("Price")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_heston_smile() -> None:
    out_path = os.path.join(RESULTS_DIR, "heston_smile.png")
    rows = _read_csv(os.path.join(RESULTS_DIR, "heston_smile.csv"))

    if not rows:
        _save_placeholder(out_path, "Heston Smile", "No heston_smile.csv found in results/")
        return

    strikes = []
    iv = []
    bs = []

    for row in rows:
        try:
            k = float(row.get("Strike", ""))
            h = float(row.get("ImpliedVol", ""))
            b = float(row.get("BS_Flat", ""))
        except ValueError:
            continue
        strikes.append(k)
        iv.append(h)
        bs.append(b)

    if not strikes:
        _save_placeholder(out_path, "Heston Smile", "heston_smile.csv missing required columns")
        return

    plt.figure(figsize=(8, 4.5))
    plt.plot(strikes, iv, marker="o", label="Heston IV")
    plt.plot(strikes, bs, linestyle="--", label="Black-Scholes (flat)")
    plt.title("Volatility Smile under Heston")
    plt.xlabel("Strike K")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_variance_reduction()
    plot_convergence()
    plot_heston_smile()


if __name__ == "__main__":
    main()
