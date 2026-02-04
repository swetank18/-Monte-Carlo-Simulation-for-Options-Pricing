"""Streamlit app for visualizing precomputed Monte Carlo results."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st


ROOT = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, "results")


def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_image(name: str) -> str | None:
    path = os.path.join(RESULTS_DIR, name)
    return path if os.path.exists(path) else None


def main() -> None:
    st.set_page_config(page_title="Monte Carlo Pricing Results", layout="wide")

    st.title("Monte Carlo Pricing Results")
    st.header("Overview")
    st.write(
        "This dashboard visualizes precomputed Monte Carlo outputs only. "
        "No simulations run here; all data is loaded from CSVs and plots in results/."
    )

    st.header("Pricing Summary")
    st.write(
        "Summary table for European, Asian, Barrier, and Heston models with standard errors "
        "and 95% confidence intervals."
    )
    summary_df = load_csv("results_summary.csv")
    if summary_df.empty:
        st.warning("results_summary.csv not found in results/")
    else:
        st.dataframe(summary_df, use_container_width=True)

    st.header("Greeks Summary")
    st.write(
        "Comparison of Delta (and Gamma where applicable) across Monte Carlo estimators and "
        "the Black-Scholes analytical benchmark."
    )
    greeks_df = load_csv("greeks_summary.csv")
    if greeks_df.empty:
        st.warning("greeks_summary.csv not found in results/")
    else:
        st.dataframe(greeks_df, use_container_width=True)

    st.header("Variance Reduction")
    st.write("Bar chart comparing naive, antithetic, and control variate estimators.")
    vr_img = load_image("variance_reduction.png")
    if vr_img:
        st.image(vr_img, use_container_width=True)
    else:
        st.warning("variance_reduction.png not found in results/")

    st.header("Convergence")
    st.write("Monte Carlo price convergence with 95% confidence intervals as paths increase.")
    conv_img = load_image("convergence.png")
    if conv_img:
        st.image(conv_img, use_container_width=True)
    else:
        st.warning("convergence.png not found in results/")

    st.header("Heston Volatility Smile")
    st.write("Implied volatility smile under Heston with flat Black-Scholes overlay.")
    smile_img = load_image("heston_smile.png")
    if smile_img:
        st.image(smile_img, use_container_width=True)
    else:
        st.warning("heston_smile.png not found in results/")


if __name__ == "__main__":
    main()
