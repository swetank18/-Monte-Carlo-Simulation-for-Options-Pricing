# Monte Carlo Pricing

## Motivation
Black?Scholes assumes constant volatility and lognormal returns. Empirically, implied volatility varies with strike and maturity, producing a volatility smile/skew that violates those assumptions. This project uses Monte Carlo simulation to (i) price options that are path-dependent or outside the Black?Scholes analytic scope, and (ii) compare constant-volatility pricing against stochastic-volatility dynamics (Heston) to expose model risk.

## Monte Carlo Methodology
Under the risk-neutral measure, option prices are discounted expectations of payoffs. For European options under GBM, terminal prices are simulated using the exact solution:

S_T = S_0 * exp((r - 0.5*sigma^2) * T + sigma * sqrt(T) * Z),  Z ~ N(0, 1)

For path-dependent options (Asian arithmetic-average and up-and-out barrier), full paths are simulated with time-stepping and payoffs computed from the path functional. Standard errors and 95% confidence intervals are reported for all Monte Carlo estimators.

## Variance Reduction
Two variance-reduction techniques are implemented:
- Antithetic variates: pairs Z with -Z to reduce variance via negative correlation.
- Control variates: a Black?Scholes linear control using analytic price and Delta, with an optimal control coefficient estimated from the sample.

## Stochastic Volatility and Model Risk
Heston stochastic volatility is simulated via Euler discretization. Because volatility is random and correlated with returns, option prices and implied volatilities deviate from the constant-volatility Black?Scholes benchmark. The resulting implied-volatility smile demonstrates model risk: a single sigma cannot explain prices across strikes.

## Key Numerical Findings
Quantitative outputs are written to `results/`:
- `results/results_summary.csv`: pricing summaries with standard errors and confidence intervals.
- `results/greeks_summary.csv`: Delta/Gamma comparisons.
- `results/convergence.csv`: convergence vs path count.
- `results/heston_smile.csv`: implied volatility smile data.

The accompanying plots in `results/` visualize variance reduction effectiveness, Monte Carlo convergence, and the Heston smile. See `results/README.md` for the exact column definitions.

## Reproducibility
From the project root:

```powershell
pip install -r .\requirements.txt
python .\experiments\run_all.py
python .\experiments\plot_results.py
```

These commands generate the CSV outputs and plots under `results/`.

## Streamlit Dashboard
Launch the precomputed-results dashboard:

```powershell
streamlit run .\app.py
```

The app reads data from `results/` only and does not run simulations.

## Repository Structure
- models/: GBM and Heston simulators
- pricers/: European, Asian, and barrier pricers + Greeks
- variance_reduction/: antithetic and control variate
- analytics/: Black?Scholes and implied volatility utilities
- experiments/: batch runners and plot generation
- utils/: statistical utilities and CSV reporting
- results/: generated CSVs and plots
