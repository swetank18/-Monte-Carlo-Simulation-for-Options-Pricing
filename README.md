# Monte Carlo Simulation for Options Pricing

This project prices European options by Monte Carlo simulation under the risk-neutral measure, and cross-checks the result against the Black-Scholes analytical formula.

## Mathematical Model
Under geometric Brownian motion (GBM) with risk-neutral drift:

- **SDE**: dS = rS dt + sigma S dW
- **Exact solution**:
  - S_T = S_0 * exp((r - 0.5 * sigma^2) T + sigma * sqrt(T) * Z), Z ~ N(0, 1)

Risk-neutral pricing assumes the discounted asset price is a martingale, so the option price is:

- Price = E[ exp(-rT) * payoff(S_T) ]

## Monte Carlo Estimator
For N simulated paths:

- Payoff_i = max(S_T^i - K, 0) (call) or max(K - S_T^i, 0) (put)
- Price = exp(-rT) * (1/N) * sum(Payoff_i)

### Statistical Diagnostics
Monte Carlo is a statistical estimator, so we report standard error and confidence intervals:

- Std error = exp(-rT) * std(Payoff) / sqrt(N)
- 95% CI = Price +/- 1.96 * Std error

## Black-Scholes Benchmark
The Black-Scholes formula provides a closed-form European option price and Delta, used as a benchmark and as a control variate reference.

## Variance Reduction Techniques
We implement two variance reduction methods:

- **Antithetic variates**: pair each normal draw Z with -Z to reduce variance.
- **Control variate (Black-Scholes linear control)**:
  - X = BS_price + BS_delta * (exp(-rT) * S_T - S_0)
  - E[X] = BS_price
  - Y_cv = Y - b*(X - BS_price), with b chosen to minimize variance

We report the **variance reduction factor** to quantify improvement over naive Monte Carlo.

## Convergence Discussion
Monte Carlo error decays at rate O(1/sqrt(N)). Increasing the number of paths reduces variance but with diminishing returns. Variance-reduction techniques (control variates, antithetic variates) are often more effective than brute-force sampling.

## Limitations
- This implementation assumes constant volatility and rates.
- Only European options are priced directly.
- Path-dependent options require full-path simulation (supported via a flag).

## Future Extensions
- Stratified sampling
- Time-varying volatility or stochastic volatility (Heston)
- Path-dependent payoffs (Asian, barrier)
- Greeks via likelihood ratio or adjoint methods

## Run
```powershell
python .\monte_carlo_option_pricing.py
```

## Parameters
Edit the defaults in `monte_carlo_option_pricing.py` to adjust S0, K, r, sigma, T, and simulation settings.
