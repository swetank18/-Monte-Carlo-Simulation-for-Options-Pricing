# Monte Carlo Pricing

This project implements Monte Carlo pricing for European, Asian, and barrier options under GBM and Heston models. It includes variance reduction methods, analytical Black-Scholes benchmarks, and small experiments for convergence and estimator comparison.

## Structure
- models/
  - gbm.py: exact GBM simulation (terminal and paths)
  - heston.py: Heston stochastic volatility (Euler discretization)
- pricers/
  - european.py: European option MC pricing and Greeks
  - asian.py: arithmetic-average Asian call (path-dependent)
  - barrier.py: up-and-out call (path-dependent)
- variance_reduction/
  - antithetic.py: antithetic variates for European options
  - control_variate.py: Black-Scholes control variate
- analytics/
  - black_scholes.py: price and delta
- experiments/
  - convergence.py: price vs paths and confidence intervals
  - comparison.py: naive vs antithetic vs control variate
- utils/
  - statistics.py: MC standard error and CI

## Risk-Neutral Model
GBM under the risk-neutral measure:

S_T = S_0 * exp((r - 0.5*sigma^2) * T + sigma * sqrt(T) * Z)

Option prices are computed as:

Price = E[ exp(-rT) * payoff(S_T) ]

## Variance Reduction
- Antithetic variates: pairs each Z with -Z.
- Control variate: linear Black-Scholes control using BS price and Delta.

## Path Dependence
- Asian arithmetic-average call: depends on the average of the path.
- Up-and-out barrier call: knocked out if the path crosses a barrier.
Black-Scholes does not apply to these payoffs because they are not functions of S_T alone.

## Heston Model
The Heston model introduces stochastic variance and is simulated with Euler discretization. Prices can differ from Black-Scholes because volatility is no longer constant.

## Experiments
Run from the project root:

```powershell
python .\experiments\convergence.py
python .\experiments\comparison.py
```

## Requirements
- Python 3.10+
- NumPy
