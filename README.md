# Variance Swap Replication via Gauss-Legendre Quadrature

## Overview

This repository implements and validates four experiments from a paper on
efficient variance swap replication using Gauss-Legendre (GL) quadrature.
The central claim is that a small number of vanilla options (≈10 total:
5 puts + 5 calls) can accurately replicate the fair variance swap strike
when the variance-swap integrand is smooth.

## Repository Structure

```
repo/
├── data/                   # Data loading (Massive API)
│   └── load_data.py
├── src/                    # Core implementation
│   ├── bs_pricing.py       # Black-76 pricing and Greeks
│   ├── forward.py          # Forward estimation from put-call parity
│   ├── gauss_legendre.py   # GL quadrature nodes, weights, and replication
│   ├── polynomial_fit.py   # Polynomial fitting and R² computation
│   ├── synthetic_surface.py # Synthetic vol surface for Exp4
│   └── variance_swap.py    # Variance swap fair strike and MTM
├── exp/                    # Experiment scripts
│   ├── exp1_put_integrand_smoothness.py
│   ├── exp2_daily_variance_strike_comparison.py
│   ├── exp3_aug5_hedge.py
│   ├── exp4_synthetic_skew_shock.py
│   └── run_all_experiments.py
├── tests/                  # pytest tests
│   ├── test_bs_pricing.py
│   ├── test_gauss_legendre.py
│   ├── test_polynomial_fit.py
│   ├── test_synthetic_surface.py
│   └── test_experiments.py
├── results/                # Outputs
│   └── RESULTS.md
├── README.md
└── .gitignore
```

## Experiments

### Exp1: SPXW Put-Integrand Smoothness (2024-12-06)
Reproduces the paper's Dec 6, 2024 example. Fits a 4th-order polynomial to
the put-side variance-swap integrand g(K) = put(F,K)/K² and verifies R² ≈ 0.9992.

### Exp2: Daily 2024 Full-Strip vs 10-Point GL Comparison
Compares a full-strip trapezoidal benchmark against a 10-point GL approximation
(5 puts + 5 calls) for each trading day in 2024.

### Exp3: Static 10-Point Hedge Under Aug 5, 2024 Shock
Constructs the 10-point GL portfolio on 2024-08-02 and revalues it on 2024-08-05
after the SPX volatility shock. Compares replication PnL (~11.3 vol pts) against
the benchmark variance swap MTM (~11.7 vol pts).

### Exp4: Synthetic 1-Year Skew-Shock Hedging Test
Fully synthetic test with F=100, T=1yr, σ=20%. Applies a stylized skew shock
and verifies that an 8-point GL portfolio (4 puts + 4 calls) closely replicates
the benchmark variance swap MTM.

## Setup

### Requirements

```bash
pip install numpy pandas scipy matplotlib scikit-learn statsmodels
```

### Environment Variables

For real market data (optional; experiments fall back to synthetic data):
```bash
export MASSIVE_TOKEN=<your_massive_api_key>
```

## Usage

### Run all experiments
```bash
python exp/run_all_experiments.py
```

### Run individual experiments
```bash
python exp/exp1_put_integrand_smoothness.py
python exp/exp2_daily_variance_strike_comparison.py
python exp/exp3_aug5_hedge.py
python exp/exp4_synthetic_skew_shock.py
```

### Run tests
```bash
pytest tests/ -v
```

## Key Methodology

### Variance Swap Fair Strike
```
K_v² = (2/T) [ ∫₀ᶠ put(F,K)/K² dK + ∫ᶠ∞ call(F,K)/K² dK ]
```

### Gauss-Legendre Node Mapping
- **Put side** (linear in K): `K_i = 0.5[(F + K_min) + ξᵢ(F - K_min)]`
- **Call side** (linear in x=1/K): `x_j = 0.5[(1/F + 1/K_max) + ξⱼ(1/F - 1/K_max)]`

### Hedge Quantities
```
q_i^put = (v/(2K_v)) × ((F-K_min)/(T×(K_i^put)²)) × wᵢ
q_j^call = (v/(2K_v)) × ((1/F - 1/K_max)/T) × wⱼ
```

## Results

See `results/RESULTS.md` for a summary of all experiment outcomes.

## Data Sources

- **Primary**: Massive API (requires `MASSIVE_TOKEN` environment variable)
- **Fallback**: Synthetic Black-Scholes option chains with realistic SPX parameters

## License

MIT
