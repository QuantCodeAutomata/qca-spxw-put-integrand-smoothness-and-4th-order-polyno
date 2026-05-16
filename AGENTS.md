# Repository Knowledge: Variance Swap Replication via Gauss-Legendre Quadrature

## Project Summary
Implements 4 experiments validating Gauss-Legendre quadrature for variance swap
replication using SPX/SPXW options. Core claim: ~10 vanilla options suffice.

## Key Architecture Decisions

### Pricing Convention
- All pricing uses Black-76 under the forward measure with r=0
- Forward F estimated from put-call parity: C(K) - P(K) = F - K
- Vol units: decimal internally, vol points (×100) for reporting

### GL Node Mapping
- Put side: linear in K → K_i = 0.5[(F+K_min) + ξᵢ(F-K_min)]
- Call side: linear in x=1/K → x_j = 0.5[(1/F+1/K_max) + ξⱼ(1/F-1/K_max)]
- Use numpy.polynomial.legendre.leggauss(n) for nodes/weights

### Data Strategy
- Primary: Massive API via MASSIVE_TOKEN env var
- Fallback: Synthetic Black-Scholes chains with realistic SPX skew parameters
- All experiments work without real data (synthetic fallback)

## Module Map
- `src/bs_pricing.py`: Black-76 price, delta, gamma, vega, implied vol
- `src/forward.py`: Forward estimation from put-call parity
- `src/gauss_legendre.py`: GL nodes, strike mapping, fair strike, hedge quantities
- `src/polynomial_fit.py`: numpy.polyfit wrapper + R² computation
- `src/synthetic_surface.py`: Flat + shocked vol surface for Exp4
- `src/variance_swap.py`: Integrand, benchmark fair strike, MTM formula
- `exp/exp1_*.py`: Put integrand smoothness, 4th-order poly fit
- `exp/exp2_*.py`: Daily 2024 full-strip vs 10-pt GL comparison
- `exp/exp3_*.py`: Aug 5, 2024 shock hedge (static 10-pt portfolio)
- `exp/exp4_*.py`: Synthetic skew shock (8-pt GL, fully synthetic)
- `data/load_data.py`: Massive API data loading

## Paper Parameters
- Exp1: Trade 2024-12-06, Expiry 2025-01-17, F≈6120, T≈42/365
- Exp2: 2024-01-01 to 2024-12-06, K_min=500, K_max=8000, n=5+5
- Exp3: Trade 2024-08-02, Eval 2024-08-05, Expiry 2024-10-18, K_min=1000, K_max=10000
- Exp4: F=100, T=1yr, σ=20%, K_min=10, K_max=200, n=4+4

## Test Strategy
- Unit tests: bs_pricing, gauss_legendre, polynomial_fit, synthetic_surface
- Integration tests: all 4 experiments run end-to-end
- Key assertions: GL weights sum to 2, nodes in range, R²>0.99 for smooth integrand

## Running
```bash
python exp/run_all_experiments.py  # all experiments
pytest tests/ -v                   # all tests
```
