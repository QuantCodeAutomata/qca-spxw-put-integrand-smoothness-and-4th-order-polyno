"""
Experiment 1: SPXW Put-Integrand Smoothness and 4th-Order Polynomial Fit
on 2024-12-06.

Reproduces the paper's Dec 6, 2024 example:
- Loads SPXW option chain for expiry 2025-01-17
- Estimates forward F ≈ 6120 from put-call parity
- Computes put-side variance-swap integrand g(K) = put(F,K)/K^2
- Fits 4th-order polynomial and reports R² ≈ 0.9992
- Produces diagnostic plots

If Massive API data is unavailable, falls back to a synthetic reconstruction
using Black-Scholes with a realistic SPX implied vol surface.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bs_pricing import black76_price, implied_vol_bisection
from src.forward import estimate_forward_parity
from src.variance_swap import compute_put_integrand
from src.polynomial_fit import fit_polynomial, compute_r_squared, compute_fit_statistics, fit_multiple_degrees

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Paper parameters
TRADE_DATE = "2024-12-06"
EXPIRY_DATE = "2025-01-17"
FORWARD_APPROX = 6120.0
SPOT_APPROX = 6090.0
T_YEARS = 42.0 / 365.0  # ~42 calendar days to Jan 17 from Dec 6


def build_synthetic_spxw_chain(
    F: float = FORWARD_APPROX,
    T: float = T_YEARS,
    spot: float = SPOT_APPROX,
    n_strikes: int = 80,
) -> pd.DataFrame:
    """
    Build a synthetic SPXW option chain using a realistic SPX skew surface.

    Uses a simple parametric skew: sigma(K) = atm_vol + skew * (K/F - 1)
    with atm_vol ≈ 14% and skew ≈ -0.15 (typical SPX Dec 2024 conditions).

    Parameters
    ----------
    F : float
        Forward price.
    T : float
        Time to expiry in years.
    spot : float
        SPX spot price.
    n_strikes : int
        Number of strikes to generate.

    Returns
    -------
    pd.DataFrame
        Synthetic option chain.
    """
    # Realistic SPX skew parameters for Dec 2024
    atm_vol = 0.135
    skew = -0.12
    convexity = 0.08

    # Strike range: 80% to 110% of forward for puts, 100% to 115% for calls
    put_strikes = np.linspace(0.75 * F, 0.999 * F, n_strikes // 2)
    call_strikes = np.linspace(1.001 * F, 1.15 * F, n_strikes // 2)
    all_strikes = np.concatenate([put_strikes, call_strikes])

    records = []
    for K in all_strikes:
        moneyness = K / F - 1.0
        # Parametric skew: quadratic in moneyness
        iv = atm_vol + skew * moneyness + convexity * moneyness**2
        iv = max(iv, 0.01)

        # Price put
        put_price = black76_price(F, K, T, iv, "put")
        # Price call
        call_price = black76_price(F, K, T, iv, "call")

        if K < F:
            records.append({
                "date": TRADE_DATE,
                "expiry": EXPIRY_DATE,
                "strike": K,
                "type": "put",
                "price": put_price,
                "iv": iv,
                "spot": spot,
            })
        else:
            records.append({
                "date": TRADE_DATE,
                "expiry": EXPIRY_DATE,
                "strike": K,
                "type": "call",
                "price": call_price,
                "iv": iv,
                "spot": spot,
            })

    df = pd.DataFrame(records)
    return df


def load_or_build_chain() -> pd.DataFrame:
    """
    Attempt to load real SPXW data from Massive API; fall back to synthetic.

    Returns
    -------
    pd.DataFrame
        Option chain with columns [date, expiry, strike, type, price, iv, spot].
    """
    try:
        from data.load_data import fetch_option_chain
        df = fetch_option_chain("SPXW", TRADE_DATE, EXPIRY_DATE)
        if len(df) > 10:
            logger.info(f"Loaded {len(df)} real SPXW options from Massive API")
            return df
        else:
            logger.warning("Insufficient real data; using synthetic chain")
    except Exception as e:
        logger.warning(f"Massive API unavailable ({e}); using synthetic chain")

    df = build_synthetic_spxw_chain()
    logger.info(f"Built synthetic chain with {len(df)} options")
    return df


def run_exp1() -> dict:
    """
    Run Experiment 1: Put-integrand smoothness and 4th-order polynomial fit.

    Returns
    -------
    dict
        Results dictionary with R², coefficients, and statistics.
    """
    logger.info("=" * 60)
    logger.info("EXP1: SPXW Put-Integrand Smoothness (2024-12-06)")
    logger.info("=" * 60)

    # Step 1: Load option chain
    df = load_or_build_chain()
    data_source = "real" if "real" in str(df.get("source", "synthetic")) else "synthetic"

    # Step 2: Determine forward F
    spot = df["spot"].dropna().median()
    if np.isnan(spot) or spot <= 0:
        spot = SPOT_APPROX

    try:
        F = estimate_forward_parity(df, spot=spot, atm_band_pct=0.05)
        logger.info(f"Forward estimated from put-call parity: F = {F:.2f}")
    except Exception as e:
        logger.warning(f"Parity estimation failed ({e}); using F = {FORWARD_APPROX}")
        F = FORWARD_APPROX

    logger.info(f"Forward F = {F:.2f}, Spot = {spot:.2f}, T = {T_YEARS:.4f} years")

    # Step 3: Filter to put options with K < F
    puts = df[
        (df["type"].str.lower() == "put")
        & (df["strike"] < F)
        & (df["price"] > 0)
    ].copy().sort_values("strike")

    logger.info(f"Put options with K < F: {len(puts)} strikes")
    logger.info(f"Strike range: [{puts['strike'].min():.0f}, {puts['strike'].max():.0f}]")

    # Step 4: Build put-side variance-swap integrand g(K) = put(F,K)/K^2
    K_arr = puts["strike"].values.astype(float)
    P_arr = puts["price"].values.astype(float)
    integrand = compute_put_integrand(K_arr, P_arr)

    # Step 5: Select strike region for fit
    # Use the full available put region below F
    K_fit = K_arr.copy()
    g_fit = integrand.copy()

    logger.info(f"Fitting region: K in [{K_fit.min():.0f}, {K_fit.max():.0f}], n={len(K_fit)}")

    # Step 6 & 7: Fit 4th-order polynomial and compute R²
    multi_fits = fit_multiple_degrees(K_fit, g_fit, degrees=[2, 3, 4, 5])

    main_degree = 4
    coeffs_4, fitted_4 = fit_polynomial(K_fit, g_fit, main_degree)
    stats_4 = compute_fit_statistics(K_fit, g_fit, fitted_4)

    logger.info(f"\n4th-order polynomial fit:")
    logger.info(f"  R² = {stats_4['r_squared']:.6f}")
    logger.info(f"  RMSE = {stats_4['rmse']:.2e}")
    logger.info(f"  MAE  = {stats_4['mae']:.2e}")
    logger.info(f"  Max |residual| = {stats_4['max_abs_residual']:.2e}")
    logger.info(f"  Coefficients (deg 4 to 0): {coeffs_4}")

    # Step 9: Compare multiple degrees
    logger.info("\nComparison across polynomial degrees:")
    for deg, res in multi_fits.items():
        logger.info(f"  Degree {deg}: R² = {res['stats']['r_squared']:.6f}")

    # Step 8: Plot empirical integrand and polynomial fit overlay
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"SPXW Put-Side Variance-Swap Integrand\n"
        f"Trade Date: {TRADE_DATE}, Expiry: {EXPIRY_DATE}, F ≈ {F:.0f}",
        fontsize=13,
    )

    # Main plot: empirical vs 4th-order fit
    ax = axes[0, 0]
    ax.plot(K_fit, g_fit * 1e6, "b.", markersize=4, label="Empirical g(K)", alpha=0.7)
    ax.plot(K_fit, fitted_4 * 1e6, "r-", linewidth=2,
            label=f"4th-order poly (R²={stats_4['r_squared']:.4f})")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("g(K) = put(K)/K² × 10⁶")
    ax.set_title("Put Integrand vs 4th-Order Polynomial Fit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residuals plot
    ax = axes[0, 1]
    residuals_4 = g_fit - fitted_4
    ax.plot(K_fit, residuals_4 * 1e6, "g.", markersize=4)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Residual × 10⁶")
    ax.set_title("Residuals of 4th-Order Fit")
    ax.grid(True, alpha=0.3)

    # Multi-degree comparison
    ax = axes[1, 0]
    ax.plot(K_fit, g_fit * 1e6, "b.", markersize=3, label="Empirical", alpha=0.6)
    colors = {2: "orange", 3: "green", 4: "red", 5: "purple"}
    for deg, res in multi_fits.items():
        r2 = res["stats"]["r_squared"]
        ax.plot(K_fit, res["fitted"] * 1e6, "-",
                color=colors.get(deg, "gray"), linewidth=1.5,
                label=f"Deg {deg} (R²={r2:.4f})")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("g(K) × 10⁶")
    ax.set_title("Polynomial Fits: Degrees 2–5")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # R² bar chart
    ax = axes[1, 1]
    degs = sorted(multi_fits.keys())
    r2_vals = [multi_fits[d]["stats"]["r_squared"] for d in degs]
    bars = ax.bar([str(d) for d in degs], r2_vals, color=["orange", "green", "red", "purple"])
    ax.axhline(0.9992, color="k", linestyle="--", linewidth=1.5, label="Paper target R²=0.9992")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("R²")
    ax.set_title("R² by Polynomial Degree")
    ax.set_ylim([max(0, min(r2_vals) - 0.01), 1.001])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, r2 in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{r2:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp1_put_integrand_smoothness.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to {plot_path}")

    # Step 10: Report
    r2_4 = stats_4["r_squared"]
    paper_target = 0.9992
    logger.info(f"\n{'='*50}")
    logger.info(f"RESULT: 4th-order R² = {r2_4:.6f}")
    logger.info(f"Paper target R²      = {paper_target:.4f}")
    logger.info(f"Difference           = {abs(r2_4 - paper_target):.6f}")
    logger.info(f"Close to paper?      = {abs(r2_4 - paper_target) < 0.01}")

    # Step 11: Gauss-Legendre heuristic
    logger.info("\nGauss-Legendre heuristic:")
    logger.info("  If integrand ≈ degree-4 polynomial, then n=3 GL points integrate")
    logger.info("  exactly up to degree 2*3-1=5, covering the degree-4 polynomial.")
    logger.info("  For a narrow subrange, n=2 to 3 quadrature points may suffice.")

    results = {
        "trade_date": TRADE_DATE,
        "expiry_date": EXPIRY_DATE,
        "forward_F": F,
        "spot": spot,
        "T_years": T_YEARS,
        "n_put_strikes": len(K_fit),
        "strike_range": (float(K_fit.min()), float(K_fit.max())),
        "r_squared_4th_order": r2_4,
        "paper_target_r2": paper_target,
        "coefficients_4th_order": coeffs_4.tolist(),
        "fit_stats_4th_order": stats_4,
        "multi_degree_r2": {d: multi_fits[d]["stats"]["r_squared"] for d in multi_fits},
        "data_source": "synthetic (Massive API fallback)",
    }
    return results


if __name__ == "__main__":
    results = run_exp1()
    print(f"\nExp1 complete. R²(4th order) = {results['r_squared_4th_order']:.6f}")
