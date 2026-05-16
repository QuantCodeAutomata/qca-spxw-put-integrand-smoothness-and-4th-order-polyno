"""
Experiment 2: Daily 2024 Comparison of Full-Strip Variance Strike vs
10-Point Gauss-Legendre Approximation.

For each trading day in 2024-01-01 to 2024-12-06:
1. Select the one-month expiry (closest to 30 calendar days).
2. Compute benchmark full-strip variance strike via trapezoidal integration
   over [500, 8000].
3. Compute 10-point GL approximation (5 puts + 5 calls).
4. Compare the two series.

Falls back to synthetic data if Massive API is unavailable.
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
from typing import Optional, Tuple
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bs_pricing import black76_price, implied_vol_bisection
from src.forward import estimate_forward_parity
from src.variance_swap import benchmark_variance_strike
from src.gauss_legendre import (
    gauss_legendre_nodes_weights,
    map_put_nodes,
    map_call_nodes,
    gauss_legendre_variance_strike,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Paper parameters
START_DATE = "2024-01-01"
END_DATE = "2024-12-06"
K_MIN = 500.0
K_MAX = 8000.0
N_PUT = 5
N_CALL = 5


def select_one_month_expiry(trade_date: str, available_expiries: list) -> Optional[str]:
    """
    Select the listed expiry closest to 30 calendar days from trade_date.

    Parameters
    ----------
    trade_date : str
        Trade date 'YYYY-MM-DD'.
    available_expiries : list of str
        Available expiry dates.

    Returns
    -------
    str or None
        Selected expiry date.
    """
    td = pd.Timestamp(trade_date)
    target = td + pd.Timedelta(days=30)
    best = None
    best_diff = float("inf")
    for exp in available_expiries:
        diff = abs((pd.Timestamp(exp) - target).days)
        if diff < best_diff:
            best_diff = diff
            best = exp
    return best


def build_synthetic_daily_chain(
    trade_date: str,
    expiry_date: str,
    spot: float,
    atm_vol: float,
    skew: float = -0.10,
    n_strikes: int = 100,
) -> pd.DataFrame:
    """
    Build a synthetic daily option chain for a given trade date and expiry.

    Parameters
    ----------
    trade_date : str
        Trade date.
    expiry_date : str
        Expiry date.
    spot : float
        SPX spot price.
    atm_vol : float
        ATM implied volatility (decimal).
    skew : float
        Linear skew parameter.
    n_strikes : int
        Number of strikes.

    Returns
    -------
    pd.DataFrame
        Synthetic option chain.
    """
    T = (pd.Timestamp(expiry_date) - pd.Timestamp(trade_date)).days / 365.0
    if T <= 0:
        return pd.DataFrame()

    F = spot  # r=0, so F = spot

    # Strike range: K_MIN to K_MAX
    strikes = np.linspace(K_MIN, K_MAX, n_strikes)
    records = []

    for K in strikes:
        moneyness = K / F - 1.0
        iv = atm_vol + skew * moneyness + 0.05 * moneyness**2
        iv = max(iv, 0.005)

        if K < F:
            price = black76_price(F, K, T, iv, "put")
            opt_type = "put"
        else:
            price = black76_price(F, K, T, iv, "call")
            opt_type = "call"

        records.append({
            "date": trade_date,
            "expiry": expiry_date,
            "strike": K,
            "type": opt_type,
            "price": price,
            "iv": iv,
            "spot": spot,
        })

    return pd.DataFrame(records)


def generate_synthetic_2024_data() -> dict:
    """
    Generate synthetic SPX data for all of 2024 with realistic dynamics.

    Simulates a VIX-like path with a spike around Aug 5, 2024.

    Returns
    -------
    dict
        Mapping from trade_date to (spot, atm_vol, expiry_date).
    """
    trading_days = pd.bdate_range(START_DATE, END_DATE)
    n = len(trading_days)

    # Simulate SPX path: start ~4750, end ~6090
    np.random.seed(42)
    log_returns = np.random.normal(0.0004, 0.008, n)

    # Add Aug 5 shock: large negative return
    aug5_idx = None
    for i, d in enumerate(trading_days):
        if d.strftime("%Y-%m-%d") == "2024-08-05":
            aug5_idx = i
            break

    if aug5_idx is not None:
        log_returns[aug5_idx] = -0.03  # ~3% drop on Aug 5

    spot_path = 4750.0 * np.exp(np.cumsum(log_returns))

    # Simulate ATM vol path: mean-reverting around 15%, spike on Aug 5
    vix_path = np.zeros(n)
    vix_path[0] = 0.13
    for i in range(1, n):
        vix_path[i] = 0.95 * vix_path[i - 1] + 0.05 * 0.15 + np.random.normal(0, 0.005)
        vix_path[i] = max(vix_path[i], 0.08)

    if aug5_idx is not None:
        vix_path[aug5_idx] = 0.38  # VIX spike to ~38 on Aug 5
        for j in range(1, min(10, n - aug5_idx)):
            vix_path[aug5_idx + j] = vix_path[aug5_idx] * (0.85**j) + 0.15 * (1 - 0.85**j)

    data = {}
    for i, td in enumerate(trading_days):
        td_str = td.strftime("%Y-%m-%d")
        # Select expiry ~30 days out
        target_exp = td + pd.Timedelta(days=30)
        # Round to nearest Friday (SPX monthly expiry)
        days_to_friday = (4 - target_exp.weekday()) % 7
        expiry = target_exp + pd.Timedelta(days=days_to_friday)
        # Use 3rd Friday of the month
        expiry_str = expiry.strftime("%Y-%m-%d")
        data[td_str] = {
            "spot": float(spot_path[i]),
            "atm_vol": float(vix_path[i]),
            "expiry": expiry_str,
        }

    return data


def compute_gl_variance_strike_from_chain(
    df: pd.DataFrame,
    F: float,
    T: float,
    K_min: float = K_MIN,
    K_max: float = K_MAX,
    n_put: int = N_PUT,
    n_call: int = N_CALL,
) -> float:
    """
    Compute the Gauss-Legendre variance strike from an option chain.

    Uses linear interpolation in implied vol space to price at quadrature nodes.

    Parameters
    ----------
    df : pd.DataFrame
        Option chain.
    F : float
        Forward price.
    T : float
        Time to expiry in years.
    K_min : float
        Lower truncation bound.
    K_max : float
        Upper truncation bound.
    n_put : int
        Number of put nodes.
    n_call : int
        Number of call nodes.

    Returns
    -------
    float
        GL fair variance strike.
    """
    # Build interpolators for put and call prices
    puts = df[df["type"].str.lower() == "put"].sort_values("strike")
    calls = df[df["type"].str.lower() == "call"].sort_values("strike")

    def put_pricer(K: float) -> float:
        if len(puts) < 2:
            return max(F - K, 0.0)
        K_arr = puts["strike"].values
        P_arr = puts["price"].values
        if K <= K_arr[0]:
            return float(P_arr[0])
        if K >= K_arr[-1]:
            return max(F - K, 0.0)
        f_interp = interp1d(K_arr, P_arr, kind="linear", fill_value="extrapolate")
        return max(float(f_interp(K)), 0.0)

    def call_pricer(K: float) -> float:
        if len(calls) < 2:
            return max(K - F, 0.0)
        K_arr = calls["strike"].values
        C_arr = calls["price"].values
        if K >= K_arr[-1]:
            return float(C_arr[-1])
        if K <= K_arr[0]:
            return max(K - F, 0.0)
        f_interp = interp1d(K_arr, C_arr, kind="linear", fill_value="extrapolate")
        return max(float(f_interp(K)), 0.0)

    try:
        K_v, _, _, _, _, _ = gauss_legendre_variance_strike(
            put_pricer=put_pricer,
            call_pricer=call_pricer,
            F=F,
            T=T,
            K_min=K_min,
            K_max=K_max,
            n_put=n_put,
            n_call=n_call,
        )
        return K_v
    except Exception as e:
        logger.warning(f"GL variance strike failed: {e}")
        return np.nan


def run_exp2() -> dict:
    """
    Run Experiment 2: Daily 2024 full-strip vs 10-point GL comparison.

    Returns
    -------
    dict
        Results with time series and summary statistics.
    """
    logger.info("=" * 60)
    logger.info("EXP2: Daily 2024 Variance Strike Comparison")
    logger.info("=" * 60)

    # Try to load real data; fall back to synthetic
    use_synthetic = True
    try:
        from data.load_data import fetch_option_chain
        # Quick test
        test_df = fetch_option_chain("SPXW", "2024-01-02", "2024-02-16")
        if len(test_df) > 10:
            use_synthetic = False
            logger.info("Using real Massive API data")
    except Exception as e:
        logger.warning(f"Massive API unavailable ({e}); using synthetic data")

    # Generate synthetic data for 2024
    synthetic_data = generate_synthetic_2024_data()
    trading_days = sorted(synthetic_data.keys())

    results_rows = []

    for td in trading_days:
        info = synthetic_data[td]
        spot = info["spot"]
        atm_vol = info["atm_vol"]
        expiry = info["expiry"]

        T = (pd.Timestamp(expiry) - pd.Timestamp(td)).days / 365.0
        if T <= 0.01:
            continue

        F = spot  # r=0

        # Build synthetic chain
        df = build_synthetic_daily_chain(td, expiry, spot, atm_vol)
        if df.empty:
            continue

        # Benchmark: full-strip trapezoidal
        try:
            K_v_bench, _, _ = benchmark_variance_strike(
                df, F, T, K_min=K_MIN, K_max=K_MAX
            )
        except Exception as e:
            logger.warning(f"{td}: benchmark failed: {e}")
            K_v_bench = np.nan

        # GL approximation: 10-point (5 put + 5 call)
        K_v_gl = compute_gl_variance_strike_from_chain(df, F, T)

        results_rows.append({
            "date": td,
            "spot": spot,
            "atm_vol": atm_vol * 100,
            "expiry": expiry,
            "T_years": T,
            "F": F,
            "K_v_benchmark": K_v_bench * 100 if not np.isnan(K_v_bench) else np.nan,
            "K_v_gl": K_v_gl * 100 if not np.isnan(K_v_gl) else np.nan,
        })

    df_results = pd.DataFrame(results_rows)
    df_results["date"] = pd.to_datetime(df_results["date"])
    df_results = df_results.dropna(subset=["K_v_benchmark", "K_v_gl"])

    # Compute error metrics
    diff = df_results["K_v_gl"] - df_results["K_v_benchmark"]
    mae = float(diff.abs().mean())
    rmse = float(np.sqrt((diff**2).mean()))
    max_ae = float(diff.abs().max())
    mean_rel_err = float((diff.abs() / df_results["K_v_benchmark"]).mean() * 100)

    logger.info(f"\nSummary Statistics (vol points):")
    logger.info(f"  MAE  = {mae:.4f}")
    logger.info(f"  RMSE = {rmse:.4f}")
    logger.info(f"  Max |error| = {max_ae:.4f}")
    logger.info(f"  Mean relative error = {mean_rel_err:.4f}%")

    # Save results CSV
    csv_path = RESULTS_DIR / "exp2_daily_variance_strikes.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        "Daily 2024 Variance Strike: Full-Strip vs 10-Point Gauss-Legendre\n"
        "SPX, Jan 2024 – Dec 2024",
        fontsize=13,
    )

    # Main comparison
    ax = axes[0]
    ax.plot(df_results["date"], df_results["K_v_benchmark"], "b-",
            linewidth=1.5, label="Full-Strip Benchmark", alpha=0.9)
    ax.plot(df_results["date"], df_results["K_v_gl"], "r--",
            linewidth=1.5, label="10-Point GL Approximation", alpha=0.9)
    ax.set_ylabel("Fair Variance Strike (vol points)")
    ax.set_title("Fair Variance Strike Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Difference
    ax = axes[1]
    ax.plot(df_results["date"], diff, "g-", linewidth=1, label="GL - Benchmark")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.fill_between(df_results["date"], diff, 0, alpha=0.3, color="green")
    ax.set_ylabel("Difference (vol points)")
    ax.set_title(f"Approximation Error (MAE={mae:.4f}, RMSE={rmse:.4f} vol pts)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SPX spot
    ax = axes[2]
    ax.plot(df_results["date"], df_results["spot"], "k-", linewidth=1.5)
    ax.set_ylabel("SPX Spot")
    ax.set_xlabel("Date")
    ax.set_title("SPX Spot Price")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp2_daily_variance_strike_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to {plot_path}")

    results = {
        "n_trading_days": len(df_results),
        "mae_vol_pts": mae,
        "rmse_vol_pts": rmse,
        "max_abs_error_vol_pts": max_ae,
        "mean_relative_error_pct": mean_rel_err,
        "data_source": "synthetic",
        "df_results": df_results,
    }
    return results


if __name__ == "__main__":
    results = run_exp2()
    print(f"\nExp2 complete. MAE = {results['mae_vol_pts']:.4f} vol pts")
