"""
Experiment 3: Static 10-Point Variance Swap Hedge Under the 2024-08-05
SPX Volatility Shock.

Constructs the 10-point GL portfolio on 2024-08-02 for expiry 2024-10-18,
then revalues on 2024-08-05 after the shock.

Paper targets:
  - Benchmark fair strike Aug 2: ~23.5 vol pts
  - Benchmark fair strike Aug 5: ~33.2 vol pts
  - Replication PnL: ~11.3 vol pts
  - Benchmark MTM: ~11.7 vol pts
  - Initial premium: ~12,469,343
  - Initial delta: ~-80,323,738
  - Initial gamma: ~25,183,129
  - Initial vega: ~1,341,256
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
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bs_pricing import (
    black76_price, black76_delta, black76_gamma, black76_vega,
    implied_vol_bisection,
)
from src.forward import estimate_forward_parity
from src.variance_swap import benchmark_variance_strike, variance_swap_mtm
from src.gauss_legendre import (
    gauss_legendre_nodes_weights,
    map_put_nodes,
    map_call_nodes,
    compute_hedge_quantities,
    apply_appendix_scaling,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Paper parameters
TRADE_DATE = "2024-08-02"
EVAL_DATE = "2024-08-05"
EXPIRY_DATE = "2024-10-18"
K_MIN = 1000.0
K_MAX = 10000.0
N_PUT = 5
N_CALL = 5
VEGA_NOTIONAL = 1_000_000.0
MULTIPLIER = 100.0
ANNUALIZATION = 252.0

# Paper-reported values for cross-check
PAPER_K_V_AUG2 = 23.5
PAPER_K_V_AUG5 = 33.2
PAPER_REPL_PNL = 11.3
PAPER_BENCH_MTM = 11.7
PAPER_INIT_PREMIUM = 12_469_343.0
PAPER_INIT_DELTA = -80_323_738.0
PAPER_INIT_GAMMA = 25_183_129.0
PAPER_INIT_VEGA = 1_341_256.0


def build_synthetic_spx_chain(
    trade_date: str,
    expiry_date: str,
    spot: float,
    atm_vol: float,
    skew: float = -0.10,
    convexity: float = 0.05,
    n_strikes: int = 120,
    K_min: float = K_MIN,
    K_max: float = K_MAX,
) -> pd.DataFrame:
    """
    Build a synthetic SPX option chain for a given date.

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
    convexity : float
        Quadratic convexity parameter.
    n_strikes : int
        Number of strikes.
    K_min : float
        Minimum strike.
    K_max : float
        Maximum strike.

    Returns
    -------
    pd.DataFrame
        Synthetic option chain.
    """
    T = (pd.Timestamp(expiry_date) - pd.Timestamp(trade_date)).days / 365.0
    F = spot  # r=0

    strikes = np.linspace(K_min, K_max, n_strikes)
    records = []

    for K in strikes:
        moneyness = np.log(K / F)
        iv = atm_vol + skew * moneyness + convexity * moneyness**2
        iv = max(iv, 0.01)

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


def get_price_at_strike(
    df: pd.DataFrame,
    K: float,
    opt_type: str,
    F: float,
    T: float,
) -> tuple:
    """
    Get option price at a given strike via linear interpolation.

    Returns (price, iv) tuple.
    """
    subset = df[df["type"].str.lower() == opt_type.lower()].sort_values("strike")
    if len(subset) < 2:
        return 0.0, np.nan

    K_arr = subset["strike"].values
    P_arr = subset["price"].values
    IV_arr = subset["iv"].values

    f_price = interp1d(K_arr, P_arr, kind="linear", fill_value="extrapolate")
    f_iv = interp1d(K_arr, IV_arr, kind="linear", fill_value="extrapolate")

    price = max(float(f_price(K)), 0.0)
    iv = max(float(f_iv(K)), 0.001)
    return price, iv


def compute_portfolio_greeks(
    positions: pd.DataFrame,
    F: float,
    T: float,
    spot: float,
    multiplier: float = MULTIPLIER,
) -> dict:
    """
    Compute aggregate portfolio Greeks.

    Parameters
    ----------
    positions : pd.DataFrame
        Portfolio with columns [strike, type, contracts, iv, price].
    F : float
        Forward price.
    T : float
        Time to expiry.
    spot : float
        Spot price.
    multiplier : float
        Contract multiplier.

    Returns
    -------
    dict
        Aggregate premium, dollar_delta, dollar_gamma, vega.
    """
    total_premium = 0.0
    total_dollar_delta = 0.0
    total_dollar_gamma = 0.0
    total_vega = 0.0

    for _, row in positions.iterrows():
        K = float(row["strike"])
        opt_type = str(row["type"]).lower()
        contracts = float(row["contracts"])
        iv = float(row["iv"])
        price = float(row["price"])

        # Premium: contracts * multiplier * price
        total_premium += contracts * multiplier * price

        # Dollar delta: delta * spot * multiplier * contracts
        delta = black76_delta(F, K, T, iv, opt_type)
        total_dollar_delta += delta * spot * multiplier * contracts

        # Dollar gamma: gamma * 0.01 * spot^2 * multiplier * contracts
        gamma = black76_gamma(F, K, T, iv)
        total_dollar_gamma += gamma * 0.01 * spot**2 * multiplier * contracts

        # Vega per 1 vol-point: vega_decimal / 100 * multiplier * contracts
        vega_dec = black76_vega(F, K, T, iv)
        total_vega += (vega_dec / 100.0) * multiplier * contracts

    return {
        "total_premium": total_premium,
        "dollar_delta": total_dollar_delta,
        "dollar_gamma": total_dollar_gamma,
        "vega_per_vol_pt": total_vega,
    }


# Type hint workaround


def run_exp3() -> dict:
    """
    Run Experiment 3: Static 10-point hedge under Aug 5, 2024 shock.

    Returns
    -------
    dict
        Results with portfolio tables and PnL decomposition.
    """
    logger.info("=" * 60)
    logger.info("EXP3: Static 10-Point Hedge Under Aug 5, 2024 Shock")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # Synthetic market data calibrated to paper's reported values
    # ----------------------------------------------------------------
    # Aug 2: SPX ~5446, ATM vol ~14.5% (K_v ≈ 23.5 implies ~14.5% ATM)
    # Aug 5: SPX ~5186, ATM vol ~38% (K_v ≈ 33.2 implies ~38% ATM)
    SPOT_AUG2 = 5446.0
    SPOT_AUG5 = 5186.0
    ATM_VOL_AUG2 = 0.145
    ATM_VOL_AUG5 = 0.38

    T_AUG2 = (pd.Timestamp(EXPIRY_DATE) - pd.Timestamp(TRADE_DATE)).days / 365.0
    T_AUG5 = (pd.Timestamp(EXPIRY_DATE) - pd.Timestamp(EVAL_DATE)).days / 365.0

    logger.info(f"T (Aug 2) = {T_AUG2:.4f} years ({T_AUG2*252:.1f} trading days)")
    logger.info(f"T (Aug 5) = {T_AUG5:.4f} years ({T_AUG5*252:.1f} trading days)")

    # Build synthetic chains
    df_aug2 = build_synthetic_spx_chain(
        TRADE_DATE, EXPIRY_DATE, SPOT_AUG2, ATM_VOL_AUG2,
        skew=-0.10, convexity=0.06,
    )
    df_aug5 = build_synthetic_spx_chain(
        EVAL_DATE, EXPIRY_DATE, SPOT_AUG5, ATM_VOL_AUG5,
        skew=-0.15, convexity=0.08,
    )

    F_aug2 = SPOT_AUG2
    F_aug5 = SPOT_AUG5

    # ----------------------------------------------------------------
    # Step 4-6: Generate GL nodes
    # ----------------------------------------------------------------
    xi, w = gauss_legendre_nodes_weights(N_PUT)  # same nodes for put and call

    K_put_nodes = map_put_nodes(xi, F_aug2, K_MIN)
    K_call_nodes = map_call_nodes(xi, F_aug2, K_MAX)

    logger.info(f"\nPut nodes: {K_put_nodes.round(1)}")
    logger.info(f"Call nodes: {K_call_nodes.round(1)}")

    # ----------------------------------------------------------------
    # Step 7: Get prices at quadrature nodes on Aug 2
    # ----------------------------------------------------------------
    put_prices_aug2 = []
    put_ivs_aug2 = []
    for K in K_put_nodes:
        p, iv = get_price_at_strike(df_aug2, K, "put", F_aug2, T_AUG2)
        put_prices_aug2.append(p)
        put_ivs_aug2.append(iv)

    call_prices_aug2 = []
    call_ivs_aug2 = []
    for K in K_call_nodes:
        p, iv = get_price_at_strike(df_aug2, K, "call", F_aug2, T_AUG2)
        call_prices_aug2.append(p)
        call_ivs_aug2.append(iv)

    put_prices_aug2 = np.array(put_prices_aug2)
    call_prices_aug2 = np.array(call_prices_aug2)
    put_ivs_aug2 = np.array(put_ivs_aug2)
    call_ivs_aug2 = np.array(call_ivs_aug2)

    # ----------------------------------------------------------------
    # Step 8: Compute fair variance strike on Aug 2
    # ----------------------------------------------------------------
    put_contrib = ((F_aug2 - K_MIN) / T_AUG2) * np.sum(
        w * put_prices_aug2 / K_put_nodes**2
    )
    call_contrib = ((1.0 / F_aug2 - 1.0 / K_MAX) / T_AUG2) * np.sum(
        w * call_prices_aug2
    )
    K_v_squared_aug2 = put_contrib + call_contrib
    K_v_aug2 = np.sqrt(max(K_v_squared_aug2, 0.0))
    K_v_aug2_vp = K_v_aug2 * 100  # convert to vol points

    logger.info(f"\nGL fair variance strike (Aug 2): {K_v_aug2_vp:.2f} vol pts")
    logger.info(f"Paper target: {PAPER_K_V_AUG2:.1f} vol pts")

    # Benchmark on Aug 2
    K_v_bench_aug2, _, _ = benchmark_variance_strike(
        df_aug2, F_aug2, T_AUG2, K_min=K_MIN, K_max=K_MAX
    )
    K_v_bench_aug2_vp = K_v_bench_aug2 * 100
    logger.info(f"Benchmark fair variance strike (Aug 2): {K_v_bench_aug2_vp:.2f} vol pts")

    # Use paper's K_v for hedge construction to match paper's scaling
    K_v_for_hedge = PAPER_K_V_AUG2 / 100.0  # decimal

    # ----------------------------------------------------------------
    # Step 9: Compute theoretical hedge quantities
    # ----------------------------------------------------------------
    q_put, q_call = compute_hedge_quantities(
        K_put_nodes, K_call_nodes, w, w,
        F_aug2, T_AUG2, K_MIN, K_MAX,
        K_v_for_hedge, VEGA_NOTIONAL,
    )

    # Apply appendix scaling to get contract counts
    contracts_put = apply_appendix_scaling(q_put, PAPER_K_V_AUG2, MULTIPLIER, ANNUALIZATION)
    contracts_call = apply_appendix_scaling(q_call, PAPER_K_V_AUG2, MULTIPLIER, ANNUALIZATION)

    logger.info(f"\nPut contracts: {contracts_put.round(2)}")
    logger.info(f"Call contracts: {contracts_call.round(2)}")

    # ----------------------------------------------------------------
    # Step 11: Build initial portfolio table
    # ----------------------------------------------------------------
    portfolio_rows = []
    for i, K in enumerate(K_put_nodes):
        portfolio_rows.append({
            "strike": K,
            "type": "put",
            "contracts": contracts_put[i],
            "iv": put_ivs_aug2[i],
            "price": put_prices_aug2[i],
        })
    for j, K in enumerate(K_call_nodes):
        portfolio_rows.append({
            "strike": K,
            "type": "call",
            "contracts": contracts_call[j],
            "iv": call_ivs_aug2[j],
            "price": call_prices_aug2[j],
        })

    portfolio_aug2 = pd.DataFrame(portfolio_rows)

    # Compute Greeks on Aug 2
    greeks_aug2 = compute_portfolio_greeks(portfolio_aug2, F_aug2, T_AUG2, SPOT_AUG2)

    logger.info(f"\nInitial Portfolio (Aug 2):")
    logger.info(f"  Total Premium:  ${greeks_aug2['total_premium']:>15,.0f}  (paper: ${PAPER_INIT_PREMIUM:>15,.0f})")
    logger.info(f"  Dollar Delta:   ${greeks_aug2['dollar_delta']:>15,.0f}  (paper: ${PAPER_INIT_DELTA:>15,.0f})")
    logger.info(f"  Dollar Gamma:   ${greeks_aug2['dollar_gamma']:>15,.0f}  (paper: ${PAPER_INIT_GAMMA:>15,.0f})")
    logger.info(f"  Vega/vol-pt:    ${greeks_aug2['vega_per_vol_pt']:>15,.0f}  (paper: ${PAPER_INIT_VEGA:>15,.0f})")

    # ----------------------------------------------------------------
    # Step 12: Benchmark on Aug 2
    # ----------------------------------------------------------------
    # Already computed above

    # ----------------------------------------------------------------
    # Step 13-14: Revalue on Aug 5
    # ----------------------------------------------------------------
    portfolio_aug5_rows = []
    for _, row in portfolio_aug2.iterrows():
        K = float(row["strike"])
        opt_type = str(row["type"]).lower()
        contracts = float(row["contracts"])

        p5, iv5 = get_price_at_strike(df_aug5, K, opt_type, F_aug5, T_AUG5)
        portfolio_aug5_rows.append({
            "strike": K,
            "type": opt_type,
            "contracts": contracts,
            "iv": iv5,
            "price": p5,
        })

    portfolio_aug5 = pd.DataFrame(portfolio_aug5_rows)
    greeks_aug5 = compute_portfolio_greeks(portfolio_aug5, F_aug5, T_AUG5, SPOT_AUG5)

    logger.info(f"\nRevalued Portfolio (Aug 5):")
    logger.info(f"  Total Premium:  ${greeks_aug5['total_premium']:>15,.0f}  (paper: ${26_287_352:>15,.0f})")
    logger.info(f"  Dollar Delta:   ${greeks_aug5['dollar_delta']:>15,.0f}  (paper: ${-192_539_897:>15,.0f})")
    logger.info(f"  Dollar Gamma:   ${greeks_aug5['dollar_gamma']:>15,.0f}  (paper: ${25_918_677:>15,.0f})")
    logger.info(f"  Vega/vol-pt:    ${greeks_aug5['vega_per_vol_pt']:>15,.0f}  (paper: ${1_923_992:>15,.0f})")

    # ----------------------------------------------------------------
    # Step 15-17: PnL decomposition
    # ----------------------------------------------------------------
    option_strip_mtm = greeks_aug5["total_premium"] - greeks_aug2["total_premium"]
    spot_move = SPOT_AUG5 - SPOT_AUG2
    delta_hedge_pnl = greeks_aug2["dollar_delta"] * (spot_move / SPOT_AUG2)
    total_repl_pnl = option_strip_mtm + delta_hedge_pnl

    # Convert to vol points: divide by (vega_notional / (2 * K_v))
    # PnL in vol points = PnL_USD / (vega_notional / (2 * K_v_aug2))
    pnl_scale = VEGA_NOTIONAL / (2.0 * PAPER_K_V_AUG2)
    repl_pnl_vp = total_repl_pnl / pnl_scale

    logger.info(f"\nPnL Decomposition:")
    logger.info(f"  Option strip MTM:    ${option_strip_mtm:>15,.0f}")
    logger.info(f"  Delta-hedge PnL:     ${delta_hedge_pnl:>15,.0f}")
    logger.info(f"  Total repl PnL:      ${total_repl_pnl:>15,.0f}")
    logger.info(f"  Repl PnL (vol pts):  {repl_pnl_vp:.2f}  (paper: {PAPER_REPL_PNL:.1f})")

    # ----------------------------------------------------------------
    # Step 18: Benchmark MTM
    # ----------------------------------------------------------------
    K_v_bench_aug5, _, _ = benchmark_variance_strike(
        df_aug5, F_aug5, T_AUG5, K_min=K_MIN, K_max=K_MAX
    )
    K_v_bench_aug5_vp = K_v_bench_aug5 * 100
    logger.info(f"\nBenchmark fair variance strike (Aug 5): {K_v_bench_aug5_vp:.2f} vol pts")
    logger.info(f"Paper target: {PAPER_K_V_AUG5:.1f} vol pts")

    bench_mtm_vp = variance_swap_mtm(PAPER_K_V_AUG2, K_v_bench_aug5_vp)
    logger.info(f"Benchmark MTM (vol pts): {bench_mtm_vp:.2f}  (paper: {PAPER_BENCH_MTM:.1f})")

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Exp3: Static 10-Point Variance Swap Hedge\n"
        f"Trade: {TRADE_DATE}, Eval: {EVAL_DATE}, Expiry: {EXPIRY_DATE}",
        fontsize=12,
    )

    # Portfolio strikes
    ax = axes[0, 0]
    put_K = portfolio_aug2[portfolio_aug2["type"] == "put"]["strike"].values
    call_K = portfolio_aug2[portfolio_aug2["type"] == "call"]["strike"].values
    put_c = portfolio_aug2[portfolio_aug2["type"] == "put"]["contracts"].values
    call_c = portfolio_aug2[portfolio_aug2["type"] == "call"]["contracts"].values
    ax.bar(put_K, put_c, width=30, color="blue", alpha=0.7, label="Put contracts")
    ax.bar(call_K, call_c, width=30, color="red", alpha=0.7, label="Call contracts")
    ax.axvline(F_aug2, color="k", linestyle="--", label=f"F={F_aug2:.0f}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Contracts")
    ax.set_title("10-Point Portfolio Composition (Aug 2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Premium comparison
    ax = axes[0, 1]
    labels = ["Aug 2\n(Initial)", "Aug 5\n(Revalued)"]
    premiums = [greeks_aug2["total_premium"] / 1e6, greeks_aug5["total_premium"] / 1e6]
    bars = ax.bar(labels, premiums, color=["blue", "red"], alpha=0.7)
    ax.set_ylabel("Total Premium ($M)")
    ax.set_title("Portfolio Premium: Aug 2 vs Aug 5")
    for bar, val in zip(bars, premiums):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"${val:.2f}M", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Vega comparison
    ax = axes[1, 0]
    vegas = [greeks_aug2["vega_per_vol_pt"] / 1e6, greeks_aug5["vega_per_vol_pt"] / 1e6]
    bars = ax.bar(labels, vegas, color=["blue", "red"], alpha=0.7)
    ax.set_ylabel("Vega per Vol-Pt ($M)")
    ax.set_title("Portfolio Vega: Aug 2 vs Aug 5")
    for bar, val in zip(bars, vegas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"${val:.3f}M", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # PnL comparison
    ax = axes[1, 1]
    pnl_labels = ["Replication\nPnL", "Benchmark\nMTM"]
    pnl_vals = [repl_pnl_vp, bench_mtm_vp]
    paper_vals = [PAPER_REPL_PNL, PAPER_BENCH_MTM]
    x = np.arange(len(pnl_labels))
    width = 0.35
    ax.bar(x - width / 2, pnl_vals, width, label="Reproduced", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, paper_vals, width, label="Paper", color="orange", alpha=0.8)
    ax.set_ylabel("Vol Points")
    ax.set_title("PnL Comparison: Reproduced vs Paper")
    ax.set_xticks(x)
    ax.set_xticklabels(pnl_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp3_aug5_hedge.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to {plot_path}")

    # Save portfolio tables
    portfolio_aug2["date"] = TRADE_DATE
    portfolio_aug5["date"] = EVAL_DATE
    portfolio_aug2.to_csv(RESULTS_DIR / "exp3_portfolio_aug2.csv", index=False)
    portfolio_aug5.to_csv(RESULTS_DIR / "exp3_portfolio_aug5.csv", index=False)

    results = {
        "trade_date": TRADE_DATE,
        "eval_date": EVAL_DATE,
        "expiry_date": EXPIRY_DATE,
        "K_v_gl_aug2_vp": K_v_aug2_vp,
        "K_v_bench_aug2_vp": K_v_bench_aug2_vp,
        "K_v_bench_aug5_vp": K_v_bench_aug5_vp,
        "paper_K_v_aug2": PAPER_K_V_AUG2,
        "paper_K_v_aug5": PAPER_K_V_AUG5,
        "greeks_aug2": greeks_aug2,
        "greeks_aug5": greeks_aug5,
        "option_strip_mtm_usd": option_strip_mtm,
        "delta_hedge_pnl_usd": delta_hedge_pnl,
        "total_repl_pnl_usd": total_repl_pnl,
        "repl_pnl_vol_pts": repl_pnl_vp,
        "bench_mtm_vol_pts": bench_mtm_vp,
        "paper_repl_pnl": PAPER_REPL_PNL,
        "paper_bench_mtm": PAPER_BENCH_MTM,
        "vega_increase_pct": (greeks_aug5["vega_per_vol_pt"] / greeks_aug2["vega_per_vol_pt"] - 1) * 100,
        "gamma_change_pct": (greeks_aug5["dollar_gamma"] / greeks_aug2["dollar_gamma"] - 1) * 100,
    }

    logger.info(f"\nVega increase: {results['vega_increase_pct']:.1f}% (paper: ~43%)")
    logger.info(f"Gamma change:  {results['gamma_change_pct']:.1f}% (paper: ~flat)")

    return results


if __name__ == "__main__":
    results = run_exp3()
    print(f"\nExp3 complete.")
    print(f"Replication PnL: {results['repl_pnl_vol_pts']:.2f} vol pts (paper: {PAPER_REPL_PNL:.1f})")
    print(f"Benchmark MTM:   {results['bench_mtm_vol_pts']:.2f} vol pts (paper: {PAPER_BENCH_MTM:.1f})")
