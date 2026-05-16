"""
Experiment 4: Synthetic 1-Year Skew-Shock Hedging Test with an 8-Point
Gauss-Legendre Portfolio.

Fully synthetic setting:
- F = 100, T = 1 year, sigma = 20%
- Stylized skew shock applied to puts and calls
- 8-point GL portfolio (4 puts + 4 calls)
- Compare portfolio MTM vs benchmark variance swap MTM

Paper's reported synthetic strike locations:
  Puts:  ~16.2, 39.7, 70.3, 93.8
  Calls: ~103.6, 119.8, 150.4, 187.0
"""

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bs_pricing import black76_price, black76_delta, black76_gamma, black76_vega
from src.synthetic_surface import base_vol, shocked_vol
from src.variance_swap import variance_swap_mtm
from src.gauss_legendre import (
    gauss_legendre_nodes_weights,
    map_put_nodes,
    map_call_nodes,
    compute_hedge_quantities,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Synthetic parameters (paper specification)
F = 100.0
T = 1.0
SIGMA = 0.20
DELTA_SIGMA_PUT = 0.30
DELTA_SIGMA_CALL = -0.15
M_PUT = 3.0
M_CALL = 3.0
K_MIN = 10.0
K_MAX = 200.0
N_BENCH_STRIKES = 200
N_PUT = 4
N_CALL = 4

# Paper's reported synthetic strike locations
PAPER_PUT_STRIKES = [16.2, 39.7, 70.3, 93.8]
PAPER_CALL_STRIKES = [103.6, 119.8, 150.4, 187.0]


def price_option_synthetic(
    K: float,
    F: float,
    T: float,
    sigma_fn,
    option_type: str,
) -> float:
    """
    Price a European option using Black-76 with the given vol function.

    Parameters
    ----------
    K : float
        Strike.
    F : float
        Forward.
    T : float
        Time to expiry.
    sigma_fn : callable
        Function(K) -> implied vol.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Option price.
    """
    iv = sigma_fn(K)
    return black76_price(F, K, T, iv, option_type)


def build_benchmark_grid(
    F: float,
    T: float,
    K_min: float,
    K_max: float,
    n_strikes: int,
    sigma_fn,
) -> tuple:
    """
    Build a dense benchmark grid and compute the variance swap fair strike.

    Parameters
    ----------
    F : float
        Forward.
    T : float
        Time to expiry.
    K_min : float
        Lower bound.
    K_max : float
        Upper bound.
    n_strikes : int
        Number of benchmark strikes.
    sigma_fn : callable
        Volatility function.

    Returns
    -------
    tuple
        (K_v, K_v_squared, K_arr, integrand_put, integrand_call)
    """
    K_arr = np.linspace(K_min, K_max, n_strikes)

    put_integrands = []
    call_integrands = []
    K_put_arr = []
    K_call_arr = []

    for K in K_arr:
        if K < F:
            p = price_option_synthetic(K, F, T, sigma_fn, "put")
            put_integrands.append(p / K**2)
            K_put_arr.append(K)
        else:
            c = price_option_synthetic(K, F, T, sigma_fn, "call")
            call_integrands.append(c / K**2)
            K_call_arr.append(K)

    K_put_arr = np.array(K_put_arr)
    K_call_arr = np.array(K_call_arr)
    put_integrands = np.array(put_integrands)
    call_integrands = np.array(call_integrands)

    # Trapezoidal integration
    from scipy.integrate import trapezoid
    put_integral = trapezoid(put_integrands, K_put_arr) if len(K_put_arr) >= 2 else 0.0
    call_integral = trapezoid(call_integrands, K_call_arr) if len(K_call_arr) >= 2 else 0.0

    K_v_squared = (2.0 / T) * (put_integral + call_integral)
    K_v = np.sqrt(max(K_v_squared, 0.0))

    return K_v, K_v_squared, K_arr, K_put_arr, K_call_arr, put_integrands, call_integrands


def run_exp4() -> dict:
    """
    Run Experiment 4: Synthetic 1-year skew-shock hedging test.

    Returns
    -------
    dict
        Results with benchmark and GL MTM comparison.
    """
    logger.info("=" * 60)
    logger.info("EXP4: Synthetic 1-Year Skew-Shock Hedging Test")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # Step 1-2: Base surface (flat vol)
    # ----------------------------------------------------------------
    sigma_base = lambda K: base_vol(K, SIGMA)

    # ----------------------------------------------------------------
    # Step 3: Post-shock surface
    # ----------------------------------------------------------------
    sigma_shocked = lambda K: shocked_vol(
        K, F, T, SIGMA, DELTA_SIGMA_PUT, DELTA_SIGMA_CALL, M_PUT, M_CALL
    )

    # ----------------------------------------------------------------
    # Step 4-6: Benchmark pre-shock fair variance strike
    # ----------------------------------------------------------------
    (K_v_pre, K_v_sq_pre,
     K_arr, K_put_arr, K_call_arr,
     put_integ_pre, call_integ_pre) = build_benchmark_grid(
        F, T, K_MIN, K_MAX, N_BENCH_STRIKES, sigma_base
    )

    logger.info(f"\nBenchmark pre-shock K_v = {K_v_pre:.4f} (expected ≈ {SIGMA:.4f})")
    logger.info(f"  (flat vol surface should give K_v ≈ sigma = {SIGMA:.4f})")

    # ----------------------------------------------------------------
    # Step 5-6: Benchmark post-shock fair variance strike
    # ----------------------------------------------------------------
    (K_v_post, K_v_sq_post,
     _, _, _,
     put_integ_post, call_integ_post) = build_benchmark_grid(
        F, T, K_MIN, K_MAX, N_BENCH_STRIKES, sigma_shocked
    )

    logger.info(f"Benchmark post-shock K_v = {K_v_post:.4f}")

    # ----------------------------------------------------------------
    # Step 7: Benchmark variance swap MTM
    # ----------------------------------------------------------------
    bench_mtm = variance_swap_mtm(K_v_pre, K_v_post)
    logger.info(f"Benchmark variance swap MTM = {bench_mtm:.6f}")
    logger.info(f"  (in vol units; multiply by 100 for vol-point convention)")

    # ----------------------------------------------------------------
    # Step 8: 8-point GL portfolio (4 puts + 4 calls)
    # ----------------------------------------------------------------
    xi, w = gauss_legendre_nodes_weights(N_PUT)

    K_put_nodes = map_put_nodes(xi, F, K_MIN)
    K_call_nodes = map_call_nodes(xi, F, K_MAX)

    logger.info(f"\n8-point GL nodes:")
    logger.info(f"  Put nodes:  {K_put_nodes.round(2)}")
    logger.info(f"  Call nodes: {K_call_nodes.round(2)}")

    # ----------------------------------------------------------------
    # Step 9: Verify against paper's reported strike locations
    # ----------------------------------------------------------------
    logger.info(f"\nPaper's reported put strikes:  {PAPER_PUT_STRIKES}")
    logger.info(f"Paper's reported call strikes: {PAPER_CALL_STRIKES}")

    put_diffs = np.abs(np.sort(K_put_nodes) - np.sort(PAPER_PUT_STRIKES))
    call_diffs = np.abs(np.sort(K_call_nodes) - np.sort(PAPER_CALL_STRIKES))
    logger.info(f"Put strike differences:  {put_diffs.round(3)}")
    logger.info(f"Call strike differences: {call_diffs.round(3)}")
    logger.info(f"Max put diff: {put_diffs.max():.3f}, Max call diff: {call_diffs.max():.3f}")

    # ----------------------------------------------------------------
    # Step 10: Pre-shock GL fair variance strike
    # ----------------------------------------------------------------
    put_prices_pre = np.array([
        price_option_synthetic(K, F, T, sigma_base, "put") for K in K_put_nodes
    ])
    call_prices_pre = np.array([
        price_option_synthetic(K, F, T, sigma_base, "call") for K in K_call_nodes
    ])

    put_contrib_pre = ((F - K_MIN) / T) * np.sum(w * put_prices_pre / K_put_nodes**2)
    call_contrib_pre = ((1.0 / F - 1.0 / K_MAX) / T) * np.sum(w * call_prices_pre)
    K_v_gl_sq_pre = put_contrib_pre + call_contrib_pre
    K_v_gl_pre = np.sqrt(max(K_v_gl_sq_pre, 0.0))

    logger.info(f"\nGL pre-shock K_v = {K_v_gl_pre:.4f}")
    logger.info(f"Benchmark pre-shock K_v = {K_v_pre:.4f}")
    logger.info(f"Difference: {abs(K_v_gl_pre - K_v_pre):.6f}")

    # ----------------------------------------------------------------
    # Step 11: Hedge quantities (v=1 for normalized comparison)
    # ----------------------------------------------------------------
    q_put, q_call = compute_hedge_quantities(
        K_put_nodes, K_call_nodes, w, w,
        F, T, K_MIN, K_MAX,
        K_v_gl_pre, vega_notional=1.0,
    )

    # ----------------------------------------------------------------
    # Step 12: Post-shock prices and portfolio MTM
    # ----------------------------------------------------------------
    put_prices_post = np.array([
        price_option_synthetic(K, F, T, sigma_shocked, "put") for K in K_put_nodes
    ])
    call_prices_post = np.array([
        price_option_synthetic(K, F, T, sigma_shocked, "call") for K in K_call_nodes
    ])

    # Portfolio MTM = sum of (post - pre) * quantity
    put_mtm = np.sum(q_put * (put_prices_post - put_prices_pre))
    call_mtm = np.sum(q_call * (call_prices_post - call_prices_pre))
    portfolio_mtm = put_mtm + call_mtm

    logger.info(f"\nPortfolio MTM (8-point GL): {portfolio_mtm:.6f}")
    logger.info(f"Benchmark variance swap MTM: {bench_mtm:.6f}")
    logger.info(f"Difference: {abs(portfolio_mtm - bench_mtm):.6f}")
    logger.info(f"Relative error: {abs(portfolio_mtm - bench_mtm) / abs(bench_mtm) * 100:.4f}%")

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Exp4: Synthetic 1-Year Skew-Shock Hedging Test\n"
        f"F={F}, T={T}yr, σ_base={SIGMA:.0%}, "
        f"Δσ_put={DELTA_SIGMA_PUT:.0%}, Δσ_call={DELTA_SIGMA_CALL:.0%}",
        fontsize=12,
    )

    # Vol surface: pre vs post shock
    ax = axes[0, 0]
    K_plot = np.linspace(K_MIN, K_MAX, 300)
    vol_pre = np.array([sigma_base(K) for K in K_plot]) * 100
    vol_post = np.array([sigma_shocked(K) for K in K_plot]) * 100
    ax.plot(K_plot, vol_pre, "b-", linewidth=2, label="Pre-shock (flat 20%)")
    ax.plot(K_plot, vol_post, "r-", linewidth=2, label="Post-shock (skew)")
    ax.axvline(F, color="k", linestyle="--", alpha=0.5, label=f"F={F}")
    ax.scatter(K_put_nodes, [sigma_base(K) * 100 for K in K_put_nodes],
               color="blue", s=80, zorder=5, label="Put nodes")
    ax.scatter(K_call_nodes, [sigma_base(K) * 100 for K in K_call_nodes],
               color="red", s=80, zorder=5, label="Call nodes")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title("Implied Volatility Surface: Pre vs Post Shock")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Integrand: pre vs post shock
    ax = axes[0, 1]
    ax.plot(K_put_arr, put_integ_pre * 1e4, "b-", linewidth=1.5, label="Put pre-shock")
    ax.plot(K_put_arr, put_integ_post * 1e4, "b--", linewidth=1.5, label="Put post-shock")
    ax.plot(K_call_arr, call_integ_pre * 1e4, "r-", linewidth=1.5, label="Call pre-shock")
    ax.plot(K_call_arr, call_integ_post * 1e4, "r--", linewidth=1.5, label="Call post-shock")
    ax.axvline(F, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Integrand × 10⁴")
    ax.set_title("Variance-Swap Integrand: Pre vs Post Shock")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # GL nodes on integrand
    ax = axes[1, 0]
    ax.plot(K_put_arr, put_integ_pre * 1e4, "b-", linewidth=1.5, alpha=0.7)
    ax.plot(K_call_arr, call_integ_pre * 1e4, "r-", linewidth=1.5, alpha=0.7)
    put_integ_at_nodes = np.array([
        price_option_synthetic(K, F, T, sigma_base, "put") / K**2 for K in K_put_nodes
    ])
    call_integ_at_nodes = np.array([
        price_option_synthetic(K, F, T, sigma_base, "call") / K**2 for K in K_call_nodes
    ])
    ax.scatter(K_put_nodes, put_integ_at_nodes * 1e4, color="blue", s=100,
               zorder=5, label=f"Put GL nodes (n={N_PUT})")
    ax.scatter(K_call_nodes, call_integ_at_nodes * 1e4, color="red", s=100,
               zorder=5, label=f"Call GL nodes (n={N_CALL})")
    ax.axvline(F, color="k", linestyle="--", alpha=0.5, label=f"F={F}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Integrand × 10⁴")
    ax.set_title("8-Point GL Nodes on Pre-Shock Integrand")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # MTM comparison
    ax = axes[1, 1]
    mtm_labels = ["8-Point GL\nPortfolio", "Benchmark\nVariance Swap"]
    mtm_vals = [portfolio_mtm, bench_mtm]
    colors = ["steelblue", "orange"]
    bars = ax.bar(mtm_labels, mtm_vals, color=colors, alpha=0.8)
    ax.set_ylabel("MTM (normalized, v=1)")
    ax.set_title(
        f"MTM Comparison\n"
        f"Rel. Error = {abs(portfolio_mtm - bench_mtm) / abs(bench_mtm) * 100:.3f}%"
    )
    for bar, val in zip(bars, mtm_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp4_synthetic_skew_shock.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to {plot_path}")

    # Save detailed results table
    node_table = pd.DataFrame({
        "type": ["put"] * N_PUT + ["call"] * N_CALL,
        "strike": list(K_put_nodes) + list(K_call_nodes),
        "paper_strike": PAPER_PUT_STRIKES + PAPER_CALL_STRIKES,
        "strike_diff": list(np.sort(K_put_nodes) - np.sort(PAPER_PUT_STRIKES)) +
                       list(np.sort(K_call_nodes) - np.sort(PAPER_CALL_STRIKES)),
        "price_pre": list(put_prices_pre) + list(call_prices_pre),
        "price_post": list(put_prices_post) + list(call_prices_post),
        "quantity": list(q_put) + list(q_call),
        "mtm_contribution": list(q_put * (put_prices_post - put_prices_pre)) +
                            list(q_call * (call_prices_post - call_prices_pre)),
    })
    node_table.to_csv(RESULTS_DIR / "exp4_node_table.csv", index=False)

    results = {
        "F": F,
        "T": T,
        "sigma_base": SIGMA,
        "K_v_pre_benchmark": K_v_pre,
        "K_v_post_benchmark": K_v_post,
        "K_v_pre_gl": K_v_gl_pre,
        "benchmark_mtm": bench_mtm,
        "portfolio_mtm": portfolio_mtm,
        "relative_error_pct": abs(portfolio_mtm - bench_mtm) / abs(bench_mtm) * 100,
        "K_put_nodes": K_put_nodes.tolist(),
        "K_call_nodes": K_call_nodes.tolist(),
        "paper_put_strikes": PAPER_PUT_STRIKES,
        "paper_call_strikes": PAPER_CALL_STRIKES,
        "max_put_strike_diff": float(put_diffs.max()),
        "max_call_strike_diff": float(call_diffs.max()),
        "node_table": node_table,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULT SUMMARY:")
    logger.info(f"  Pre-shock K_v (benchmark): {K_v_pre:.4f}")
    logger.info(f"  Post-shock K_v (benchmark): {K_v_post:.4f}")
    logger.info(f"  Benchmark MTM: {bench_mtm:.6f}")
    logger.info(f"  8-point GL MTM: {portfolio_mtm:.6f}")
    logger.info(f"  Relative error: {results['relative_error_pct']:.4f}%")

    return results


if __name__ == "__main__":
    results = run_exp4()
    print(f"\nExp4 complete.")
    print(f"Benchmark MTM:  {results['benchmark_mtm']:.6f}")
    print(f"GL Portfolio MTM: {results['portfolio_mtm']:.6f}")
    print(f"Relative error: {results['relative_error_pct']:.4f}%")
