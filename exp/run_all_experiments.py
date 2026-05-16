"""
Main runner for all variance swap replication experiments.

Runs Exp1 through Exp4 sequentially and saves all results to results/.
"""

import sys
import logging
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "results" / "experiment_run.log"),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_all():
    """Run all experiments and save results."""
    all_results = {}

    # ----------------------------------------------------------------
    # Experiment 1
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING EXPERIMENT 1: Put-Integrand Smoothness")
    logger.info("=" * 70)
    try:
        from exp.exp1_put_integrand_smoothness import run_exp1
        r1 = run_exp1()
        all_results["exp1"] = {
            "r_squared_4th_order": r1["r_squared_4th_order"],
            "paper_target_r2": r1["paper_target_r2"],
            "forward_F": r1["forward_F"],
            "n_put_strikes": r1["n_put_strikes"],
            "strike_range": list(r1["strike_range"]),
            "multi_degree_r2": r1["multi_degree_r2"],
            "data_source": r1["data_source"],
        }
        logger.info(f"Exp1 DONE: R²(4th order) = {r1['r_squared_4th_order']:.6f}")
    except Exception as e:
        logger.error(f"Exp1 FAILED: {e}", exc_info=True)
        all_results["exp1"] = {"error": str(e)}

    # ----------------------------------------------------------------
    # Experiment 2
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING EXPERIMENT 2: Daily 2024 Variance Strike Comparison")
    logger.info("=" * 70)
    try:
        from exp.exp2_daily_variance_strike_comparison import run_exp2
        r2 = run_exp2()
        all_results["exp2"] = {
            "n_trading_days": r2["n_trading_days"],
            "mae_vol_pts": r2["mae_vol_pts"],
            "rmse_vol_pts": r2["rmse_vol_pts"],
            "max_abs_error_vol_pts": r2["max_abs_error_vol_pts"],
            "mean_relative_error_pct": r2["mean_relative_error_pct"],
            "data_source": r2["data_source"],
        }
        logger.info(f"Exp2 DONE: MAE = {r2['mae_vol_pts']:.4f} vol pts")
    except Exception as e:
        logger.error(f"Exp2 FAILED: {e}", exc_info=True)
        all_results["exp2"] = {"error": str(e)}

    # ----------------------------------------------------------------
    # Experiment 3
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING EXPERIMENT 3: Aug 5 Shock Hedge")
    logger.info("=" * 70)
    try:
        from exp.exp3_aug5_hedge import run_exp3
        r3 = run_exp3()
        all_results["exp3"] = {
            "K_v_gl_aug2_vp": r3["K_v_gl_aug2_vp"],
            "K_v_bench_aug2_vp": r3["K_v_bench_aug2_vp"],
            "K_v_bench_aug5_vp": r3["K_v_bench_aug5_vp"],
            "paper_K_v_aug2": r3["paper_K_v_aug2"],
            "paper_K_v_aug5": r3["paper_K_v_aug5"],
            "repl_pnl_vol_pts": r3["repl_pnl_vol_pts"],
            "bench_mtm_vol_pts": r3["bench_mtm_vol_pts"],
            "paper_repl_pnl": r3["paper_repl_pnl"],
            "paper_bench_mtm": r3["paper_bench_mtm"],
            "vega_increase_pct": r3["vega_increase_pct"],
            "gamma_change_pct": r3["gamma_change_pct"],
            "greeks_aug2": {k: float(v) for k, v in r3["greeks_aug2"].items()},
            "greeks_aug5": {k: float(v) for k, v in r3["greeks_aug5"].items()},
        }
        logger.info(f"Exp3 DONE: Repl PnL = {r3['repl_pnl_vol_pts']:.2f} vol pts")
    except Exception as e:
        logger.error(f"Exp3 FAILED: {e}", exc_info=True)
        all_results["exp3"] = {"error": str(e)}

    # ----------------------------------------------------------------
    # Experiment 4
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING EXPERIMENT 4: Synthetic Skew-Shock Hedging Test")
    logger.info("=" * 70)
    try:
        from exp.exp4_synthetic_skew_shock import run_exp4
        r4 = run_exp4()
        all_results["exp4"] = {
            "K_v_pre_benchmark": r4["K_v_pre_benchmark"],
            "K_v_post_benchmark": r4["K_v_post_benchmark"],
            "K_v_pre_gl": r4["K_v_pre_gl"],
            "benchmark_mtm": r4["benchmark_mtm"],
            "portfolio_mtm": r4["portfolio_mtm"],
            "relative_error_pct": r4["relative_error_pct"],
            "K_put_nodes": r4["K_put_nodes"],
            "K_call_nodes": r4["K_call_nodes"],
            "paper_put_strikes": r4["paper_put_strikes"],
            "paper_call_strikes": r4["paper_call_strikes"],
            "max_put_strike_diff": r4["max_put_strike_diff"],
            "max_call_strike_diff": r4["max_call_strike_diff"],
        }
        logger.info(f"Exp4 DONE: Relative error = {r4['relative_error_pct']:.4f}%")
    except Exception as e:
        logger.error(f"Exp4 FAILED: {e}", exc_info=True)
        all_results["exp4"] = {"error": str(e)}

    # ----------------------------------------------------------------
    # Save JSON summary
    # ----------------------------------------------------------------
    json_path = RESULTS_DIR / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nAll results saved to {json_path}")

    return all_results


if __name__ == "__main__":
    results = run_all()
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    for exp_id, res in results.items():
        if "error" in res:
            print(f"  {exp_id}: FAILED - {res['error']}")
        else:
            print(f"  {exp_id}: OK")
