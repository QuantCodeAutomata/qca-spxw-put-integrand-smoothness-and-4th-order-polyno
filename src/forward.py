"""
Forward price estimation from put-call parity under r=0.

Under r=0, put-call parity gives:
    C(K) - P(K) = F - K

So for each strike K with both a call and put price:
    F_estimate(K) = C(K) - P(K) + K

We aggregate across near-ATM strikes to get a robust estimate.

Custom — Context7 found no library equivalent (paper methodology, r=0 parity).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from typing import Optional


def estimate_forward_parity(
    df: pd.DataFrame,
    strike_col: str = "strike",
    type_col: str = "type",
    price_col: str = "price",
    spot: Optional[float] = None,
    atm_band_pct: float = 0.10,
) -> float:
    """
    Estimate the forward price from put-call parity under r=0.

    For each strike K where both a call and put are available:
        F_hat(K) = C(K) - P(K) + K

    The final estimate is the median of F_hat values over near-ATM strikes.

    Parameters
    ----------
    df : pd.DataFrame
        Option chain with columns [strike, type, price].
    strike_col : str
        Column name for strike.
    type_col : str
        Column name for option type ('call' or 'put').
    price_col : str
        Column name for option price.
    spot : float or None
        Spot price used to define the ATM band. If None, uses median strike.
    atm_band_pct : float
        Fraction of spot defining the near-ATM band for parity estimation.

    Returns
    -------
    float
        Estimated forward price.
    """
    calls = df[df[type_col].str.lower() == "call"].set_index(strike_col)[price_col]
    puts = df[df[type_col].str.lower() == "put"].set_index(strike_col)[price_col]

    common_strikes = calls.index.intersection(puts.index)
    if len(common_strikes) == 0:
        raise ValueError("No common strikes found for put-call parity estimation.")

    f_estimates = calls[common_strikes] - puts[common_strikes] + common_strikes

    # Filter to near-ATM strikes
    ref = spot if spot is not None else float(np.median(common_strikes))
    lo = ref * (1 - atm_band_pct)
    hi = ref * (1 + atm_band_pct)
    near_atm = f_estimates[(f_estimates.index >= lo) & (f_estimates.index <= hi)]

    if len(near_atm) == 0:
        # Fall back to all common strikes
        near_atm = f_estimates

    return float(np.median(near_atm))


def estimate_forward_regression(
    df: pd.DataFrame,
    strike_col: str = "strike",
    type_col: str = "type",
    price_col: str = "price",
) -> float:
    """
    Estimate forward by minimising sum of squared parity errors over all
    common strikes.

    Minimises: sum_K [ (C(K) - P(K)) - (F - K) ]^2  over F.

    Parameters
    ----------
    df : pd.DataFrame
        Option chain with columns [strike, type, price].
    strike_col : str
        Column name for strike.
    type_col : str
        Column name for option type.
    price_col : str
        Column name for option price.

    Returns
    -------
    float
        Estimated forward price.
    """
    calls = df[df[type_col].str.lower() == "call"].set_index(strike_col)[price_col]
    puts = df[df[type_col].str.lower() == "put"].set_index(strike_col)[price_col]

    common_strikes = calls.index.intersection(puts.index)
    if len(common_strikes) == 0:
        raise ValueError("No common strikes found for forward regression.")

    K = np.array(common_strikes, dtype=float)
    parity_lhs = np.array(calls[common_strikes] - puts[common_strikes], dtype=float)

    # Analytical solution: F = mean(C(K) - P(K) + K)
    F_hat = float(np.mean(parity_lhs + K))
    return F_hat
