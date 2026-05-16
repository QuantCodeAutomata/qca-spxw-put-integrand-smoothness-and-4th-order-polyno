"""
Variance swap fair strike computation.

The fair variance strike squared is:
    K_v^2 = (2/T) [ integral_0^F put(F,K)/K^2 dK + integral_F^inf call(F,K)/K^2 dK ]

For discrete market data, we approximate with trapezoidal integration over
observed strikes in the truncation interval [K_min, K_max].

Custom — Context7 found no library equivalent for the exact variance-swap
replication formula used in the paper.
"""

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from typing import Tuple


def compute_put_integrand(
    strikes: np.ndarray,
    put_prices: np.ndarray,
) -> np.ndarray:
    """
    Compute the put-side variance-swap integrand g(K) = put(F,K) / K^2.

    Parameters
    ----------
    strikes : np.ndarray
        Array of put strike prices (K < F).
    put_prices : np.ndarray
        Corresponding put option prices.

    Returns
    -------
    np.ndarray
        Integrand values g(K) = put_price / K^2.
    """
    assert len(strikes) == len(put_prices), "strikes and put_prices must have same length"
    assert np.all(strikes > 0), "All strikes must be positive"
    return put_prices / strikes**2


def compute_call_integrand(
    strikes: np.ndarray,
    call_prices: np.ndarray,
) -> np.ndarray:
    """
    Compute the call-side variance-swap integrand h(K) = call(F,K) / K^2.

    Parameters
    ----------
    strikes : np.ndarray
        Array of call strike prices (K > F).
    call_prices : np.ndarray
        Corresponding call option prices.

    Returns
    -------
    np.ndarray
        Integrand values h(K) = call_price / K^2.
    """
    assert len(strikes) == len(call_prices), "strikes and call_prices must have same length"
    assert np.all(strikes > 0), "All strikes must be positive"
    return call_prices / strikes**2


def benchmark_variance_strike(
    df: pd.DataFrame,
    F: float,
    T: float,
    K_min: float = 500.0,
    K_max: float = 8000.0,
    strike_col: str = "strike",
    type_col: str = "type",
    price_col: str = "price",
) -> Tuple[float, float, float]:
    """
    Compute the benchmark fair variance strike using full-strip trapezoidal
    integration over observed market strikes in [K_min, K_max].

    K_v^2 = (2/T) [ trapz(put/K^2, K in [K_min, F]) + trapz(call/K^2, K in [F, K_max]) ]

    Parameters
    ----------
    df : pd.DataFrame
        Option chain with columns [strike, type, price].
    F : float
        Forward price for the expiry.
    T : float
        Time to expiry in years.
    K_min : float
        Lower truncation bound for integration.
    K_max : float
        Upper truncation bound for integration.
    strike_col : str
        Column name for strike.
    type_col : str
        Column name for option type.
    price_col : str
        Column name for option price.

    Returns
    -------
    Tuple[float, float, float]
        (K_v, K_v_squared, T) where K_v is the fair variance strike in vol points.
    """
    assert T > 0, "Time to expiry must be positive"
    assert F > 0, "Forward must be positive"

    # Put side: K < F, within [K_min, F]
    puts = df[
        (df[type_col].str.lower() == "put")
        & (df[strike_col] >= K_min)
        & (df[strike_col] < F)
    ].copy()
    puts = puts.sort_values(strike_col)
    puts = puts[puts[price_col] > 0].dropna(subset=[price_col])

    # Call side: K > F, within [F, K_max]
    calls = df[
        (df[type_col].str.lower() == "call")
        & (df[strike_col] > F)
        & (df[strike_col] <= K_max)
    ].copy()
    calls = calls.sort_values(strike_col)
    calls = calls[calls[price_col] > 0].dropna(subset=[price_col])

    put_integral = 0.0
    call_integral = 0.0

    if len(puts) >= 2:
        K_put = puts[strike_col].values.astype(float)
        P_put = puts[price_col].values.astype(float)
        integrand_put = P_put / K_put**2
        put_integral = float(trapezoid(integrand_put, K_put))

    if len(calls) >= 2:
        K_call = calls[strike_col].values.astype(float)
        P_call = calls[price_col].values.astype(float)
        integrand_call = P_call / K_call**2
        call_integral = float(trapezoid(integrand_call, K_call))

    K_v_squared = (2.0 / T) * (put_integral + call_integral)
    K_v = np.sqrt(max(K_v_squared, 0.0))
    return K_v, K_v_squared, T


def variance_swap_mtm(
    K_old: float,
    K_new: float,
) -> float:
    """
    Mark-to-market of a long variance swap position.

    MTM = (1 / (2 * K_old)) * (K_new^2 - K_old^2)

    This is the paper's convention for variance swap MTM in vol-point units.

    Parameters
    ----------
    K_old : float
        Fair variance strike at inception (vol points).
    K_new : float
        Fair variance strike at revaluation date (vol points).

    Returns
    -------
    float
        Variance swap MTM in vol points.
    """
    return (K_new**2 - K_old**2) / (2.0 * K_old)
