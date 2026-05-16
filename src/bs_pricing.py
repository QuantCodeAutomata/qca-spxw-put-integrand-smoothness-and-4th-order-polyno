"""
Black-Scholes / Black-76 pricing under the forward measure with r=0.

All pricing is done under the forward measure:
  - Forward F is given directly
  - Risk-free rate r = 0
  - Prices are undiscounted (present value = intrinsic + time value)

Custom implementation — Context7 found no library equivalent for the exact
forward-measure Black formula with r=0 convention used throughout the paper.
"""

import numpy as np
from scipy.stats import norm
from typing import Literal


def black76_price(
    F: float,
    K: float,
    T: float,
    sigma: float,
    option_type: Literal["call", "put"],
) -> float:
    """
    Black-76 option price under the forward measure with r=0.

    Parameters
    ----------
    F : float
        Forward price of the underlying.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Implied volatility (annualised, decimal).
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Undiscounted option price (= present value when r=0).
    """
    if T <= 0:
        # At expiry: intrinsic value
        if option_type == "call":
            return max(F - K, 0.0)
        return max(K - F, 0.0)

    if sigma <= 0:
        if option_type == "call":
            return max(F - K, 0.0)
        return max(K - F, 0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def black76_delta(
    F: float,
    K: float,
    T: float,
    sigma: float,
    option_type: Literal["call", "put"],
    spot: float | None = None,
) -> float:
    """
    Black-76 delta with respect to the forward (dV/dF).

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Implied volatility (decimal).
    option_type : str
        'call' or 'put'.
    spot : float or None
        If provided, delta is scaled by spot/F to give dV/dS.

    Returns
    -------
    float
        Option delta.
    """
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if F > K else 0.0
        return -1.0 if F < K else 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)

    if option_type == "call":
        delta_fwd = norm.cdf(d1)
    else:
        delta_fwd = norm.cdf(d1) - 1.0

    if spot is not None and F > 0:
        return delta_fwd * (spot / F)
    return delta_fwd


def black76_gamma(
    F: float,
    K: float,
    T: float,
    sigma: float,
) -> float:
    """
    Black-76 gamma (d²V/dF²), same for calls and puts.

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Implied volatility (decimal).

    Returns
    -------
    float
        Option gamma.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    return norm.pdf(d1) / (F * sigma * sqrt_T)


def black76_vega(
    F: float,
    K: float,
    T: float,
    sigma: float,
) -> float:
    """
    Black-76 vega (dV/d_sigma), same for calls and puts.

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    sigma : float
        Implied volatility (decimal).

    Returns
    -------
    float
        Option vega per unit of decimal vol (i.e., per 1.0 move in sigma).
        Divide by 100 to get vega per 1 vol-point (1% move).
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    return F * norm.pdf(d1) * sqrt_T


def implied_vol_bisection(
    F: float,
    K: float,
    T: float,
    market_price: float,
    option_type: Literal["call", "put"],
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """
    Compute implied volatility via bisection on the Black-76 price.

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    market_price : float
        Observed option price.
    option_type : str
        'call' or 'put'.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    float
        Implied volatility (decimal), or np.nan if not found.
    """
    intrinsic = max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
    if market_price < intrinsic - tol:
        return np.nan
    if T <= 0:
        return np.nan

    lo, hi = 1e-6, 10.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = black76_price(F, K, T, mid, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
