"""
Synthetic implied volatility surface for Exp4.

Base surface: flat vol sigma = 0.20 for all strikes.

Post-shock surface (paper's stylized skew shock):
  For puts (K < F):
    sigma_post(K) = sigma - delta_sigma_put * tanh( ln(K/F) / (m_put * sigma * sqrt(T)) )
  For calls (K > F):
    sigma_post(K) = sigma + delta_sigma_call * tanh( ln(K/F) / (m_call * sigma * sqrt(T)) )

Custom — Context7 found no library equivalent (paper Eq. skew shock definition).
"""

import numpy as np
from typing import Literal


def base_vol(
    K: float | np.ndarray,
    sigma: float = 0.20,
) -> float | np.ndarray:
    """
    Flat base implied volatility surface.

    Parameters
    ----------
    K : float or np.ndarray
        Strike(s).
    sigma : float
        Base volatility level (decimal).

    Returns
    -------
    float or np.ndarray
        Implied volatility at each strike.
    """
    if isinstance(K, np.ndarray):
        return np.full_like(K, sigma, dtype=float)
    return sigma


def shocked_vol(
    K: float | np.ndarray,
    F: float,
    T: float,
    sigma: float = 0.20,
    delta_sigma_put: float = 0.30,
    delta_sigma_call: float = -0.15,
    m_put: float = 3.0,
    m_call: float = 3.0,
) -> float | np.ndarray:
    """
    Post-shock implied volatility surface using the paper's stylized skew shock.

    For puts (K < F):
        sigma_post(K) = sigma - delta_sigma_put * tanh( ln(K/F) / (m_put * sigma * sqrt(T)) )
    For calls (K > F):
        sigma_post(K) = sigma + delta_sigma_call * tanh( ln(K/F) / (m_call * sigma * sqrt(T)) )
    At K = F: use the put formula (continuous at ATM since tanh(0) = 0).

    Parameters
    ----------
    K : float or np.ndarray
        Strike(s).
    F : float
        Forward price.
    T : float
        Time to expiry in years.
    sigma : float
        Base volatility (decimal).
    delta_sigma_put : float
        Put shock amplitude (positive = downside skew steepening).
    delta_sigma_call : float
        Call shock amplitude (negative = upside skew flattening).
    m_put : float
        Put shock slope parameter.
    m_call : float
        Call shock slope parameter.

    Returns
    -------
    float or np.ndarray
        Post-shock implied volatility at each strike.
    """
    scalar_input = np.isscalar(K)
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))

    sqrt_T = np.sqrt(T)
    log_moneyness = np.log(K_arr / F)

    # Put formula applied to K <= F
    put_mask = K_arr <= F
    call_mask = K_arr > F

    result = np.empty_like(K_arr)

    # Put side
    if np.any(put_mask):
        denom_put = m_put * sigma * sqrt_T
        result[put_mask] = sigma - delta_sigma_put * np.tanh(
            log_moneyness[put_mask] / denom_put
        )

    # Call side
    if np.any(call_mask):
        denom_call = m_call * sigma * sqrt_T
        result[call_mask] = sigma + delta_sigma_call * np.tanh(
            log_moneyness[call_mask] / denom_call
        )

    # Validate: all vols must be positive
    assert np.all(result > 0), "Post-shock vols must remain positive"

    if scalar_input:
        return float(result[0])
    return result
