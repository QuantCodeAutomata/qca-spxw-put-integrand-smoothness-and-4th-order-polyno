"""
Polynomial fitting and goodness-of-fit statistics for the variance-swap integrand.

Uses numpy.polyfit for least-squares polynomial fitting.
Using numpy.polyfit — Context7 confirmed.
"""

import numpy as np
from typing import Dict, List, Tuple


def fit_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a polynomial of given degree to (x, y) data using least squares.

    Uses numpy.polyfit — Context7 confirmed.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g., strike prices).
    y : np.ndarray
        Dependent variable (e.g., integrand values).
    degree : int
        Polynomial degree.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (coefficients, fitted_values)
        Coefficients are in descending power order (numpy convention).
    """
    assert len(x) == len(y), "x and y must have same length"
    assert len(x) > degree, f"Need more data points than polynomial degree {degree}"

    coeffs = np.polyfit(x, y, degree)
    fitted = np.polyval(coeffs, x)
    return coeffs, fitted


def compute_r_squared(
    y_actual: np.ndarray,
    y_fitted: np.ndarray,
) -> float:
    """
    Compute the coefficient of determination R².

    R² = 1 - SSE / SST
    where SSE = sum((y - y_hat)^2) and SST = sum((y - mean(y))^2).

    Parameters
    ----------
    y_actual : np.ndarray
        Observed values.
    y_fitted : np.ndarray
        Fitted/predicted values.

    Returns
    -------
    float
        R² value in [0, 1] for a good fit (can be negative for very poor fits).
    """
    residuals = y_actual - y_fitted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y_actual - np.mean(y_actual))**2))
    if sst == 0:
        return 1.0 if sse == 0 else 0.0
    return 1.0 - sse / sst


def compute_fit_statistics(
    x: np.ndarray,
    y: np.ndarray,
    y_fitted: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive goodness-of-fit statistics.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    y : np.ndarray
        Observed dependent variable.
    y_fitted : np.ndarray
        Fitted values.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: r_squared, sse, sst, rmse, mae, max_abs_residual.
    """
    residuals = y - y_fitted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    n = len(y)

    return {
        "r_squared": r2,
        "sse": sse,
        "sst": sst,
        "rmse": float(np.sqrt(sse / n)),
        "mae": float(np.mean(np.abs(residuals))),
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "n_points": n,
    }


def fit_multiple_degrees(
    x: np.ndarray,
    y: np.ndarray,
    degrees: List[int],
) -> Dict[int, Dict]:
    """
    Fit polynomials of multiple degrees and return statistics for each.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    y : np.ndarray
        Dependent variable.
    degrees : List[int]
        List of polynomial degrees to fit.

    Returns
    -------
    Dict[int, Dict]
        Mapping from degree to dict with keys: coeffs, fitted, stats.
    """
    results = {}
    for deg in degrees:
        if len(x) <= deg:
            continue
        coeffs, fitted = fit_polynomial(x, y, deg)
        stats = compute_fit_statistics(x, y, fitted)
        results[deg] = {
            "coeffs": coeffs,
            "fitted": fitted,
            "stats": stats,
        }
    return results
