"""
Gauss-Legendre quadrature replication of variance swap fair strike.

The paper maps Gauss-Legendre nodes from [-1,1] to:
  - Put side (linear in K):  K_i = 0.5[(F + K_min) + xi_i*(F - K_min)]
  - Call side (linear in x=1/K): x_j = 0.5[(1/F + 1/K_max) + xi_j*(1/F - 1/K_max)]
                                  K_j = 1/x_j

The fair variance strike squared is then:
  K_v^2 = ((F - K_min)/T) * sum_i w_i * put(K_i^put) / (K_i^put)^2
         + ((1/F - 1/K_max)/T) * sum_j w_j * call(K_j^call)

Using numpy.polynomial.legendre.leggauss — Context7 confirmed.
"""

import numpy as np
from typing import Callable, Tuple


def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return n-point Gauss-Legendre nodes and weights on [-1, 1].

    Uses numpy.polynomial.legendre.leggauss — Context7 confirmed.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (nodes, weights) on [-1, 1]. Weights sum to 2.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    assert abs(np.sum(weights) - 2.0) < 1e-12, "GL weights must sum to 2"
    return nodes, weights


def map_put_nodes(
    xi: np.ndarray,
    F: float,
    K_min: float,
) -> np.ndarray:
    """
    Map Gauss-Legendre nodes from [-1,1] to put strikes in [K_min, F].

    K_i^put = 0.5 * [(F + K_min) + xi_i * (F - K_min)]

    Parameters
    ----------
    xi : np.ndarray
        GL nodes on [-1, 1].
    F : float
        Forward price.
    K_min : float
        Lower truncation bound.

    Returns
    -------
    np.ndarray
        Put quadrature strikes in [K_min, F].
    """
    K_put = 0.5 * ((F + K_min) + xi * (F - K_min))
    assert np.all(K_put >= K_min - 1e-10), "Put nodes must be >= K_min"
    assert np.all(K_put <= F + 1e-10), "Put nodes must be <= F"
    return K_put


def map_call_nodes(
    xi: np.ndarray,
    F: float,
    K_max: float,
) -> np.ndarray:
    """
    Map Gauss-Legendre nodes from [-1,1] to call strikes in [F, K_max].

    The call side uses the reciprocal-strike change of variable x = 1/K:
        x_j = 0.5 * [(1/F + 1/K_max) + xi_j * (1/F - 1/K_max)]
        K_j^call = 1 / x_j

    Parameters
    ----------
    xi : np.ndarray
        GL nodes on [-1, 1].
    F : float
        Forward price.
    K_max : float
        Upper truncation bound.

    Returns
    -------
    np.ndarray
        Call quadrature strikes in [F, K_max].
    """
    x_j = 0.5 * ((1.0 / F + 1.0 / K_max) + xi * (1.0 / F - 1.0 / K_max))
    K_call = 1.0 / x_j
    assert np.all(K_call >= F - 1e-6), "Call nodes must be >= F"
    assert np.all(K_call <= K_max + 1e-6), "Call nodes must be <= K_max"
    return K_call


def gauss_legendre_variance_strike(
    put_pricer: Callable[[float], float],
    call_pricer: Callable[[float], float],
    F: float,
    T: float,
    K_min: float,
    K_max: float,
    n_put: int = 5,
    n_call: int = 5,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Legendre approximation of the fair variance strike.

    Parameters
    ----------
    put_pricer : Callable[[float], float]
        Function that returns put price given strike K.
    call_pricer : Callable[[float], float]
        Function that returns call price given strike K.
    F : float
        Forward price.
    T : float
        Time to expiry in years.
    K_min : float
        Lower truncation bound.
    K_max : float
        Upper truncation bound.
    n_put : int
        Number of put quadrature nodes.
    n_call : int
        Number of call quadrature nodes.

    Returns
    -------
    Tuple containing:
        K_v : float
            Fair variance strike (vol points if inputs are in vol-point convention).
        K_v_squared : float
            Fair variance strike squared.
        K_put_nodes : np.ndarray
            Put quadrature strikes.
        K_call_nodes : np.ndarray
            Call quadrature strikes.
        w_put : np.ndarray
            Put quadrature weights.
        w_call : np.ndarray
            Call quadrature weights.
    """
    assert T > 0, "Time to expiry must be positive"
    assert K_min < F < K_max, f"Must have K_min < F < K_max, got {K_min} < {F} < {K_max}"

    xi_put, w_put = gauss_legendre_nodes_weights(n_put)
    xi_call, w_call = gauss_legendre_nodes_weights(n_call)

    K_put_nodes = map_put_nodes(xi_put, F, K_min)
    K_call_nodes = map_call_nodes(xi_call, F, K_max)

    # Put contribution: ((F - K_min) / T) * sum_i w_i * put(K_i) / K_i^2
    put_prices = np.array([put_pricer(K) for K in K_put_nodes])
    put_contrib = ((F - K_min) / T) * np.sum(w_put * put_prices / K_put_nodes**2)

    # Call contribution: ((1/F - 1/K_max) / T) * sum_j w_j * call(K_j)
    call_prices = np.array([call_pricer(K) for K in K_call_nodes])
    call_contrib = ((1.0 / F - 1.0 / K_max) / T) * np.sum(w_call * call_prices)

    K_v_squared = put_contrib + call_contrib
    K_v = np.sqrt(max(K_v_squared, 0.0))

    return K_v, K_v_squared, K_put_nodes, K_call_nodes, w_put, w_call


def compute_hedge_quantities(
    K_put_nodes: np.ndarray,
    K_call_nodes: np.ndarray,
    w_put: np.ndarray,
    w_call: np.ndarray,
    F: float,
    T: float,
    K_min: float,
    K_max: float,
    K_v: float,
    vega_notional: float = 1_000_000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translate Gauss-Legendre weights into hedge quantities for a long variance swap.

    For a long variance swap with vega notional v:
        q_i^put = (v / (2*K_v)) * ((F - K_min) / (T * (K_i^put)^2)) * w_i
        q_j^call = (v / (2*K_v)) * ((1/F - 1/K_max) / T) * w_j

    Parameters
    ----------
    K_put_nodes : np.ndarray
        Put quadrature strikes.
    K_call_nodes : np.ndarray
        Call quadrature strikes.
    w_put : np.ndarray
        Put GL weights.
    w_call : np.ndarray
        Call GL weights.
    F : float
        Forward price.
    T : float
        Time to expiry in years.
    K_min : float
        Lower truncation bound.
    K_max : float
        Upper truncation bound.
    K_v : float
        Fair variance strike (vol points).
    vega_notional : float
        Vega notional in USD.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (q_put, q_call) theoretical hedge quantities.
    """
    scale = vega_notional / (2.0 * K_v)

    q_put = scale * ((F - K_min) / (T * K_put_nodes**2)) * w_put
    q_call = scale * ((1.0 / F - 1.0 / K_max) / T) * w_call

    return q_put, q_call


def apply_appendix_scaling(
    q_theoretical: np.ndarray,
    K_v_vol_points: float,
    multiplier: float = 100.0,
    annualization: float = 252.0,
) -> np.ndarray:
    """
    Apply appendix scaling to convert theoretical quantities to contract counts.

    The paper's appendix scaling accounts for:
    - 252 trading-day annualization
    - 100 contract multiplier for SPX options
    - Variance strike quoted in vol points (factor 10000 = 100^2)

    The scaling factor is: 10000 / (annualization * multiplier)

    Parameters
    ----------
    q_theoretical : np.ndarray
        Theoretical hedge quantities from compute_hedge_quantities.
    K_v_vol_points : float
        Fair variance strike in vol points (e.g., 23.5).
    multiplier : float
        Option contract multiplier (100 for SPX).
    annualization : float
        Trading days per year.

    Returns
    -------
    np.ndarray
        Contract counts (may be fractional for theoretical purposes).
    """
    # Scaling: theoretical quantities are in decimal-vol units.
    # Paper uses vol-point convention: K_v in vol points means K_v_decimal = K_v/100.
    # The factor 10000 converts from decimal-variance to vol-point-variance.
    scaling = 10000.0 / (annualization * multiplier)
    return q_theoretical * scaling
