"""
Tests for src/gauss_legendre.py

Validates GL node/weight generation, strike mapping, and variance strike computation.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gauss_legendre import (
    gauss_legendre_nodes_weights,
    map_put_nodes,
    map_call_nodes,
    gauss_legendre_variance_strike,
    compute_hedge_quantities,
    apply_appendix_scaling,
)
from src.bs_pricing import black76_price


class TestGaussLegendreNodesWeights:
    """Tests for GL node and weight generation."""

    def test_weights_sum_to_2(self):
        """GL weights must sum to 2 on [-1, 1]."""
        for n in [2, 3, 4, 5, 8, 10]:
            _, w = gauss_legendre_nodes_weights(n)
            assert abs(np.sum(w) - 2.0) < 1e-12, f"Weights don't sum to 2 for n={n}"

    def test_nodes_in_minus1_plus1(self):
        """GL nodes must lie in (-1, 1)."""
        for n in [2, 3, 4, 5, 8, 10]:
            xi, _ = gauss_legendre_nodes_weights(n)
            assert np.all(xi > -1.0) and np.all(xi < 1.0), f"Nodes out of range for n={n}"

    def test_nodes_symmetric(self):
        """GL nodes are symmetric around 0."""
        for n in [2, 4, 6]:
            xi, _ = gauss_legendre_nodes_weights(n)
            xi_sorted = np.sort(xi)
            assert np.allclose(xi_sorted, -xi_sorted[::-1], atol=1e-12)

    def test_weights_positive(self):
        """GL weights must be positive."""
        for n in [2, 3, 4, 5]:
            _, w = gauss_legendre_nodes_weights(n)
            assert np.all(w > 0)

    def test_exact_integration_polynomial(self):
        """n-point GL integrates polynomials of degree <= 2n-1 exactly."""
        # Test: integrate x^3 from -1 to 1 = 0
        n = 2  # 2-point GL integrates up to degree 3 exactly
        xi, w = gauss_legendre_nodes_weights(n)
        result = np.sum(w * xi**3)
        assert abs(result) < 1e-12

        # Test: integrate x^2 from -1 to 1 = 2/3
        result = np.sum(w * xi**2)
        assert abs(result - 2.0 / 3.0) < 1e-12

    def test_5point_integrates_degree9(self):
        """5-point GL integrates polynomials of degree <= 9 exactly."""
        xi, w = gauss_legendre_nodes_weights(5)
        # Integrate x^9 from -1 to 1 = 0 (odd function)
        result = np.sum(w * xi**9)
        assert abs(result) < 1e-10
        # Integrate x^8 from -1 to 1 = 2/9
        result = np.sum(w * xi**8)
        assert abs(result - 2.0 / 9.0) < 1e-10


class TestPutNodeMapping:
    """Tests for put node mapping to [K_min, F]."""

    def test_nodes_in_range(self):
        """Put nodes must lie in [K_min, F]."""
        xi, _ = gauss_legendre_nodes_weights(5)
        F, K_min = 6120.0, 500.0
        K_put = map_put_nodes(xi, F, K_min)
        assert np.all(K_put >= K_min - 1e-10)
        assert np.all(K_put <= F + 1e-10)

    def test_midpoint_maps_to_midpoint(self):
        """xi=0 should map to (F + K_min) / 2."""
        F, K_min = 100.0, 10.0
        K_mid = map_put_nodes(np.array([0.0]), F, K_min)
        assert abs(K_mid[0] - (F + K_min) / 2.0) < 1e-10

    def test_xi_minus1_maps_to_K_min(self):
        """xi=-1 should map to K_min."""
        F, K_min = 100.0, 10.0
        K = map_put_nodes(np.array([-1.0]), F, K_min)
        assert abs(K[0] - K_min) < 1e-10

    def test_xi_plus1_maps_to_F(self):
        """xi=+1 should map to F."""
        F, K_min = 100.0, 10.0
        K = map_put_nodes(np.array([1.0]), F, K_min)
        assert abs(K[0] - F) < 1e-10

    def test_synthetic_paper_strikes_n4(self):
        """Verify 4-point put nodes match paper's reported values for F=100, K_min=10."""
        xi, _ = gauss_legendre_nodes_weights(4)
        F, K_min = 100.0, 10.0
        K_put = np.sort(map_put_nodes(xi, F, K_min))
        paper_puts = np.sort([16.2, 39.7, 70.3, 93.8])
        # Allow tolerance of 1.0 for rounding in paper
        assert np.all(np.abs(K_put - paper_puts) < 1.5), (
            f"Put nodes {K_put} don't match paper {paper_puts}"
        )


class TestCallNodeMapping:
    """Tests for call node mapping to [F, K_max]."""

    def test_nodes_in_range(self):
        """Call nodes must lie in [F, K_max]."""
        xi, _ = gauss_legendre_nodes_weights(5)
        F, K_max = 6120.0, 8000.0
        K_call = map_call_nodes(xi, F, K_max)
        assert np.all(K_call >= F - 1e-6)
        assert np.all(K_call <= K_max + 1e-6)

    def test_synthetic_paper_strikes_n4(self):
        """Verify 4-point call nodes match paper's reported values for F=100, K_max=200."""
        xi, _ = gauss_legendre_nodes_weights(4)
        F, K_max = 100.0, 200.0
        K_call = np.sort(map_call_nodes(xi, F, K_max))
        paper_calls = np.sort([103.6, 119.8, 150.4, 187.0])
        assert np.all(np.abs(K_call - paper_calls) < 1.5), (
            f"Call nodes {K_call} don't match paper {paper_calls}"
        )

    def test_reciprocal_mapping_consistency(self):
        """Verify that x_j = 1/K_j lies in [1/K_max, 1/F]."""
        xi, _ = gauss_legendre_nodes_weights(5)
        F, K_max = 100.0, 200.0
        K_call = map_call_nodes(xi, F, K_max)
        x_j = 1.0 / K_call
        assert np.all(x_j >= 1.0 / K_max - 1e-10)
        assert np.all(x_j <= 1.0 / F + 1e-10)


class TestGaussLegendreFairStrike:
    """Tests for GL variance strike computation."""

    def test_flat_vol_surface_recovers_sigma(self):
        """For flat vol surface, GL fair strike should ≈ sigma."""
        F, T, sigma = 100.0, 1.0, 0.20
        K_min, K_max = 10.0, 200.0

        def put_pricer(K):
            return black76_price(F, K, T, sigma, "put")

        def call_pricer(K):
            return black76_price(F, K, T, sigma, "call")

        K_v, _, _, _, _, _ = gauss_legendre_variance_strike(
            put_pricer, call_pricer, F, T, K_min, K_max, n_put=5, n_call=5
        )
        # With truncation [10, 200], should be close to sigma
        assert abs(K_v - sigma) < 0.01, f"K_v={K_v:.4f} not close to sigma={sigma}"

    def test_fair_strike_positive(self):
        """Fair variance strike must be positive."""
        F, T, sigma = 100.0, 1.0, 0.20
        K_min, K_max = 50.0, 200.0

        K_v, _, _, _, _, _ = gauss_legendre_variance_strike(
            lambda K: black76_price(F, K, T, sigma, "put"),
            lambda K: black76_price(F, K, T, sigma, "call"),
            F, T, K_min, K_max,
        )
        assert K_v > 0

    def test_higher_vol_gives_higher_fair_strike(self):
        """Higher vol surface should give higher fair variance strike."""
        F, T = 100.0, 1.0
        K_min, K_max = 50.0, 200.0

        K_v_low, _, _, _, _, _ = gauss_legendre_variance_strike(
            lambda K: black76_price(F, K, T, 0.15, "put"),
            lambda K: black76_price(F, K, T, 0.15, "call"),
            F, T, K_min, K_max,
        )
        K_v_high, _, _, _, _, _ = gauss_legendre_variance_strike(
            lambda K: black76_price(F, K, T, 0.30, "put"),
            lambda K: black76_price(F, K, T, 0.30, "call"),
            F, T, K_min, K_max,
        )
        assert K_v_high > K_v_low

    def test_n5_put_nodes_in_range(self):
        """5-point put nodes must lie in [K_min, F]."""
        F, T, sigma = 100.0, 1.0, 0.20
        K_min, K_max = 50.0, 200.0

        _, _, K_put, _, _, _ = gauss_legendre_variance_strike(
            lambda K: black76_price(F, K, T, sigma, "put"),
            lambda K: black76_price(F, K, T, sigma, "call"),
            F, T, K_min, K_max, n_put=5, n_call=5,
        )
        assert np.all(K_put >= K_min - 1e-10)
        assert np.all(K_put <= F + 1e-10)

    def test_n5_call_nodes_in_range(self):
        """5-point call nodes must lie in [F, K_max]."""
        F, T, sigma = 100.0, 1.0, 0.20
        K_min, K_max = 50.0, 200.0

        _, _, _, K_call, _, _ = gauss_legendre_variance_strike(
            lambda K: black76_price(F, K, T, sigma, "put"),
            lambda K: black76_price(F, K, T, sigma, "call"),
            F, T, K_min, K_max, n_put=5, n_call=5,
        )
        assert np.all(K_call >= F - 1e-6)
        assert np.all(K_call <= K_max + 1e-6)


class TestHedgeQuantities:
    """Tests for hedge quantity computation."""

    def test_put_quantities_positive(self):
        """Put hedge quantities should be positive (long puts for long var swap)."""
        xi, w = gauss_legendre_nodes_weights(5)
        F, K_min, K_max, T = 100.0, 50.0, 200.0, 1.0
        K_put = map_put_nodes(xi, F, K_min)
        K_call = map_call_nodes(xi, F, K_max)
        q_put, q_call = compute_hedge_quantities(
            K_put, K_call, w, w, F, T, K_min, K_max, K_v=0.20
        )
        assert np.all(q_put > 0)

    def test_call_quantities_positive(self):
        """Call hedge quantities should be positive."""
        xi, w = gauss_legendre_nodes_weights(5)
        F, K_min, K_max, T = 100.0, 50.0, 200.0, 1.0
        K_put = map_put_nodes(xi, F, K_min)
        K_call = map_call_nodes(xi, F, K_max)
        q_put, q_call = compute_hedge_quantities(
            K_put, K_call, w, w, F, T, K_min, K_max, K_v=0.20
        )
        assert np.all(q_call > 0)

    def test_quantities_scale_with_notional(self):
        """Hedge quantities should scale linearly with vega notional."""
        xi, w = gauss_legendre_nodes_weights(5)
        F, K_min, K_max, T = 100.0, 50.0, 200.0, 1.0
        K_put = map_put_nodes(xi, F, K_min)
        K_call = map_call_nodes(xi, F, K_max)

        q_put_1, q_call_1 = compute_hedge_quantities(
            K_put, K_call, w, w, F, T, K_min, K_max, K_v=0.20, vega_notional=1.0
        )
        q_put_2, q_call_2 = compute_hedge_quantities(
            K_put, K_call, w, w, F, T, K_min, K_max, K_v=0.20, vega_notional=2.0
        )
        assert np.allclose(q_put_2, 2.0 * q_put_1)
        assert np.allclose(q_call_2, 2.0 * q_call_1)
