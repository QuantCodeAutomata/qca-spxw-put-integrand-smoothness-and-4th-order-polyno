"""
Tests for src/polynomial_fit.py

Validates polynomial fitting and R² computation.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polynomial_fit import (
    fit_polynomial,
    compute_r_squared,
    compute_fit_statistics,
    fit_multiple_degrees,
)


class TestFitPolynomial:
    """Tests for polynomial fitting."""

    def test_perfect_fit_linear(self):
        """Linear fit on linear data should give R²=1."""
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 3.0
        coeffs, fitted = fit_polynomial(x, y, 1)
        assert np.allclose(fitted, y, atol=1e-8)

    def test_perfect_fit_quadratic(self):
        """Quadratic fit on quadratic data should give R²=1."""
        x = np.linspace(-5, 5, 100)
        y = 3.0 * x**2 - 2.0 * x + 1.0
        coeffs, fitted = fit_polynomial(x, y, 2)
        assert np.allclose(fitted, y, atol=1e-6)

    def test_perfect_fit_4th_order(self):
        """4th-order fit on 4th-order data should give R²=1."""
        x = np.linspace(0, 100, 200)
        y = 1e-8 * x**4 - 1e-5 * x**3 + 0.001 * x**2 - 0.1 * x + 5.0
        coeffs, fitted = fit_polynomial(x, y, 4)
        r2 = compute_r_squared(y, fitted)
        assert r2 > 0.9999

    def test_coefficients_shape(self):
        """Coefficients should have length degree+1."""
        x = np.linspace(0, 10, 50)
        y = np.random.randn(50)
        for deg in [1, 2, 3, 4, 5]:
            coeffs, _ = fit_polynomial(x, y, deg)
            assert len(coeffs) == deg + 1

    def test_higher_degree_better_fit(self):
        """Higher degree polynomial should fit noisy data at least as well."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)
        _, fitted_2 = fit_polynomial(x, y, 2)
        _, fitted_4 = fit_polynomial(x, y, 4)
        r2_2 = compute_r_squared(y, fitted_2)
        r2_4 = compute_r_squared(y, fitted_4)
        assert r2_4 >= r2_2


class TestRSquared:
    """Tests for R² computation."""

    def test_perfect_fit_r2_equals_1(self):
        """Perfect fit should give R²=1."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r2 = compute_r_squared(y, y)
        assert abs(r2 - 1.0) < 1e-12

    def test_mean_prediction_r2_equals_0(self):
        """Predicting the mean gives R²=0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_hat = np.full_like(y, np.mean(y))
        r2 = compute_r_squared(y, y_hat)
        assert abs(r2) < 1e-12

    def test_r2_between_0_and_1_for_good_fit(self):
        """R² should be in [0, 1] for a reasonable fit."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + 0.5 * np.random.randn(100)
        _, fitted = fit_polynomial(x, y, 1)
        r2 = compute_r_squared(y, fitted)
        assert 0 <= r2 <= 1

    def test_r2_constant_data(self):
        """Constant data: R²=1 if fit is perfect, 1 if SST=0."""
        y = np.ones(10)
        r2 = compute_r_squared(y, y)
        assert r2 == 1.0

    def test_r2_increases_with_degree(self):
        """R² should not decrease as polynomial degree increases."""
        np.random.seed(0)
        x = np.linspace(0, 10, 50)
        y = x**3 - 5 * x**2 + 3 * x + np.random.randn(50) * 0.1
        r2_prev = -np.inf
        for deg in [1, 2, 3, 4]:
            _, fitted = fit_polynomial(x, y, deg)
            r2 = compute_r_squared(y, fitted)
            assert r2 >= r2_prev - 1e-10
            r2_prev = r2


class TestFitStatistics:
    """Tests for comprehensive fit statistics."""

    def test_statistics_keys(self):
        """Statistics dict should contain expected keys."""
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + np.random.randn(50) * 0.1
        _, fitted = fit_polynomial(x, y, 1)
        stats = compute_fit_statistics(x, y, fitted)
        expected_keys = {"r_squared", "sse", "sst", "rmse", "mae", "max_abs_residual", "n_points"}
        assert expected_keys.issubset(stats.keys())

    def test_sse_non_negative(self):
        """SSE must be non-negative."""
        x = np.linspace(0, 10, 50)
        y = np.random.randn(50)
        _, fitted = fit_polynomial(x, y, 2)
        stats = compute_fit_statistics(x, y, fitted)
        assert stats["sse"] >= 0

    def test_rmse_non_negative(self):
        """RMSE must be non-negative."""
        x = np.linspace(0, 10, 50)
        y = np.random.randn(50)
        _, fitted = fit_polynomial(x, y, 2)
        stats = compute_fit_statistics(x, y, fitted)
        assert stats["rmse"] >= 0

    def test_n_points_correct(self):
        """n_points should match input length."""
        x = np.linspace(0, 10, 73)
        y = np.random.randn(73)
        _, fitted = fit_polynomial(x, y, 2)
        stats = compute_fit_statistics(x, y, fitted)
        assert stats["n_points"] == 73


class TestFitMultipleDegrees:
    """Tests for multi-degree fitting."""

    def test_returns_all_degrees(self):
        """Should return results for all requested degrees."""
        x = np.linspace(0, 10, 100)
        y = np.random.randn(100)
        results = fit_multiple_degrees(x, y, [2, 3, 4, 5])
        assert set(results.keys()) == {2, 3, 4, 5}

    def test_each_result_has_required_keys(self):
        """Each degree result should have coeffs, fitted, stats."""
        x = np.linspace(0, 10, 100)
        y = np.random.randn(100)
        results = fit_multiple_degrees(x, y, [2, 3])
        for deg, res in results.items():
            assert "coeffs" in res
            assert "fitted" in res
            assert "stats" in res

    def test_4th_order_fit_smooth_integrand(self):
        """
        4th-order fit on a smooth integrand-like function should give high R².

        This is the core methodology test: the paper claims R² ≈ 0.9992 for
        the put-side variance-swap integrand fitted with a 4th-order polynomial.
        """
        # Simulate a smooth integrand: put(K)/K^2 for a Black-Scholes surface
        from src.bs_pricing import black76_price
        F, T, sigma = 6120.0, 42.0 / 365.0, 0.135
        K_arr = np.linspace(0.75 * F, 0.999 * F, 80)
        integrand = np.array([
            black76_price(F, K, T, max(sigma - 0.12 * (K / F - 1), 0.01), "put") / K**2
            for K in K_arr
        ])

        results = fit_multiple_degrees(K_arr, integrand, [2, 3, 4, 5])
        r2_4 = results[4]["stats"]["r_squared"]

        # The integrand should be very smooth; 4th-order fit should give high R²
        assert r2_4 > 0.99, f"4th-order R²={r2_4:.4f} should be > 0.99 for smooth integrand"
