"""
Tests for src/variance_swap.py

Validates variance swap integrand and fair strike computation.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.variance_swap import (
    compute_put_integrand,
    compute_call_integrand,
    benchmark_variance_strike,
    variance_swap_mtm,
)
from src.bs_pricing import black76_price


class TestIntegrand:
    """Tests for variance-swap integrand computation."""

    def test_put_integrand_positive(self):
        """Put integrand must be non-negative."""
        K = np.array([80.0, 90.0, 95.0, 99.0])
        P = np.array([5.0, 3.0, 1.5, 0.5])
        g = compute_put_integrand(K, P)
        assert np.all(g >= 0)

    def test_call_integrand_positive(self):
        """Call integrand must be non-negative."""
        K = np.array([101.0, 110.0, 120.0])
        C = np.array([3.0, 1.0, 0.2])
        h = compute_call_integrand(K, C)
        assert np.all(h >= 0)

    def test_put_integrand_formula(self):
        """Put integrand = put_price / K^2."""
        K = np.array([90.0, 95.0])
        P = np.array([4.0, 2.0])
        g = compute_put_integrand(K, P)
        expected = P / K**2
        assert np.allclose(g, expected)

    def test_integrand_decreasing_for_otm_puts(self):
        """OTM put integrand should generally decrease as K decreases."""
        F, T, sigma = 100.0, 1.0, 0.20
        K_arr = np.linspace(50.0, 99.0, 50)
        P_arr = np.array([black76_price(F, K, T, sigma, "put") for K in K_arr])
        g = compute_put_integrand(K_arr, P_arr)
        # Integrand should peak near ATM and decrease for deep OTM
        # Check that the last few values (near ATM) are larger than first few (deep OTM)
        assert np.mean(g[-10:]) > np.mean(g[:10])


class TestBenchmarkVarianceStrike:
    """Tests for benchmark variance strike computation."""

    def _make_chain(self, F: float, T: float, sigma: float, n: int = 100) -> pd.DataFrame:
        """Build a synthetic option chain for testing."""
        K_min, K_max = 0.5 * F, 1.5 * F
        K_arr = np.linspace(K_min, K_max, n)
        records = []
        for K in K_arr:
            if K < F:
                price = black76_price(F, K, T, sigma, "put")
                opt_type = "put"
            else:
                price = black76_price(F, K, T, sigma, "call")
                opt_type = "call"
            records.append({"strike": K, "type": opt_type, "price": price})
        return pd.DataFrame(records)

    def test_flat_vol_recovers_sigma(self):
        """Benchmark fair strike should ≈ sigma for flat vol surface."""
        F, T, sigma = 100.0, 1.0, 0.20
        df = self._make_chain(F, T, sigma, n=200)
        K_v, _, _ = benchmark_variance_strike(df, F, T, K_min=50.0, K_max=150.0)
        # With truncation, should be close to sigma
        assert abs(K_v - sigma) < 0.02, f"K_v={K_v:.4f} not close to sigma={sigma}"

    def test_fair_strike_positive(self):
        """Fair variance strike must be positive."""
        F, T, sigma = 100.0, 1.0, 0.20
        df = self._make_chain(F, T, sigma)
        K_v, _, _ = benchmark_variance_strike(df, F, T)
        assert K_v > 0

    def test_higher_vol_gives_higher_fair_strike(self):
        """Higher vol should give higher fair variance strike."""
        F, T = 100.0, 1.0
        df_low = self._make_chain(F, T, 0.15, n=200)
        df_high = self._make_chain(F, T, 0.30, n=200)
        K_v_low, _, _ = benchmark_variance_strike(df_low, F, T, K_min=50.0, K_max=150.0)
        K_v_high, _, _ = benchmark_variance_strike(df_high, F, T, K_min=50.0, K_max=150.0)
        assert K_v_high > K_v_low

    def test_empty_chain_returns_zero(self):
        """Empty chain should return zero fair strike."""
        df = pd.DataFrame(columns=["strike", "type", "price"])
        K_v, _, _ = benchmark_variance_strike(df, 100.0, 1.0)
        assert K_v == 0.0

    def test_fair_strike_squared_non_negative(self):
        """Fair variance strike squared must be non-negative."""
        F, T, sigma = 100.0, 1.0, 0.20
        df = self._make_chain(F, T, sigma)
        _, K_v_sq, _ = benchmark_variance_strike(df, F, T)
        assert K_v_sq >= 0


class TestVarianceSwapMTM:
    """Tests for variance swap mark-to-market."""

    def test_mtm_positive_when_vol_rises(self):
        """Long var swap MTM should be positive when vol rises."""
        mtm = variance_swap_mtm(20.0, 30.0)
        assert mtm > 0

    def test_mtm_negative_when_vol_falls(self):
        """Long var swap MTM should be negative when vol falls."""
        mtm = variance_swap_mtm(20.0, 15.0)
        assert mtm < 0

    def test_mtm_zero_when_unchanged(self):
        """MTM should be zero when fair strike is unchanged."""
        mtm = variance_swap_mtm(20.0, 20.0)
        assert abs(mtm) < 1e-12

    def test_mtm_formula(self):
        """MTM = (K_new^2 - K_old^2) / (2 * K_old)."""
        K_old, K_new = 23.5, 33.2
        expected = (K_new**2 - K_old**2) / (2.0 * K_old)
        result = variance_swap_mtm(K_old, K_new)
        assert abs(result - expected) < 1e-10

    def test_paper_example_mtm(self):
        """Paper example: K_old=23.5, K_new=33.2 → MTM ≈ 11.7."""
        mtm = variance_swap_mtm(23.5, 33.2)
        # Paper reports ~11.7
        assert abs(mtm - 11.7) < 0.5, f"MTM={mtm:.2f} not close to paper's 11.7"
