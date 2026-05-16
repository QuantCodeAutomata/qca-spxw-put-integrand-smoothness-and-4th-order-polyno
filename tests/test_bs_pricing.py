"""
Tests for src/bs_pricing.py

Validates Black-76 pricing, Greeks, and implied vol inversion.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bs_pricing import (
    black76_price,
    black76_delta,
    black76_gamma,
    black76_vega,
    implied_vol_bisection,
)


class TestBlack76Price:
    """Tests for Black-76 option pricing."""

    def test_call_put_parity_r0(self):
        """Under r=0, C(K) - P(K) = F - K (put-call parity)."""
        F, K, T, sigma = 100.0, 95.0, 1.0, 0.20
        call = black76_price(F, K, T, sigma, "call")
        put = black76_price(F, K, T, sigma, "put")
        assert abs((call - put) - (F - K)) < 1e-10, (
            f"Parity violated: C-P={call-put:.6f}, F-K={F-K:.6f}"
        )

    def test_call_put_parity_atm(self):
        """ATM put-call parity: C(F) - P(F) = 0."""
        F, K, T, sigma = 100.0, 100.0, 0.5, 0.25
        call = black76_price(F, K, T, sigma, "call")
        put = black76_price(F, K, T, sigma, "put")
        assert abs(call - put) < 1e-10

    def test_call_price_positive(self):
        """Call price must be non-negative."""
        for K in [80, 100, 120]:
            price = black76_price(100.0, K, 1.0, 0.20, "call")
            assert price >= 0, f"Negative call price at K={K}"

    def test_put_price_positive(self):
        """Put price must be non-negative."""
        for K in [80, 100, 120]:
            price = black76_price(100.0, K, 1.0, 0.20, "put")
            assert price >= 0, f"Negative put price at K={K}"

    def test_call_lower_bound(self):
        """Call price >= max(F - K, 0)."""
        F, K, T, sigma = 100.0, 90.0, 1.0, 0.20
        call = black76_price(F, K, T, sigma, "call")
        assert call >= max(F - K, 0) - 1e-10

    def test_put_lower_bound(self):
        """Put price >= max(K - F, 0)."""
        F, K, T, sigma = 100.0, 110.0, 1.0, 0.20
        put = black76_price(F, K, T, sigma, "put")
        assert put >= max(K - F, 0) - 1e-10

    def test_zero_vol_call_intrinsic(self):
        """Zero vol: call = max(F-K, 0)."""
        F, K = 100.0, 90.0
        call = black76_price(F, K, 1.0, 0.0, "call")
        assert abs(call - max(F - K, 0)) < 1e-10

    def test_zero_vol_put_intrinsic(self):
        """Zero vol: put = max(K-F, 0)."""
        F, K = 100.0, 110.0
        put = black76_price(F, K, 1.0, 0.0, "put")
        assert abs(put - max(K - F, 0)) < 1e-10

    def test_zero_time_call_intrinsic(self):
        """Zero time: call = max(F-K, 0)."""
        F, K = 100.0, 90.0
        call = black76_price(F, K, 0.0, 0.20, "call")
        assert abs(call - max(F - K, 0)) < 1e-10

    def test_call_monotone_in_vol(self):
        """Call price is increasing in volatility."""
        F, K, T = 100.0, 100.0, 1.0
        prices = [black76_price(F, K, T, sigma, "call") for sigma in [0.10, 0.20, 0.30]]
        assert prices[0] < prices[1] < prices[2]

    def test_known_atm_call_value(self):
        """ATM call ≈ F * sigma * sqrt(T) / sqrt(2*pi) for small sigma."""
        F, K, T, sigma = 100.0, 100.0, 1.0, 0.20
        call = black76_price(F, K, T, sigma, "call")
        # Approximate: ATM call ≈ F * sigma * sqrt(T) * 0.3989
        approx = F * sigma * np.sqrt(T) * 0.3989
        assert abs(call - approx) / approx < 0.05  # within 5%

    def test_deep_otm_put_near_zero(self):
        """Deep OTM put should be near zero."""
        put = black76_price(100.0, 10.0, 1.0, 0.20, "put")
        assert put < 0.01

    def test_deep_itm_call_near_intrinsic(self):
        """Deep ITM call should be near F - K."""
        F, K = 100.0, 10.0
        call = black76_price(F, K, 1.0, 0.20, "call")
        assert abs(call - (F - K)) < 1.0  # within $1


class TestBlack76Greeks:
    """Tests for Black-76 Greeks."""

    def test_call_delta_between_0_and_1(self):
        """Call delta must be in (0, 1)."""
        delta = black76_delta(100.0, 100.0, 1.0, 0.20, "call")
        assert 0 < delta < 1

    def test_put_delta_between_minus1_and_0(self):
        """Put delta must be in (-1, 0)."""
        delta = black76_delta(100.0, 100.0, 1.0, 0.20, "put")
        assert -1 < delta < 0

    def test_call_put_delta_sum(self):
        """Call delta - put delta = 1 (from put-call parity dC/dF - dP/dF = 1)."""
        F, K, T, sigma = 100.0, 95.0, 1.0, 0.20
        call_delta = black76_delta(F, K, T, sigma, "call")
        put_delta = black76_delta(F, K, T, sigma, "put")
        # C - P = F - K => dC/dF - dP/dF = 1
        assert abs(call_delta - put_delta - 1.0) < 1e-10

    def test_gamma_positive(self):
        """Gamma must be positive."""
        gamma = black76_gamma(100.0, 100.0, 1.0, 0.20)
        assert gamma > 0

    def test_vega_positive(self):
        """Vega must be positive."""
        vega = black76_vega(100.0, 100.0, 1.0, 0.20)
        assert vega > 0

    def test_gamma_symmetric_around_atm(self):
        """Gamma is approximately symmetric around ATM for small skew."""
        F, T, sigma = 100.0, 1.0, 0.20
        gamma_itm = black76_gamma(F, 95.0, T, sigma)
        gamma_otm = black76_gamma(F, 105.0, T, sigma)
        # Not exactly equal but should be close
        assert abs(gamma_itm - gamma_otm) / gamma_itm < 0.15

    def test_numerical_delta_consistency(self):
        """Numerical delta should match analytical delta."""
        F, K, T, sigma = 100.0, 100.0, 1.0, 0.20
        h = 0.01
        call_up = black76_price(F + h, K, T, sigma, "call")
        call_dn = black76_price(F - h, K, T, sigma, "call")
        num_delta = (call_up - call_dn) / (2 * h)
        ana_delta = black76_delta(F, K, T, sigma, "call")
        assert abs(num_delta - ana_delta) < 1e-4

    def test_numerical_gamma_consistency(self):
        """Numerical gamma should match analytical gamma."""
        F, K, T, sigma = 100.0, 100.0, 1.0, 0.20
        h = 0.1
        call_up = black76_price(F + h, K, T, sigma, "call")
        call_mid = black76_price(F, K, T, sigma, "call")
        call_dn = black76_price(F - h, K, T, sigma, "call")
        num_gamma = (call_up - 2 * call_mid + call_dn) / h**2
        ana_gamma = black76_gamma(F, K, T, sigma)
        assert abs(num_gamma - ana_gamma) / ana_gamma < 0.01

    def test_numerical_vega_consistency(self):
        """Numerical vega should match analytical vega."""
        F, K, T, sigma = 100.0, 100.0, 1.0, 0.20
        h = 0.001
        call_up = black76_price(F, K, T, sigma + h, "call")
        call_dn = black76_price(F, K, T, sigma - h, "call")
        num_vega = (call_up - call_dn) / (2 * h)
        ana_vega = black76_vega(F, K, T, sigma)
        assert abs(num_vega - ana_vega) / ana_vega < 0.01


class TestImpliedVol:
    """Tests for implied volatility inversion."""

    def test_round_trip_call(self):
        """Implied vol should recover the input vol for a call."""
        F, K, T, sigma = 100.0, 100.0, 1.0, 0.20
        price = black76_price(F, K, T, sigma, "call")
        iv = implied_vol_bisection(F, K, T, price, "call")
        assert abs(iv - sigma) < 1e-6

    def test_round_trip_put(self):
        """Implied vol should recover the input vol for a put."""
        F, K, T, sigma = 100.0, 95.0, 1.0, 0.25
        price = black76_price(F, K, T, sigma, "put")
        iv = implied_vol_bisection(F, K, T, price, "put")
        assert abs(iv - sigma) < 1e-6

    def test_round_trip_otm_put(self):
        """Implied vol round-trip for deep OTM put."""
        F, K, T, sigma = 100.0, 70.0, 1.0, 0.30
        price = black76_price(F, K, T, sigma, "put")
        iv = implied_vol_bisection(F, K, T, price, "put")
        assert abs(iv - sigma) < 1e-5

    def test_round_trip_various_vols(self):
        """Round-trip test for various volatility levels."""
        F, K, T = 100.0, 100.0, 0.5
        for sigma in [0.05, 0.10, 0.20, 0.40, 0.80]:
            price = black76_price(F, K, T, sigma, "call")
            iv = implied_vol_bisection(F, K, T, price, "call")
            assert abs(iv - sigma) < 1e-5, f"Round-trip failed for sigma={sigma}"

    def test_below_intrinsic_returns_nan(self):
        """Price below intrinsic should return nan."""
        F, K = 100.0, 90.0
        intrinsic = F - K
        iv = implied_vol_bisection(F, K, 1.0, intrinsic - 1.0, "call")
        assert np.isnan(iv)
