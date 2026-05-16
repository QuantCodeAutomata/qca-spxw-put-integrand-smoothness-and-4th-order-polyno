"""
Tests for src/synthetic_surface.py

Validates the synthetic volatility surface construction for Exp4.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic_surface import base_vol, shocked_vol


class TestBaseVol:
    """Tests for flat base volatility surface."""

    def test_flat_vol_scalar(self):
        """Base vol should return sigma for any scalar K."""
        assert base_vol(100.0, 0.20) == 0.20
        assert base_vol(50.0, 0.15) == 0.15

    def test_flat_vol_array(self):
        """Base vol should return constant array."""
        K = np.array([50.0, 100.0, 150.0])
        v = base_vol(K, 0.20)
        assert np.all(v == 0.20)
        assert len(v) == 3

    def test_flat_vol_default(self):
        """Default sigma should be 0.20."""
        assert base_vol(100.0) == 0.20


class TestShockedVol:
    """Tests for post-shock volatility surface."""

    def test_atm_vol_unchanged(self):
        """At K=F, tanh(0)=0, so shocked vol = base vol."""
        F, T, sigma = 100.0, 1.0, 0.20
        v = shocked_vol(F, F, T, sigma)
        assert abs(v - sigma) < 1e-10

    def test_put_vol_increases_for_otm(self):
        """For K < F (OTM put), shocked vol should be > base vol (downside skew)."""
        F, T, sigma = 100.0, 1.0, 0.20
        K_otm = 80.0  # OTM put
        v_shocked = shocked_vol(K_otm, F, T, sigma)
        # delta_sigma_put=0.30 > 0, tanh(ln(K/F)) < 0 for K < F
        # so sigma_post = sigma - 0.30 * tanh(negative) = sigma + positive
        assert v_shocked > sigma

    def test_call_vol_decreases_for_otm(self):
        """For K > F (OTM call), shocked vol should be < base vol (call skew flattening)."""
        F, T, sigma = 100.0, 1.0, 0.20
        K_otm = 120.0  # OTM call
        v_shocked = shocked_vol(K_otm, F, T, sigma)
        # delta_sigma_call=-0.15, tanh(ln(K/F)) > 0 for K > F
        # so sigma_post = sigma + (-0.15) * tanh(positive) = sigma - positive
        assert v_shocked < sigma

    def test_all_vols_positive(self):
        """All shocked vols must remain positive."""
        F, T, sigma = 100.0, 1.0, 0.20
        K_arr = np.linspace(10.0, 200.0, 200)
        v = shocked_vol(K_arr, F, T, sigma)
        assert np.all(v > 0)

    def test_vol_array_shape(self):
        """Output shape should match input shape."""
        K = np.linspace(50.0, 150.0, 100)
        v = shocked_vol(K, 100.0, 1.0, 0.20)
        assert v.shape == K.shape

    def test_put_formula_correctness(self):
        """Verify put formula: sigma_post = sigma - delta_put * tanh(ln(K/F)/(m*sigma*sqrt(T)))."""
        F, T, sigma = 100.0, 1.0, 0.20
        K = 80.0
        delta_put = 0.30
        m_put = 3.0
        expected = sigma - delta_put * np.tanh(np.log(K / F) / (m_put * sigma * np.sqrt(T)))
        result = shocked_vol(K, F, T, sigma, delta_put, -0.15, m_put, 3.0)
        assert abs(result - expected) < 1e-12

    def test_call_formula_correctness(self):
        """Verify call formula: sigma_post = sigma + delta_call * tanh(ln(K/F)/(m*sigma*sqrt(T)))."""
        F, T, sigma = 100.0, 1.0, 0.20
        K = 120.0
        delta_call = -0.15
        m_call = 3.0
        expected = sigma + delta_call * np.tanh(np.log(K / F) / (m_call * sigma * np.sqrt(T)))
        result = shocked_vol(K, F, T, sigma, 0.30, delta_call, 3.0, m_call)
        assert abs(result - expected) < 1e-12

    def test_skew_monotone_for_puts(self):
        """Put vol should be monotonically decreasing as K increases toward F."""
        F, T, sigma = 100.0, 1.0, 0.20
        K_arr = np.linspace(20.0, 99.0, 50)
        v = shocked_vol(K_arr, F, T, sigma)
        # As K increases toward F, put vol should decrease (less OTM)
        assert np.all(np.diff(v) >= -0.01)  # allow small numerical noise

    def test_continuity_at_atm(self):
        """Vol should be continuous at K=F."""
        F, T, sigma = 100.0, 1.0, 0.20
        eps = 1e-6
        v_below = shocked_vol(F - eps, F, T, sigma)
        v_above = shocked_vol(F + eps, F, T, sigma)
        v_at = shocked_vol(F, F, T, sigma)
        assert abs(v_below - v_at) < 0.001
        assert abs(v_above - v_at) < 0.001


class TestExp4StrikeVerification:
    """Tests verifying Exp4 synthetic strike locations match paper."""

    def test_4point_put_nodes_match_paper(self):
        """4-point put nodes for F=100, K_min=10 should match paper's values."""
        from src.gauss_legendre import gauss_legendre_nodes_weights, map_put_nodes
        xi, _ = gauss_legendre_nodes_weights(4)
        K_put = np.sort(map_put_nodes(xi, 100.0, 10.0))
        paper_puts = np.sort([16.2, 39.7, 70.3, 93.8])
        diffs = np.abs(K_put - paper_puts)
        assert np.all(diffs < 1.5), f"Put nodes {K_put} differ from paper {paper_puts}"

    def test_4point_call_nodes_match_paper(self):
        """4-point call nodes for F=100, K_max=200 should match paper's values."""
        from src.gauss_legendre import gauss_legendre_nodes_weights, map_call_nodes
        xi, _ = gauss_legendre_nodes_weights(4)
        K_call = np.sort(map_call_nodes(xi, 100.0, 200.0))
        paper_calls = np.sort([103.6, 119.8, 150.4, 187.0])
        diffs = np.abs(K_call - paper_calls)
        assert np.all(diffs < 1.5), f"Call nodes {K_call} differ from paper {paper_calls}"
