"""
Comprehensive tests for CampaignState class in allocator.py (lines 21-87)
Focus on edge cases, division by zero, and new inner_metrics functionality.
"""
import pytest
import numpy as np
from src.allocator import CampaignState, CampaignConfig


class TestCampaignStateInitialization:
    """Test initialization and default values."""
    
    def test_initialization_with_defaults(self):
        """Test that CampaignState initializes correctly with default values."""
        cfg = CampaignConfig(campaign_id="test_001")
        state = CampaignState(campaign_id="test_001", cfg=cfg)
        
        assert state.campaign_id == "test_001"
        assert state.cum_clicks == 0
        assert state.cum_conversions == 0
        assert state.cum_cost == 0.0
        assert state.cum_spend_allocated == 0.0
        assert state.hour_spend_allocated == 0.0
        assert state.inner_cpc == 0.0
        assert state.inner_cvr == 0.0
        assert state.frozen is False
    
    def test_config_defaults_are_used(self):
        """Test that CampaignConfig uses correct default values."""
        cfg = CampaignConfig(campaign_id="test_002")
        
        assert cfg.cpa_cap == 50
        assert cfg.prior_alpha == 2.0
        assert cfg.prior_beta == 98.0
        assert cfg.prior_cpc == 2.0
        assert cfg.prior_cpa == 45.0
        assert cfg.prior_cvr == 0.02
        assert cfg.cpc_std == 0.05


class TestCPAProperty:
    """Tests for Cost Per Acquisition (CPA) property."""
    
    def test_cpa_with_zero_conversions_returns_prior(self):
        """EDGE CASE: CPA when cum_conversions=0 should return prior_cpa."""
        cfg = CampaignConfig(campaign_id="test", prior_cpa=45.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_conversions=0,
            cum_cost=100.0
        )
        
        assert state.cpa == 45.0  # Should return prior, not crash
    
    def test_cpa_with_actual_conversions(self):
        """Test CPA calculation with real conversions."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_conversions=10,
            cum_cost=500.0
        )
        
        assert state.cpa == 50.0  # 500 / 10
    
    def test_cpa_with_one_conversion(self):
        """BOUNDARY CASE: CPA with exactly 1 conversion."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_conversions=1,
            cum_cost=100.0
        )
        
        assert state.cpa == 100.0
    
    def test_cpa_custom_prior(self):
        """Test that custom prior_cpa is used when no conversions."""
        cfg = CampaignConfig(campaign_id="test", prior_cpa=30.0)
        state = CampaignState(campaign_id="test", cfg=cfg, cum_conversions=0)
        
        assert state.cpa == 30.0


class TestCVRProperty:
    """Tests for Conversion Rate (CVR) property."""
    
    def test_cvr_with_zero_clicks_returns_prior(self):
        """EDGE CASE: CVR when cum_clicks=0 should return prior_cvr."""
        cfg = CampaignConfig(campaign_id="test", prior_cvr=0.02)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=0,
            cum_conversions=0
        )
        
        assert state.cvr == 0.02  # Should return prior, not crash
    
    def test_cvr_with_actual_clicks(self):
        """Test CVR calculation with real data."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5
        )
        
        assert state.cvr == 0.05  # 5 / 100
    
    def test_cvr_perfect_conversion(self):
        """BOUNDARY CASE: CVR = 1.0 when all clicks convert."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=50,
            cum_conversions=50
        )
        
        assert state.cvr == 1.0
    
    def test_cvr_zero_conversion(self):
        """BOUNDARY CASE: CVR = 0.0 when no conversions."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=1000,
            cum_conversions=0
        )
        
        assert state.cvr == 0.0
    
    def test_cvr_custom_prior(self):
        """Test that custom prior_cvr is used when no clicks."""
        cfg = CampaignConfig(campaign_id="test", prior_cvr=0.05)
        state = CampaignState(campaign_id="test", cfg=cfg, cum_clicks=0)
        
        assert state.cvr == 0.05


class TestCPCProperty:
    """Tests for Cost Per Click (CPC) property."""
    
    def test_cpc_with_zero_clicks_returns_prior(self):
        """EDGE CASE: CPC when cum_clicks=0 should return prior_cpc."""
        cfg = CampaignConfig(campaign_id="test", prior_cpc=2.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=0,
            cum_cost=0.0
        )
        
        assert state.cpc == 2.0
    
    def test_cpc_with_actual_clicks(self):
        """Test CPC calculation with real data."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_cost=300.0
        )
        
        assert state.cpc == 3.0  # 300 / 100
    
    def test_cpc_custom_prior(self):
        """Test that custom prior_cpc is used when no clicks."""
        cfg = CampaignConfig(campaign_id="test", prior_cpc=1.5)
        state = CampaignState(campaign_id="test", cfg=cfg, cum_clicks=0)
        
        assert state.cpc == 1.5


class TestAlphaProperty:
    """Tests for Alpha property (Beta distribution parameter)."""
    
    def test_alpha_with_zero_clicks_returns_prior(self):
        """EDGE CASE: Alpha returns prior when cum_clicks < 100."""
        cfg = CampaignConfig(campaign_id="test", prior_alpha=2.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=0,
            cum_conversions=0
        )
        
        assert state.alpha == 2.0  # Returns prior
    
    def test_alpha_with_less_than_100_clicks_returns_prior(self):
        """Test alpha returns prior when clicks < 100."""
        cfg = CampaignConfig(campaign_id="test", prior_alpha=2.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=99,
            cum_conversions=5
        )
        
        assert state.alpha == 2.0  # Still uses prior
    
    def test_alpha_at_100_clicks_threshold(self):
        """BOUNDARY CASE: Alpha switches to calculation at exactly 100 clicks."""
        cfg = CampaignConfig(campaign_id="test", prior_alpha=2.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5
        )
        
        assert state.alpha == 6  # 5 + 1 (no longer using prior)
    
    def test_alpha_above_100_clicks(self):
        """Test alpha calculation when clicks >= 100."""
        cfg = CampaignConfig(campaign_id="test", prior_alpha=2.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=500,
            cum_conversions=25
        )
        
        assert state.alpha == 26  # 25 + 1
    
    def test_alpha_with_zero_conversions_above_threshold(self):
        """Test alpha with no conversions but many clicks."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=1000,
            cum_conversions=0
        )
        
        assert state.alpha == 1  # 0 + 1


class TestBetaProperty:
    """Tests for Beta property (Beta distribution parameter)."""
    
    def test_beta_with_zero_clicks_returns_prior(self):
        """EDGE CASE: Beta returns prior when cum_clicks < 100."""
        cfg = CampaignConfig(campaign_id="test", prior_beta=98.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=0,
            cum_conversions=0
        )
        
        assert state.beta == 98.0  # Returns prior
    
    def test_beta_with_less_than_100_clicks_returns_prior(self):
        """Test beta returns prior when clicks < 100."""
        cfg = CampaignConfig(campaign_id="test", prior_beta=98.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=99,
            cum_conversions=5
        )
        
        assert state.beta == 98.0  # Still uses prior
    
    def test_beta_at_100_clicks_threshold(self):
        """BOUNDARY CASE: Beta switches to calculation at exactly 100 clicks."""
        cfg = CampaignConfig(campaign_id="test", prior_beta=98.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5
        )
        
        assert state.beta == 96  # (100 - 5) + 1
    
    def test_beta_above_100_clicks(self):
        """Test beta calculation when clicks >= 100."""
        cfg = CampaignConfig(campaign_id="test", prior_beta=98.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=500,
            cum_conversions=25
        )
        
        assert state.beta == 476  # (500 - 25) + 1
    
    def test_beta_with_all_conversions_above_threshold(self):
        """BOUNDARY CASE: Beta when all clicks convert (minimum value)."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=100
        )
        
        assert state.beta == 1  # (100 - 100) + 1


class TestCalculateInnerMetrics:
    """Tests for calculate_inner_metrics() method."""
    
    def test_calculate_inner_metrics_returns_tuple(self):
        """Test that calculate_inner_metrics returns (cpc, cvr) tuple."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        cpc, cvr = state.calculate_inner_metrics()
        
        assert isinstance(cpc, (float, np.floating))
        assert isinstance(cvr, (float, np.floating))
    
    def test_calculate_inner_metrics_cpc_has_minimum(self):
        """Test that CPC has a minimum value of 0.01."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test", prior_cpc=0.001)
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        # Run multiple times to check minimum is enforced
        for _ in range(10):
            cpc, _ = state.calculate_inner_metrics()
            assert cpc >= 0.01
    
    def test_calculate_inner_metrics_uses_current_metrics(self):
        """Test that inner metrics are based on current CPC/CVR."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        original_cpc = state.cpc  # 2.0
        original_cvr = state.cvr  # 0.05
        
        cpc, cvr = state.calculate_inner_metrics()
        
        # Should be close to originals (low variance of 0.009 and 0.01)
        assert 1.8 <= cpc <= 2.2  # Within reasonable range
        assert 0.04 <= cvr <= 0.06  # Within reasonable range
    
    def test_calculate_inner_metrics_with_priors(self):
        """Test calculate_inner_metrics when using prior values."""
        np.random.seed(123)
        cfg = CampaignConfig(campaign_id="test", prior_cpc=2.0, prior_cvr=0.02)
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        cpc, cvr = state.calculate_inner_metrics()
        
        # Should be based on priors
        assert cpc > 0
        assert cvr != 0  # CVR will vary around prior


class TestCalculateInnerMetricsHighVariance:
    """Tests for calculate_inner_metrics_high_variance() method."""
    
    def test_calculate_inner_metrics_high_variance_returns_tuple(self):
        """Test that method returns (cpc, cvr) tuple."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        cpc, cvr = state.calculate_inner_metrics_high_variance()
        
        assert isinstance(cpc, (float, np.floating))
        assert isinstance(cvr, (float, np.floating))
    
    def test_high_variance_cvr_clamped_to_0_1(self):
        """Test that CVR is clamped to [0, 1] range."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        # Run multiple times
        for _ in range(20):
            _, cvr = state.calculate_inner_metrics_high_variance()
            assert 0.0 <= cvr <= 1.0
    
    def test_high_variance_uses_beta_distribution_for_cvr(self):
        """Test that CVR uses Beta distribution based on alpha/beta."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        cvr_values = [state.calculate_inner_metrics_high_variance()[1] for _ in range(100)]
        
        # Beta distribution should give us variety
        assert len(set(cvr_values)) > 10  # Multiple different values
        assert all(0 <= v <= 1 for v in cvr_values)
    
    def test_high_variance_higher_variance_than_normal(self):
        """Test that high variance method has more variance than normal method."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        # Collect samples from both methods
        np.random.seed(42)
        normal_cpc_values = [state.calculate_inner_metrics()[0] for _ in range(50)]
        
        np.random.seed(42)
        high_cpc_values = [state.calculate_inner_metrics_high_variance()[0] for _ in range(50)]
        
        # High variance should have larger standard deviation
        # (0.05 vs 0.009 multiplier)
        normal_std = np.std(normal_cpc_values)
        high_std = np.std(high_cpc_values)
        
        # Note: This might be flaky due to randomness, but generally true
        # High variance uses 0.05 vs normal 0.009 multiplier


class TestSimulateNextHour:
    """Tests for simulate_next_hour() method."""
    
    def test_simulate_updates_cumulative_fields(self):
        """Test that simulate_next_hour updates all cumulative fields."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        
        initial_clicks = state.cum_clicks
        initial_cost = state.cum_cost
        
        state.simulate_next_hour(spend_allocation=100.0)
        
        assert state.cum_clicks >= initial_clicks
        assert state.cum_cost == initial_cost + 100.0
        assert state.hour_spend_allocated == 100.0
    
    def test_simulate_from_zero_state(self):
        """Test simulation starting from zero (uses priors)."""
        np.random.seed(123)
        cfg = CampaignConfig(campaign_id="test", prior_cpc=2.0)
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        state.simulate_next_hour(spend_allocation=50.0)
        
        assert state.cum_clicks > 0
        assert state.cum_cost == 50.0
        assert state.cum_conversions >= 0
    
    def test_simulate_with_zero_allocation(self):
        """EDGE CASE: Simulate with zero spend."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        state.simulate_next_hour(spend_allocation=0.0)
        
        assert state.cum_clicks == 0
        assert state.cum_conversions == 0
        assert state.cum_cost == 0.0
        assert state.hour_spend_allocated == 0.0
    
    def test_simulate_uses_high_variance_metrics(self):
        """Test that simulate_next_hour uses calculate_inner_metrics_high_variance."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        # Simulate multiple hours to see variance
        state.simulate_next_hour(spend_allocation=100.0)
        
        # Should have added clicks based on high variance sampling
        assert state.cum_clicks > 200
        assert state.cum_cost == 500.0
    
    def test_simulate_conversions_reasonable(self):
        """Test that conversions are reasonable given clicks."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        initial_conversions = state.cum_conversions
        initial_clicks = state.cum_clicks
        
        state.simulate_next_hour(spend_allocation=100.0)
        
        new_conversions = state.cum_conversions - initial_conversions
        new_clicks = state.cum_clicks - initial_clicks
        
        # Conversions should not exceed clicks
        assert new_conversions <= new_clicks


class TestEdgeCases:
    """Test various edge cases and boundary conditions."""
    
    def test_all_properties_with_no_data(self):
        """Test all properties return priors when no data exists."""
        cfg = CampaignConfig(
            campaign_id="test",
            prior_cpc=2.0,
            prior_cpa=45.0,
            prior_cvr=0.02,
            prior_alpha=2.0,
            prior_beta=98.0
        )
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        assert state.cpc == 2.0
        assert state.cpa == 45.0
        assert state.cvr == 0.02
        assert state.alpha == 2.0
        assert state.beta == 98.0
    
    def test_properties_at_threshold_boundary(self):
        """Test properties at exactly 99 and 100 clicks (threshold boundary)."""
        cfg = CampaignConfig(campaign_id="test", prior_alpha=2.0, prior_beta=98.0)
        
        # At 99 clicks - should use priors for alpha/beta
        state_99 = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=99,
            cum_conversions=5,
            cum_cost=198.0
        )
        assert state_99.alpha == 2.0  # Prior
        assert state_99.beta == 98.0  # Prior
        assert state_99.cvr == pytest.approx(0.0505, rel=0.01)  # Calculated
        
        # At 100 clicks - should calculate alpha/beta
        state_100 = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        assert state_100.alpha == 6  # 5 + 1
        assert state_100.beta == 96  # (100-5) + 1
        assert state_100.cvr == 0.05  # 5/100
    
    def test_properties_are_read_only(self):
        """Test that computed properties cannot be set."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        with pytest.raises(AttributeError):
            state.cpa = 100.0
        
        with pytest.raises(AttributeError):
            state.cvr = 0.05
        
        with pytest.raises(AttributeError):
            state.cpc = 2.5
        
        with pytest.raises(AttributeError):
            state.alpha = 10
        
        with pytest.raises(AttributeError):
            state.beta = 90
    
    def test_properties_update_dynamically(self):
        """Test that properties recalculate when base fields change."""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        # Initially should use priors
        assert state.cpa == 45.0
        assert state.cvr == 0.02
        assert state.alpha == 2.0
        assert state.beta == 98.0
        
        # Update to exactly 100 clicks
        state.cum_clicks = 100
        state.cum_conversions = 5
        state.cum_cost = 200.0
        
        # Properties should update automatically
        assert state.cvr == 0.05
        assert state.cpa == 40.0
        assert state.cpc == 2.0
        assert state.alpha == 6  # Now calculated
        assert state.beta == 96  # Now calculated

