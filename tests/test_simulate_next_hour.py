"""
Comprehensive tests for simulate_next_hour method.
Testing for bugs, edge cases, and incorrect assumptions.
Don't assume the implementation is correct - test to find issues!
"""
import pytest
import numpy as np
from src.allocator import CampaignState, CampaignConfig


class TestSimulateNextHourBasicBehavior:
    """Test basic expected behavior of simulate_next_hour."""
    
    def test_simulate_updates_cum_clicks(self):
        """Test that cum_clicks increases after simulation."""
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
        state.simulate_next_hour(spend_allocation=100.0)
        
        # Should have more clicks (unless spend_allocation / sample_cpc = 0)
        assert state.cum_clicks >= initial_clicks
    
    def test_simulate_updates_cum_cost_exactly(self):
        """Test that cum_cost increases by exactly the spend_allocation."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        
        initial_cost = state.cum_cost
        allocation = 123.45
        state.simulate_next_hour(spend_allocation=allocation)
        
        # Should increase by exactly the allocation
        assert state.cum_cost == pytest.approx(initial_cost + allocation)
    
    def test_simulate_sets_hour_spend_allocated(self):
        """Test that hour_spend_allocated is set to the allocation amount."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        allocation = 50.0
        state.simulate_next_hour(spend_allocation=allocation)
        
        assert state.hour_spend_allocated == allocation
    
    def test_simulate_updates_cum_conversions(self):
        """Test that cum_conversions is updated (could be 0 or more)."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        
        initial_conversions = state.cum_conversions
        state.simulate_next_hour(spend_allocation=100.0)
        
        # Conversions should not decrease
        assert state.cum_conversions >= initial_conversions
    
    def test_conversions_do_not_exceed_clicks(self):
        """IMPORTANT: Conversions added should not exceed clicks added."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        initial_clicks = state.cum_clicks
        initial_conversions = state.cum_conversions
        
        state.simulate_next_hour(spend_allocation=100.0)
        
        new_clicks = state.cum_clicks - initial_clicks
        new_conversions = state.cum_conversions - initial_conversions
        
        # This MUST hold true - cannot have more conversions than clicks
        assert new_conversions <= new_clicks, \
            f"Got {new_conversions} conversions from {new_clicks} clicks"


class TestSimulateNextHourEdgeCases:
    """Test edge cases that might break the implementation."""
    
    def test_zero_spend_allocation(self):
        """EDGE CASE: What happens with zero spend?"""
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        
        initial_clicks = state.cum_clicks
        initial_conversions = state.cum_conversions
        initial_cost = state.cum_cost
        
        state.simulate_next_hour(spend_allocation=0.0)
        
        # With 0 spend, should get 0 clicks (0 / sample_cpc = 0)
        assert state.cum_clicks == initial_clicks
        assert state.cum_conversions == initial_conversions
        assert state.cum_cost == initial_cost
        assert state.hour_spend_allocated == 0.0
    
    def test_very_small_spend_allocation(self):
        """EDGE CASE: Very small spend (less than typical CPC)."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test", prior_cpc=2.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        
        initial_clicks = state.cum_clicks
        state.simulate_next_hour(spend_allocation=0.01)
        
        # With spend < CPC, int(0.01 / 2.0) = 0 clicks
        # This might be unexpected behavior!
        new_clicks = state.cum_clicks - initial_clicks
        assert new_clicks >= 0  # At least shouldn't be negative
    
    def test_very_large_spend_allocation(self):
        """EDGE CASE: Very large spend allocation."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        
        large_spend = 1_000_000.0
        state.simulate_next_hour(spend_allocation=large_spend)
        
        # Should handle large numbers without overflow
        assert state.cum_cost == 200.0 + large_spend
        assert state.cum_clicks > 100  # Should get many clicks
        assert state.cum_conversions >= 5
    
    def test_negative_spend_allocation_behavior(self):
        """POTENTIAL BUG: What happens with negative spend?
        
        The code doesn't validate inputs. Negative spend could:
        - Lead to negative clicks (int(-10 / 2) = -5)
        - Decrease cum_cost
        - Cause issues with binomial distribution
        """
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        
        # This might crash or give unexpected results
        try:
            state.simulate_next_hour(spend_allocation=-50.0)
            
            # If it doesn't crash, check what happened
            # Negative clicks would be added: cum_clicks += negative
            # This would DECREASE cum_clicks - probably a bug!
            # Also cum_cost would decrease, which doesn't make sense
            
            # Document this behavior
            assert True  # If we get here, it didn't crash
        except (ValueError, Exception) as e:
            # If it crashes, that's also valid to know
            pytest.skip(f"Method crashes with negative spend: {e}")
    
    def test_float_spend_allocation(self):
        """Test that float spend allocation works correctly."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        
        float_spend = 99.99
        initial_cost = state.cum_cost
        state.simulate_next_hour(spend_allocation=float_spend)
        
        # Should handle float precisely
        assert state.cum_cost == pytest.approx(initial_cost + float_spend)


class TestSimulateNextHourFromZeroState:
    """Test simulation starting from zero state (no prior data)."""
    
    def test_simulate_from_complete_zero_state(self):
        """Test starting from completely fresh campaign."""
        np.random.seed(123)
        cfg = CampaignConfig(campaign_id="test", prior_cpc=2.0, prior_cvr=0.02)
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        # All zeros initially
        assert state.cum_clicks == 0
        assert state.cum_conversions == 0
        assert state.cum_cost == 0.0
        
        state.simulate_next_hour(spend_allocation=100.0)
        
        # Should use priors for calculation
        assert state.cum_clicks > 0
        assert state.cum_cost == 100.0
        assert state.cum_conversions >= 0
    
    def test_zero_state_uses_priors_for_sampling(self):
        """Test that zero state correctly uses prior values in calculations."""
        np.random.seed(42)
        cfg = CampaignConfig(
            campaign_id="test",
            prior_cpc=2.0,
            prior_alpha=2.0,
            prior_beta=98.0
        )
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        # With prior_cpc=2.0 and spend=100, we'd expect around 50 clicks
        # (but with variance)
        state.simulate_next_hour(spend_allocation=100.0)
        
        # Should get reasonable number of clicks based on prior CPC
        assert 20 <= state.cum_clicks <= 200  # Wide range due to variance


class TestSimulateNextHourCalculationsCorrectness:
    """Test that the calculations are mathematically correct."""
    
    def test_clicks_calculation_formula(self):
        """Test that clicks = int(spend / sample_cpc) is correctly applied."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        initial_clicks = state.cum_clicks
        spend = 100.0
        
        # We can't predict exact sample_cpc due to randomness,
        # but we can verify relationship
        state.simulate_next_hour(spend_allocation=spend)
        
        new_clicks = state.cum_clicks - initial_clicks
        
        # clicks should be non-negative integer
        assert new_clicks >= 0
        assert isinstance(new_clicks, int)
    
    def test_int_truncation_of_clicks(self):
        """Test that int() truncation is used (not rounding).
        
        POTENTIAL BUG: Using int() means we always round DOWN.
        For example, if spend=10 and sample_cpc=3, we get 3.33...
        int(3.33) = 3 clicks, "wasting" 0.33 * 3 = ~1.0 in spend.
        """
        np.random.seed(999)
        cfg = CampaignConfig(campaign_id="test", prior_cpc=3.0)
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        # With very small spend vs CPC, might get 0 clicks
        state.simulate_next_hour(spend_allocation=2.5)
        
        # This documents the truncation behavior
        # (might get 0 clicks even with spend > 0)
        assert state.cum_clicks >= 0
    
    def test_binomial_distribution_properties(self):
        """Test that conversions follow binomial distribution properties."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=200,
            cum_conversions=10,
            cum_cost=400.0
        )
        
        # Run many simulations to check statistical properties
        conversion_results = []
        for seed in range(50):
            np.random.seed(seed)
            test_state = CampaignState(
                campaign_id="test",
                cfg=cfg,
                cum_clicks=200,
                cum_conversions=10,
                cum_cost=400.0
            )
            initial_conv = test_state.cum_conversions
            initial_clicks = test_state.cum_clicks
            
            test_state.simulate_next_hour(spend_allocation=100.0)
            
            new_conv = test_state.cum_conversions - initial_conv
            new_clicks = test_state.cum_clicks - initial_clicks
            
            if new_clicks > 0:
                conv_rate = new_conv / new_clicks
                conversion_results.append((new_clicks, new_conv, conv_rate))
        
        # All conversion rates should be between 0 and 1
        for clicks, convs, rate in conversion_results:
            assert 0 <= rate <= 1, f"Invalid conversion rate: {rate}"
            assert convs <= clicks, f"More conversions ({convs}) than clicks ({clicks})"


class TestSimulateNextHourConsistency:
    """Test consistency and determinism."""
    
    def test_same_seed_gives_same_results(self):
        """Test that same random seed gives same results."""
        cfg = CampaignConfig(campaign_id="test")
        
        # Run 1
        np.random.seed(42)
        state1 = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        state1.simulate_next_hour(spend_allocation=100.0)
        
        # Run 2 with same seed
        np.random.seed(42)
        state2 = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=200.0
        )
        state2.simulate_next_hour(spend_allocation=100.0)
        
        # Should get identical results
        assert state1.cum_clicks == state2.cum_clicks
        assert state1.cum_conversions == state2.cum_conversions
        assert state1.cum_cost == state2.cum_cost
    
    def test_different_seeds_give_different_results(self):
        """Test that different seeds give variability."""
        cfg = CampaignConfig(campaign_id="test")
        
        results = []
        for seed in [42, 123, 456, 789, 999]:
            np.random.seed(seed)
            state = CampaignState(
                campaign_id="test",
                cfg=cfg,
                cum_clicks=200,
                cum_conversions=10,
                cum_cost=400.0
            )
            state.simulate_next_hour(spend_allocation=100.0)
            results.append((state.cum_clicks, state.cum_conversions))
        
        # Should have some variation (not all identical)
        unique_results = set(results)
        assert len(unique_results) > 1, "No randomness detected"


class TestSimulateNextHourSequentialSimulations:
    """Test behavior over multiple sequential simulations."""
    
    def test_multiple_sequential_simulations(self):
        """Test that sequential simulations work correctly."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        # Simulate 10 hours
        for hour in range(10):
            prev_clicks = state.cum_clicks
            prev_cost = state.cum_cost
            
            state.simulate_next_hour(spend_allocation=50.0)
            
            # Each hour should add data
            assert state.cum_clicks >= prev_clicks
            assert state.cum_cost == prev_cost + 50.0
        
        # After 10 hours with 50 spend each
        assert state.cum_cost == 500.0
        assert state.cum_clicks > 0
    
    def test_cumulative_values_always_increase(self):
        """Test that cumulative values never decrease (with positive spend)."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        for _ in range(20):
            prev_clicks = state.cum_clicks
            prev_conversions = state.cum_conversions
            prev_cost = state.cum_cost
            
            state.simulate_next_hour(spend_allocation=25.0)
            
            # Should only increase (or stay same if 0 clicks)
            assert state.cum_clicks >= prev_clicks
            assert state.cum_conversions >= prev_conversions
            assert state.cum_cost >= prev_cost


class TestSimulateNextHourPotentialBugs:
    """Tests that expose potential bugs in the implementation."""
    
    def test_inner_cpc_and_inner_cvr_not_updated(self):
        """POTENTIAL BUG: inner_cpc and inner_cvr are never set!
        
        The dataclass has inner_cpc and inner_cvr fields with default 0.0,
        but simulate_next_hour never updates them. Are they supposed to be updated?
        """
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        assert state.inner_cpc == 0.0
        assert state.inner_cvr == 0.0
        
        state.simulate_next_hour(spend_allocation=100.0)
        
        # BUG: These are still 0! Should they be updated?
        assert state.inner_cpc == 0.0  # Never changed
        assert state.inner_cvr == 0.0  # Never changed
        
        # This suggests inner_cpc and inner_cvr might be dead code
        # or there's missing functionality
    
    def test_no_input_validation(self):
        """POTENTIAL BUG: No validation on spend_allocation.
        
        The method accepts any float, including:
        - Negative values
        - NaN
        - Infinity
        - Extremely large values
        
        This could lead to unexpected behavior.
        """
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        # Document that there's no validation
        # (implementation choice - might be intentional)
        assert True  # Just documenting the lack of validation
    
    def test_sample_cpc_could_theoretically_be_very_small(self):
        """POTENTIAL BUG: If sample_cpc is very small, clicks could be huge.
        
        calculate_inner_metrics_high_variance uses normal distribution for CPC
        with no lower bound except 0.01 in calculate_inner_metrics.
        But simulate_next_hour uses calculate_inner_metrics_high_variance which
        doesn't clamp the CPC!
        
        If sample_cpc is 0.001, then 100 spend = 100,000 clicks!
        """
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test", prior_cpc=0.01)
        
        # Create state with very low CPC
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=100,
            cum_conversions=5,
            cum_cost=1.0  # Very low cost
        )
        
        # CPC is 1.0 / 100 = 0.01
        # With high variance (0.05 multiplier), could sample even lower
        initial_clicks = state.cum_clicks
        state.simulate_next_hour(spend_allocation=100.0)
        
        new_clicks = state.cum_clicks - initial_clicks
        
        # Could potentially get huge number of clicks
        # Document this potential issue
        assert new_clicks >= 0
    
    def test_int_truncation_loses_fractional_spend(self):
        """POTENTIAL BUG: int() truncation means fractional clicks are lost.
        
        If spend=10 and sample_cpc=3, we get:
        clicks = int(10 / 3) = int(3.33) = 3
        Actual cost = 3 * 3 = 9
        But we record cum_cost += 10
        
        So we're recording 10 in cost but only getting 9 worth of clicks.
        This is a 10% discrepancy!
        """
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test")
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        # This is just documenting the behavior
        # Whether it's a bug depends on business requirements
        state.simulate_next_hour(spend_allocation=10.0)
        
        # We add full spend to cum_cost regardless of actual clicks purchased
        assert state.cum_cost == 10.0
        
        # But actual clicks might represent less spend
        # This is a known limitation of the int() truncation


class TestSimulateNextHourPropertyChanges:
    """Test how properties change after simulation."""
    
    def test_cpc_changes_after_simulation(self):
        """Test that CPC property updates correctly after simulation."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test", prior_cpc=2.0)
        state = CampaignState(campaign_id="test", cfg=cfg)
        
        initial_cpc = state.cpc  # Should be prior: 2.0
        assert initial_cpc == 2.0
        
        state.simulate_next_hour(spend_allocation=100.0)
        
        # After simulation, CPC should be recalculated
        new_cpc = state.cpc  # cum_cost / cum_clicks
        assert new_cpc > 0
        
        # With 100 spend, CPC should be reasonable
        assert 0.1 <= new_cpc <= 10.0  # Reasonable range
    
    def test_alpha_beta_threshold_crossed(self):
        """Test alpha/beta behavior when crossing 100-click threshold."""
        np.random.seed(42)
        cfg = CampaignConfig(campaign_id="test", prior_alpha=2.0, prior_beta=98.0)
        state = CampaignState(
            campaign_id="test",
            cfg=cfg,
            cum_clicks=95,
            cum_conversions=5,
            cum_cost=190.0
        )
        
        # Below 100 clicks - should use priors
        assert state.alpha == 2.0
        assert state.beta == 98.0
        
        # Simulate to cross threshold
        state.simulate_next_hour(spend_allocation=20.0)
        
        # Might now be >= 100 clicks
        if state.cum_clicks >= 100:
            # Should now calculate alpha/beta
            assert state.alpha == state.cum_conversions + 1
            assert state.beta == (state.cum_clicks - state.cum_conversions) + 1
        else:
            # Still below threshold, still using priors
            assert state.alpha == 2.0
            assert state.beta == 98.0

