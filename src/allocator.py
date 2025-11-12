from dataclasses import dataclass
from matplotlib.dates import TU
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from matplotlib import pyplot as plt

@dataclass
class CampaignConfig:
    campaign_id: str
    cpa_cap: float = 100
    prior_alpha: float = 2.0
    prior_beta: float = 98.0
    prior_cpc: float = 2.0
    prior_cpa: float = 45.0
    prior_cvr: float = 0.02
    cpc_std: float = 0.05


@dataclass
class CampaignState:
    campaign_id: str
    cfg: CampaignConfig

    # Stats tracking
    cum_clicks: int = 0
    cum_conversions: int = 0
    cum_cost: float = 0.0
    cum_spend_allocated: float = 0.0
    hour_spend_allocated: float = 0.0

    inner_cpc: float = 0.0
    inner_cvr: float = 0.0

    frozen: bool = False

    @property
    def cpa(self):
        if self.cum_conversions == 0:
            return self.cfg.prior_cpa
        return self.cum_cost / self.cum_conversions
    
    @property
    def cvr(self):
        if self.cum_clicks == 0:
            return self.cfg.prior_cvr
        return self.cum_conversions / self.cum_clicks
    
    @property
    def cpc(self):
        if self.cum_clicks == 0:
            return self.cfg.prior_cpc
        return self.cum_cost / self.cum_clicks
    
    @property
    def alpha(self):
        if self.cum_clicks < 100:
            return self.cfg.prior_alpha
        return self.cum_conversions + 1
    
    @property
    def beta(self):
        if self.cum_clicks < 100:
            return self.cfg.prior_beta
        return self.cum_clicks - self.cum_conversions + 1

    def calculate_inner_metrics(self):
        new_inner_cpc = max(0.01, np.random.normal(self.cpc, self.cpc * 0.009))
        new_inner_cvr = np.random.normal(self.cvr, self.cvr * 0.01) # we don't want a beta because we assume the CPC doesn't vary through that (we are not pickling a sample)
        return new_inner_cpc, new_inner_cvr
    

    def calculate_inner_metrics_high_variance(self):
        new_inner_cpc = np.random.normal(self.cpc, self.cpc * 0.05)
        # Sample conversion rate from Beta distribution based on alpha and beta
        new_inner_cvr = np.random.beta(max(0.0, self.alpha), max(0.0, self.beta))
        new_inner_cvr = max(0.0, min(1.0, new_inner_cvr))
        return new_inner_cpc, new_inner_cvr

    def simulate_next_hour(self, spend_allocation: float):
        sample_cpc, sample_cvr = self.calculate_inner_metrics_high_variance()
        clicks = int(spend_allocation / sample_cpc)
        conversions = np.random.binomial(clicks, sample_cvr)
        self.cum_clicks += clicks
        self.cum_conversions += conversions
        self.cum_cost += spend_allocation
        self.cum_spend_allocated += spend_allocation
        self.hour_spend_allocated = spend_allocation


@dataclass
class AllocatorConfig:
    daily_budget: float = 7200
    current_daily_budget: float = 7200
    hours: int = 24


@dataclass
class BudgetAllocator:
    # TODO: avoid primitive obsession on return values
    cfg: AllocatorConfig

    def __init__(self, cfg: AllocatorConfig = AllocatorConfig()):
        self.cfg = cfg

    def allocate_hour(self, campaign_states: List[CampaignState]) -> Dict[str, float]:
        allocated_spend = {}

        scores = self.get_scores(campaign_states)
        allocated_spend = self.allocate_spend_weighted_round_robin(scores)

        self.reduce_current_daily_budget(allocated_spend)

        return allocated_spend

    def get_scores(self, campaign_states: List[CampaignState]) -> List[Tuple[str, float, float, float]]:
        """
        Bayesian scoring using Beta distribution for CVR sampling.
        Incorporates uncertainty through alpha/beta parameters.
        """
        scores: List[Tuple[str, float, float, float]] = []  # (cid, score, sampled_cpa, sample_cpc)
        for state in campaign_states:
            p_sample = np.random.beta(state.alpha, state.beta)
            p_sample = float(max(1e-8, min(1.0, p_sample)))
            sample_cpc = max(1e-8, state.cpc)
            sampled_cpa = sample_cpc / p_sample
            score = p_sample / sample_cpc  # conversions per $ (higher is better)
            
            if sampled_cpa > state.cfg.cpa_cap:
                score = 0.0

            scores.append((state.campaign_id, score, sampled_cpa, sample_cpc))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def get_scores_baseline(self, campaign_states: List[CampaignState]) -> List[Tuple[str, float, float, float]]:
        """
        Baseline trivial scoring: samples CPA using normal distribution around historical CPA.
        """
        scores: List[Tuple[str, float, float, float]] = []  # (cid, score, sampled_cpa, sample_cpc)
        
        for state in campaign_states:
            sampled_cpa = np.random.normal(state.cpa, state.cpa *0.5)
            sampled_cpa = max(1e-8, sampled_cpa)
            
            sample_cpc = state.cpc
            
            # Simple score: inverse of CPA (lower CPA = higher score)
            score = 1.0 / sampled_cpa if sampled_cpa > 0 else 0.0
            
            if sampled_cpa > state.cfg.cpa_cap:
                score = 0.0
            
            scores.append((state.campaign_id, score, sampled_cpa, sample_cpc))
        
        max_score = max((s for _, s, _, _ in scores), default=1.0)
        if max_score > 0:
            scores = [(cid, min(1.0, score / max_score), cpa, cpc) for cid, score, cpa, cpc in scores]
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def allocate_spend_square_normalization(self, scores: List[Tuple[str, float, float, float]], power: float = 3) -> Dict[str, float]:
        if not scores:
            return {}
        
        hourly_budget = self.cfg.daily_budget / 24
        
        # Apply power function to overcompensate better scores
        # Higher power = more aggressive overcompensation
        powered_scores = [max(1e-8, score) ** power for _, score, _, _ in scores]
        total_powered_score = sum(powered_scores)
        
        if total_powered_score < 1e-8:
            equal_share = hourly_budget / len(scores)
            return {cid: equal_share for cid, _, _, _ in scores}
        
        allocations = {}
        for (cid, _, _, _), powered_score in zip(scores, powered_scores):
            allocations[cid] = (powered_score / total_powered_score) * hourly_budget
        
        return allocations

    def allocate_spend_normalization(self, scores: List[Tuple[str, float, float, float]]) -> Dict[str, float]:
        if not scores:
            return {cid: 0.0 for cid, _, _, _ in scores}
        
        hourly_budget = self.cfg.daily_budget / 24
        
        total_score = sum(score for _, score, _, _ in scores)
        
        if total_score < 1e-8:
            equal_share = hourly_budget / len(scores)
            return {cid: equal_share for cid, _, _, _ in scores}
        
        allocations = {}
        for cid, score, _, _ in scores:
            allocations[cid] = (score / total_score) * hourly_budget
        
        return allocations

    def allocate_spend_weighted_round_robin(self, scores: List[Tuple[str, float, float, float]]) -> Dict[str, float]:
        allocations: Dict[str, float] = {cid: 0.0 for cid, _, _, _ in scores}
        
        scored = scores
        hourly_budget = self.cfg.daily_budget / 24
        remaining = hourly_budget
        
        # Check if all scores are 0 (all campaigns violate caps)
        total_score = sum(score for _, score, _, _ in scored)
        if total_score < 1e-8:
            # Distribute equally if all scores are 0
            equal_share = hourly_budget / len(scored) if scored else 0.0
            return {cid: equal_share for cid, _, _, _ in scored}

        i = 0
        max_iterations = len(scored) * 1000  # Safety limit to prevent infinite loops
        iterations = 0
        
        while remaining > 1e-6 and i < len(scored) and iterations < max_iterations:
            name, score, _, _ = scored[i]
            
            # Skip campaigns with score = 0
            if score > 1e-8:
                chunk = min(remaining, max(10.0, 0.2 * hourly_budget))
                amt = chunk * score / max(1e-8, score)
                allocations[name] = allocations.get(name, 0.0) + amt
                remaining -= amt
            
            i += 1
            iterations += 1
            if i == len(scored):
                i = 0

        return allocations
    
    def allocate_spend_ucb_optimal(self, scores: List[Tuple[str, float, float, float]], exploration_factor: float = 0.1) -> Dict[str, float]:
        if not scores:
            return {}
        
        hourly_budget = self.cfg.daily_budget / 24
        
        # Filter out campaigns with score=0 (cap violations)
        valid_scores = [(cid, score, cpa, cpc) for cid, score, cpa, cpc in scores if score > 0]
        
        if not valid_scores:
            # If all campaigns violate caps, distribute equally
            equal_share = hourly_budget / len(scores)
            return {cid: equal_share for cid, _, _, _ in scores}
        
        ucb_scores = []
        max_score = max(score for _, score, _, _ in valid_scores)
        
        for cid, score, cpa, cpc in valid_scores:
            normalized_score = score / max_score if max_score > 0 else 0
            exploration_bonus = exploration_factor * (1.0 / np.sqrt(max(score, 1e-6)))
            ucb = normalized_score + exploration_bonus
            ucb_scores.append((cid, score, ucb, cpa, cpc))
        
        ucb_scores.sort(key=lambda x: x[2], reverse=True)
        
        temperature = 5.0
        ucb_values = np.array([ucb for _, _, ucb, _, _ in ucb_scores])
        
        exp_ucb = np.exp((ucb_values - np.max(ucb_values)) / temperature)
        softmax_weights = exp_ucb / np.sum(exp_ucb)
        
        allocations = {}
        for (cid, _, _, _, _), weight in zip(ucb_scores, softmax_weights):
            allocations[cid] = float(weight * hourly_budget)
        
        for cid, score, _, _ in scores:
            if score == 0 and cid not in allocations:
                allocations[cid] = 0.0
        
        return allocations

    def reduce_current_daily_budget(self, allocated_spend: Dict[str, float]):
        self.cfg.current_daily_budget -= sum(allocated_spend.values())

    def restart_day(self):
        self.cfg.current_daily_budget = self.cfg.daily_budget


class Metric:
    """Individual metric calculation methods"""
    
    @staticmethod
    def regret(campaign: CampaignState, score_info: Tuple[str, float, float, float]) -> float:
        """
        Measures regret: the difference between actual CPA and sampled CPA.
        If actual CPA is higher than sampled CPA, there's regret (we're paying more than expected).
        Returns 0 if the campaign was not considered (score=0 due to cap violation).
        """
        _, score, sampled_cpa, _ = score_info
        
        # If score is 0, campaign was not considered due to cap violation
        if score == 0.0:
            return 0.0
        
        # Actual CPA achieved
        actual_cpa = campaign.cpa
        
        # Regret is the absolute difference between actual and expected CPA
        return abs(actual_cpa - sampled_cpa)
    
    @staticmethod
    def cpa(campaign: CampaignState) -> float:
        """Returns the actual Cost Per Acquisition for the campaign."""
        return campaign.cpa
    
    @staticmethod
    def total_conversions(campaign: CampaignState) -> float:
        """Returns the total cumulative conversions achieved."""
        return float(campaign.cum_conversions)
    
    @staticmethod
    def cap_violations(campaign: CampaignState) -> float:
        """
        Measures how much the actual CPA exceeds the cap.
        Returns 0 if within cap, otherwise returns the excess amount.
        """
        actual_cpa = campaign.cpa
        cap = campaign.cfg.cpa_cap
        
        # Return the violation amount (0 if no violation)
        return max(0.0, actual_cpa - cap)


class SimulatorMetrics:
    """Aggregates metrics across campaigns"""
    
    @staticmethod
    def calculate_metrics(campaigns: List[CampaignState], allocated_spend: Dict[str, float], scores: List[Tuple[str, float, float, float]]) -> Dict[str, Any]:
        """
        Calculate metrics for each campaign and aggregate metrics across all campaigns.
        
        Args:
            campaigns: List of campaign states
            allocated_spend: Dict mapping campaign_id to spend amount
            scores: List of scores from get_scores (cid, score, sampled_cpa, sample_cpc)
        
        Returns:
            Dict with structure:
            {
                'per_campaign': {
                    'campaign_id': {
                        'regret': float,
                        'cpa': float,
                        'total_conversions': float,
                        'cap_violations': float
                    },
                    ...
                },
                'aggregate': {
                    'regret': float (mean),
                    'cpa': float (weighted by total spend/conversions),
                    'total_conversions': float (sum),
                    'cap_violations': float (count of campaigns with score=0)
                }
            }
        """
        per_campaign_metrics: Dict[str, Dict[str, float]] = {}
        
        # Create a dict to quickly lookup score info by campaign_id
        score_dict = {cid: score_info for score_info in scores for cid in [score_info[0]]}
        
        # Calculate metrics for each campaign
        total_spend = 0.0
        total_conversions_sum = 0.0
        regrets = []
        
        for campaign in campaigns:
            score_info = score_dict.get(campaign.campaign_id, (campaign.campaign_id, 0.0, 0.0, 0.0))
            
            campaign_regret = Metric.regret(campaign, score_info)
            campaign_cpa = Metric.cpa(campaign)
            campaign_conversions = Metric.total_conversions(campaign)
            campaign_cap_violation = Metric.cap_violations(campaign)
            
            per_campaign_metrics[campaign.campaign_id] = {
                'regret': campaign_regret,
                'cpa': campaign_cpa,
                'total_conversions': campaign_conversions,
                'cap_violations': campaign_cap_violation
            }
            
            # Accumulate for aggregate metrics
            regrets.append(campaign_regret)
            total_spend += campaign.cum_cost
            total_conversions_sum += campaign_conversions
        
        # Calculate aggregate metrics
        aggregate_metrics: Dict[str, float] = {}
        
        # Regret: average of all regrets
        aggregate_metrics['regret'] = float(np.mean(regrets)) if regrets else 0.0
        
        # CPA: sum of all cum_cost / sum of all conversions (weighted average)
        if total_conversions_sum > 0:
            aggregate_metrics['cpa'] = total_spend / total_conversions_sum
        else:
            aggregate_metrics['cpa'] = 0.0
        
        # Total conversions: sum of all conversions
        aggregate_metrics['total_conversions'] = total_conversions_sum
        
        # Cap violations: count how many campaigns have score=0 (removed due to cap)
        cap_violation_count = sum(1 for _, score, _, _ in scores if score == 0.0)
        aggregate_metrics['cap_violations'] = float(cap_violation_count)
        
        return {
            'per_campaign': per_campaign_metrics,
            'aggregate': aggregate_metrics
        }

class SampleDataAdapter:
    def __init__(self):
        from sample_data import df
        self.df = df
    
    def load(self):
        """
        For each campaign in the dataframe, obtain the accumulative spend, clicks and conversions.
        Returns:
            pd.DataFrame with columns: 
                'campaign_id', 'accum_spend', 'accum_clicks', 'accum_conversions'
        """
        agg_df = self.df.groupby('campaign').agg(
            accum_spend=pd.NamedAgg(column='spend', aggfunc='sum'),
            accum_clicks=pd.NamedAgg(column='clicks', aggfunc='sum'),
            accum_conversions=pd.NamedAgg(column='conversions', aggfunc='sum')
        ).reset_index()
        campaign_states = []
        for _, row in agg_df.iterrows():
            state = CampaignState(
                campaign_id=row['campaign'],
                cfg=CampaignConfig(campaign_id=row['campaign']),
                cum_clicks=row.get('accum_clicks', 0),
                cum_conversions=row.get('accum_conversions', 0),
                cum_cost=row.get('accum_spend', 0.0),
            )
            campaign_states.append(state)
        return campaign_states


class Plotter:
    def __init__(self):
        self.history: Dict[str, Dict[str, List[float]]] = {}
        self.metrics_history: Dict[str, Dict[str, List[float]]] = {}
        self.aggregate_metrics_history: Dict[str, List[float]] = {}
        self.time = []
        self.t = 0
    
    def add_data_points(self, campaign_states: List[CampaignState], metrics_data: Optional[Dict[str, Any]] = None):
        self.time.append(self.t)
        self.t += 1
        
        # Add campaign state data
        for state in campaign_states:
            if state.campaign_id not in self.history:
                self.history[state.campaign_id] = {k: [] for k in ['cpa', 'cvr', 'cpc', 'conv', 'clicks', 'spend', 'hour_spend']}
            h = self.history[state.campaign_id]
            h['cpa'].append(state.cpa if state.cum_conversions > 0 else 0)
            h['cvr'].append(state.cvr if state.cum_clicks > 0 else 0)
            h['cpc'].append(state.cpc)
            h['conv'].append(state.cum_conversions)
            h['clicks'].append(state.cum_clicks)
            h['spend'].append(state.cum_cost)
            h['hour_spend'].append(state.hour_spend_allocated)
        
        # Add metrics data if provided
        if metrics_data:
            # Per-campaign metrics
            for campaign_id, campaign_metrics in metrics_data.get('per_campaign', {}).items():
                if campaign_id not in self.metrics_history:
                    self.metrics_history[campaign_id] = {k: [] for k in ['regret', 'cpa', 'total_conversions', 'cap_violations']}
                for metric_name, metric_value in campaign_metrics.items():
                    self.metrics_history[campaign_id][metric_name].append(metric_value)
            
            # Aggregate metrics
            for metric_name, metric_value in metrics_data.get('aggregate', {}).items():
                if metric_name not in self.aggregate_metrics_history:
                    self.aggregate_metrics_history[metric_name] = []
                self.aggregate_metrics_history[metric_name].append(metric_value)
    
    def show(self):
        # First plot: Campaign states
        self._show_campaign_states()
        
        # Second plot: Per-campaign metrics
        self._show_per_campaign_metrics()
        
        # Third plot: Aggregate metrics
        self._show_aggregate_metrics()
    
    def _show_campaign_states(self):
        """Show original campaign state plots"""
        fig, axs = plt.subplots(4, 2, figsize=(12, 14))
        fig.suptitle('Campaign States Over Time', fontsize=16)
        axs = axs.flatten()
        titles = ["CPA", "CVR", "CPC", "Conversions", "Clicks", "Spend", "Hour Spend Allocated"]
        metrics = ['cpa', 'cvr', 'cpc', 'conv', 'clicks', 'spend', 'hour_spend']
        
        for i, (ax, title, metric) in enumerate(zip(axs, titles, metrics)):
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            if i == 6:  # Stacked plot for hour spend allocation
                spend_data = [h['hour_spend'] for h in self.history.values()]
                if spend_data and len(spend_data[0]) > 0:
                    ax.stackplot(self.time, *spend_data, labels=list(self.history.keys()), alpha=0.7)
                ax.legend()
            else:
                for cid, h in self.history.items():
                    ax.plot(self.time, h[metric], marker='o', label=cid, markersize=3)
                ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _show_per_campaign_metrics(self):
        """Show per-campaign metrics over time"""
        if not self.metrics_history:
            return
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Per-Campaign Metrics Over Time', fontsize=16)
        axs = axs.flatten()
        
        metric_names = ['regret', 'cpa', 'total_conversions', 'cap_violations']
        metric_titles = ['Regret', 'CPA', 'Total Conversions', 'Cap Violations']
        
        for i, (metric_name, title) in enumerate(zip(metric_names, metric_titles)):
            ax = axs[i]
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel(title)
            
            for campaign_id, metrics_data in self.metrics_history.items():
                if metric_name in metrics_data and len(metrics_data[metric_name]) > 0:
                    # Skip first data point (t=0)
                    ax.plot(self.time[1:], metrics_data[metric_name], marker='o', label=campaign_id, markersize=3)
            
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _show_aggregate_metrics(self):
        """Show aggregate metrics over time"""
        if not self.aggregate_metrics_history:
            return
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Aggregate Metrics Over Time (All Campaigns)', fontsize=16)
        axs = axs.flatten()
        
        metric_names = ['regret', 'cpa', 'total_conversions', 'cap_violations']
        metric_titles = ['Regret (Mean)', 'CPA (Weighted)', 'Total Conversions (Sum)', 'Cap Violations (Count)']
        
        for i, (metric_name, title) in enumerate(zip(metric_names, metric_titles)):
            ax = axs[i]
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel(title)
            
            if metric_name in self.aggregate_metrics_history and len(self.aggregate_metrics_history[metric_name]) > 0:
                ax.plot(self.time[1:], self.aggregate_metrics_history[metric_name], marker='o', color='darkblue', linewidth=2, markersize=4)
        
        plt.tight_layout()
        plt.show()


def calculate_final_metrics(
    accumulated_per_campaign: Dict[str, Dict[str, float]], 
    accumulated_aggregate: Dict[str, float], 
    hours_count: int
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Calculate final metrics by averaging or summing depending on metric type.
    
    Args:
        accumulated_per_campaign: Accumulated per-campaign metrics
        accumulated_aggregate: Accumulated aggregate metrics
        hours_count: Number of hours simulated
    
    Returns:
        Tuple of (per_campaign_metrics, aggregate_metrics)
    """
    # Metrics that should be summed instead of averaged
    metrics_to_sum = {'cap_violations', 'total_conversions'}
    
    # Process per-campaign metrics
    final_per_campaign = {}
    for campaign_id, metrics_sum in accumulated_per_campaign.items():
        final_per_campaign[campaign_id] = {}
        for metric_name, metric_value in metrics_sum.items():
            if metric_name in metrics_to_sum:
                # Sum these metrics instead of averaging
                final_per_campaign[campaign_id][metric_name] = metric_value
            else:
                # Average other metrics
                final_per_campaign[campaign_id][metric_name] = metric_value / hours_count
    
    # Process aggregate metrics
    final_aggregate = {}
    for metric_name, metric_value in accumulated_aggregate.items():
        if metric_name in metrics_to_sum:
            # Sum these metrics instead of averaging
            final_aggregate[metric_name] = metric_value
        else:
            # Average other metrics
            final_aggregate[metric_name] = metric_value / hours_count
    
    return final_per_campaign, final_aggregate


def run_single_simulation(
        simulation_hours: int,
        allocation_method: str,
        allocation_func,
        scoring_func,
        initial_campaign_states: List[CampaignState],
        allocator_cfg: AllocatorConfig
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Run a single simulation with a specific allocation method and scoring function.
    
    Args:
        simulation_hours: Number of hours to simulate
        allocation_method: Name of the allocation method
        allocation_func: The allocation function to use
        scoring_func: The scoring function to use
        initial_campaign_states: Initial campaign states to copy
        allocator_cfg: Allocator configuration
    
    Returns:
        Tuple of (final_per_campaign_metrics, final_aggregate_metrics)
    """
    import copy
    
    campaign_states = copy.deepcopy(initial_campaign_states)
    
    allocator = BudgetAllocator(cfg=copy.deepcopy(allocator_cfg))
    
    accumulated_per_campaign: Dict[str, Dict[str, float]] = {}
    accumulated_aggregate: Dict[str, float] = {}
    hours_count = 0
    
    for hour in range(simulation_hours):
        scores = scoring_func(campaign_states)
        spend_allocation: Dict[str, float] = allocation_func(scores)
        allocator.reduce_current_daily_budget(spend_allocation)
        
        for state in campaign_states:
            state.simulate_next_hour(spend_allocation[state.campaign_id])

        if hour % 24 == 0:
            allocator.restart_day()

        metrics = SimulatorMetrics.calculate_metrics(campaign_states, spend_allocation, scores)
        
        # Accumulate per-campaign metrics
        for campaign_id, campaign_metrics in metrics['per_campaign'].items():
            if campaign_id not in accumulated_per_campaign:
                accumulated_per_campaign[campaign_id] = {k: 0.0 for k in campaign_metrics.keys()}
            for metric_name, metric_value in campaign_metrics.items():
                accumulated_per_campaign[campaign_id][metric_name] += metric_value
        
        # Accumulate aggregate metrics
        for metric_name, metric_value in metrics['aggregate'].items():
            if metric_name not in accumulated_aggregate:
                accumulated_aggregate[metric_name] = 0.0
            accumulated_aggregate[metric_name] += metric_value
        
        hours_count += 1

    # Calculate final metrics
    final_per_campaign, final_aggregate = calculate_final_metrics(
        accumulated_per_campaign, 
        accumulated_aggregate, 
        hours_count
    )
    
    return final_per_campaign, final_aggregate


def run_simulation(
        simulation_hours: int = 48,
        num_runs: int = 500
):
    """
    Run simulations comparing 2 scoring functions × 4 allocation algorithms (8 combinations).
    Each combination is run multiple times and results are averaged for accuracy.
    
    Args:
        simulation_hours: Number of hours to simulate per run
        num_runs: Number of times to run each combination (default: 500)
    """
    print("\n" + "="*80)
    print(f"RUNNING COMPARISON: 2 SCORING × 4 ALLOCATION METHODS ({num_runs} runs each)")
    print("="*80)
    
    sample_data_adapter = SampleDataAdapter()
    initial_campaign_states = sample_data_adapter.load()
    print(f"\nInitial campaign states loaded: {len(initial_campaign_states)} campaigns")
    print("--------------------------------")

    allocator_cfg = AllocatorConfig()
    print("\nAllocator config:")
    print(allocator_cfg)
    print("--------------------------------\n")

    # Define scoring methods to test
    temp_allocator = BudgetAllocator(cfg=allocator_cfg)
    scoring_methods = {
        'Bayesian': temp_allocator.get_scores,
        'Baseline': temp_allocator.get_scores_baseline
    }
    
    # Define allocation methods to test
    allocation_methods = {
        'Square Norm': temp_allocator.allocate_spend_square_normalization,
        'Simple Norm': temp_allocator.allocate_spend_normalization,
        'Weighted RR': temp_allocator.allocate_spend_weighted_round_robin,
        'UCB Optimal': temp_allocator.allocate_spend_ucb_optimal
    }
    
    # Store averaged results for each combination
    all_results: Dict[str, Dict[str, float]] = {}
    
    # Run simulation for each scoring + allocation combination
    for scoring_name, scoring_func in scoring_methods.items():
        for allocation_name, allocation_func in allocation_methods.items():
            combination_name = f"{scoring_name} + {allocation_name}"
            print(f"Running {num_runs} simulations: {combination_name}")
            print("-" * 80)
            
            # Accumulate metrics across all runs
            accumulated_metrics: Dict[str, float] = {}
            
            for run in range(num_runs):
                # Show progress for every run
                print(f"  Run {run + 1}/{num_runs}...", end='\r')
                
                final_per_campaign, final_aggregate = run_single_simulation(
                    simulation_hours=simulation_hours,
                    allocation_method=combination_name,
                    allocation_func=allocation_func,
                    scoring_func=scoring_func,
                    initial_campaign_states=initial_campaign_states,
                    allocator_cfg=allocator_cfg
                )
                
                # Accumulate aggregate metrics
                for metric_name, metric_value in final_aggregate.items():
                    if metric_name not in accumulated_metrics:
                        accumulated_metrics[metric_name] = 0.0
                    accumulated_metrics[metric_name] += metric_value
            
            # Calculate average metrics
            averaged_metrics = {
                metric_name: metric_value / num_runs 
                for metric_name, metric_value in accumulated_metrics.items()
            }
            
            all_results[combination_name] = averaged_metrics
            
            # Print summary for this combination
            print(f"  ✓ Completed all {num_runs} runs" + " " * 20)  # Clear the line
            print(f"Average results across {num_runs} runs:")
            print(f"  Total Conversions: {averaged_metrics['total_conversions']:.2f}")
            print(f"  Average CPA: {averaged_metrics['cpa']:.2f}")
            print(f"  Average Regret: {averaged_metrics['regret']:.2f}")
            print(f"  Cap Violations: {averaged_metrics['cap_violations']:.2f}")
            print()
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"FINAL COMPARISON (averaged over {num_runs} runs)")
    print("(regret & cpa: averaged | total_conversions & cap_violations: summed)")
    print("="*80 + "\n")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results).T
    comparison_df.index.name = 'Scoring + Allocation Method'
    print(comparison_df.to_string())
    print("\n" + "="*80)
    
    # Visualize comparison
    visualize_algorithm_comparison(comparison_df, num_runs)


def visualize_algorithm_comparison(comparison_df: pd.DataFrame, num_runs: int = 20):
    """
    Create visualization comparing all scoring + allocation combinations.
    
    Args:
        comparison_df: DataFrame with comparison metrics
        num_runs: Number of runs averaged for the results
    """
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Scoring + Allocation Method Comparison (averaged over {num_runs} runs)', fontsize=16, fontweight='bold')
    axs = axs.flatten()
    
    metrics = ['regret', 'cpa', 'total_conversions', 'cap_violations']
    metric_titles = ['Regret (Mean)', 'CPA (Weighted)', 'Total Conversions (Sum)', 'Cap Violations (Sum)']
    
    # Use different colors for Bayesian vs Baseline
    colors_bayesian = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, orange, green, red
    colors_baseline = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']  # Lighter versions
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axs[i]
        
        if metric in comparison_df.columns:
            values = comparison_df[metric]
            
            # Assign colors based on scoring method (Bayesian or Baseline)
            bar_colors = []
            for idx, label in enumerate(values.index):
                if 'Bayesian' in label:
                    # Use colors based on allocation method
                    if 'Square' in label:
                        bar_colors.append(colors_bayesian[0])
                    elif 'Simple' in label:
                        bar_colors.append(colors_bayesian[1])
                    elif 'Weighted' in label:
                        bar_colors.append(colors_bayesian[2])
                    else:  # UCB
                        bar_colors.append(colors_bayesian[3])
                else:  # Baseline
                    if 'Square' in label:
                        bar_colors.append(colors_baseline[0])
                    elif 'Simple' in label:
                        bar_colors.append(colors_baseline[1])
                    elif 'Weighted' in label:
                        bar_colors.append(colors_baseline[2])
                    else:  # UCB
                        bar_colors.append(colors_baseline[3])
            
            bars = ax.bar(range(len(values)), values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=45, ha='right', fontsize=7)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=7, fontweight='bold')
            
            # Highlight the best performer with thick green border
            if metric in ['total_conversions']:
                # Higher is better
                best_idx = values.argmax()
                bars[best_idx].set_edgecolor('green')
                bars[best_idx].set_linewidth(3)
            elif metric in ['regret', 'cpa', 'cap_violations']:
                # Lower is better
                best_idx = values.argmin()
                bars[best_idx].set_edgecolor('green')
                bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
