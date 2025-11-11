from dataclasses import dataclass
from matplotlib.dates import TU
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from matplotlib import pyplot as plt

@dataclass
class CampaignConfig:
    campaign_id: str
    cpa_cap: float = 50
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

    def allocate_hour(self, campaign_states: List[CampaignState]) -> Dict[str, Tuple[float, float, float]]:
        allocated_spend = {}

        scores = self.get_scores(campaign_states)
        allocated_spend = self.allocate_spend_square_normalization(scores)

        self.reduce_current_daily_budget(allocated_spend)

        return allocated_spend

    def get_scores(self, campaign_states: List[CampaignState]) -> List[Tuple[str, float, float, float]]:
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

    def allocate_spend_square_normalization(self, scores: List[Tuple[str, float, float, float]], power: float = 3) -> Dict[str, Tuple[float, float, float]]:
        if not scores:
            return {}
        
        hourly_budget = self.cfg.daily_budget / 24
        
        # Apply power function to overcompensate better scores
        # Higher power = more aggressive overcompensation
        powered_scores = [max(1e-8, score) ** power for _, score, _, _ in scores]
        total_powered_score = sum(powered_scores)
        
        if total_powered_score < 1e-8:
            equal_share = hourly_budget / len(scores)
            return {cid: (equal_share, sampled_cpa, sample_cpc) for cid, _, sampled_cpa, sample_cpc in scores}
        
        allocations = {}
        for (cid, _, sampled_cpa, sample_cpc), powered_score in zip(scores, powered_scores):
            allocations[cid] = ((powered_score / total_powered_score) * hourly_budget, sampled_cpa, sample_cpc)
        
        return allocations

    def allocate_spend_normalization(self, scores: List[Tuple[str, float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
        if not scores:
            return {cid: (0.0, 0.0, 0.0) for cid, _, _, _ in scores}
        
        hourly_budget = self.cfg.daily_budget / 24
        
        total_score = sum(score for _, score, _, _ in scores)
        
        if total_score < 1e-8:
            equal_share = hourly_budget / len(scores)
            return {cid: (equal_share, sampled_cpa, sample_cpc) for cid, _, sampled_cpa, sample_cpc in scores}
        
        allocations = {}
        for cid, score, sampled_cpa, sample_cpc in scores:
            allocations[cid] = ((score / total_score) * hourly_budget, sampled_cpa, sample_cpc)
        
        return allocations

    def allocate_spend_weighted_round_robin(self, scores: List[Tuple[str, float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
        allocations: Dict[str, Tuple[float, float, float]] = {cid: (0.0, 0.0, 0.0) for cid, _, _, _ in scores}
        
        scored = scores
        hourly_budget = self.cfg.daily_budget / 24
        remaining = hourly_budget

        i = 0
        while remaining > 1e-6 and i < len(scored):
            name, score, _, _ = scored[i]
            chunk = min(remaining, max(10.0, 0.2 * hourly_budget))
            amt = chunk * score / max(1e-8, score)
            allocations[name] = (allocations.get(name, (0.0, 0.0, 0.0))[0] + amt, allocations.get(name, (0.0, 0.0, 0.0))[1], allocations.get(name, (0.0, 0.0, 0.0))[2])
            remaining -= amt
            i += 1
            if i == len(scored):
                i = 0

        return allocations

    def reduce_current_daily_budget(self, allocated_spend: Dict[str, Tuple[float, float, float]]):
        self.cfg.current_daily_budget -= sum(amount for amount, _, _ in allocated_spend.values())

    def restart_day(self):
        self.cfg.current_daily_budget = self.cfg.daily_budget


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
        # For each campaign, create a CampaignState with available information only (no new fields computed).
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
        self.time = []
        self.t = 0
    
    def add_data_points(self, campaign_states: List[CampaignState]):
        self.time.append(self.t)
        self.t += 1
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
    
    def show(self):
        fig, axs = plt.subplots(4, 2, figsize=(12, 14))
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


def run_simulation(
        simulation_hours: int = 48,
):
    plotter = Plotter()

    sample_data_adapter = SampleDataAdapter()
    campaign_states = sample_data_adapter.load()
    print("Initial campaign states:")
    print("--------------------------------")
    plotter.add_data_points(campaign_states)

    allocator = BudgetAllocator()
    print("Allocator config:")
    print(allocator.cfg)
    print("--------------------------------")

    for hour in range(simulation_hours):
        spend_allocation: Dict[str, Tuple[float, float, float]] = allocator.allocate_hour(campaign_states)
        for state in campaign_states:
            state.simulate_next_hour(spend_allocation[state.campaign_id][0])

        if hour % 24 == 0:
            allocator.restart_day()
    
        plotter.add_data_points(campaign_states)

    plotter.show()


if __name__ == "__main__":
    run_simulation()
