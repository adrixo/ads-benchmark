# run

```
poetry install
poetry shell
python src/allocator.py

pytest tests/test_campaign_state.py -v
```

Part 1– Build Pipeline evaluator


Build a function that runs each simulated hour and decides how to split the next hour’s budget.

Requirements:
- Use the data set to estimate each campaign’s conversion (conv/clicks) rate and CPA. (cost/conv)
- Reallocate more spend to campaigns that are:
    - under their CPA cap, 
    - and getting better conversion rates.
- If data is missing (cold start), assume a prior average conversion rate.
- Keep total hourly spend ≤ daily budget.

Optional (bonus):
Use a simple bandit-like rule (e.g., Thompson Sampling with Beta priors).

Part 2 – Guardrails & Evaluation 
Add a few safety rules and checks:
- Don’t overspend the daily budget.
- Don’t give money to campaigns whose CPA exceeds the cap.

Print out each hour:
- Allocated spend per campaign,
- Estimated CPA per campaign,
- Total conversions so far.

Then run the simulation for one or two “days” and plot (or describe) how the allocator learns to favor the better campaigns.

