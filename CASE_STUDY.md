# Case Study – Benchmark of Allocation Algorithms

The purpose of this exercise is to **build a benchmark** to compare different scoring and budget allocation algorithms under controlled simulated conditions.

## Why

We aim to understand **which estimation and allocation strategies** yield the best results in ad performance.  
Although the full system involves many interacting variables (conversion variance per hour, CPC, cold start, caps, pacing, exploration), we’ll keep these **fixed** to focus only on the **allocator behavior**:

- **Score function:** how campaigns are ranked (e.g., Thompson Sampling vs. baseline CVR)
- **Allocation strategy:** how the hourly budget is distributed once scores are known (e.g., greedy, proportional, knapsack-like)

---

## Metrics

| Metric | Description |
|--------|--------------|
| **Regret** | Lost conversions vs. optimal allocation |
| **CPA (Cost per Acquisition)** | Cost per conversion achieved |
| **Total conversions** | Overall campaign performance |
| **Pacing error** | Deviation from planned hourly budget |

**Others to explore (out of scope for now):**
- Volatility  
- Exploration share  
- Final profit  
- Cap violations  

---

## Simulation Design

To ensure reproducibility, we **separate campaign simulation from estimation**:

- **Empiric campaigns:**  
  Each campaign has an internal “true” conversion rate and CPC used to generate synthetic outcomes (clicks, conversions).
  
- **Estimator / allocator:**  
  Observes only simulated data and decides hourly spend for the next period.

To have enough samples on same algorithm we compare 1 vs N attems over the same algorithm and obtain the last metrics as mean

### Algorithms & Method

TBC

# Limitations

- The inner empiric campaign in real life can vary over time so results are highly dependant on the knowledge of how that works
- we assume conversions are immediate for the scope of this project