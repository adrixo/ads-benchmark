based on the exercise create a small using this structure. don't extend too much

# Case study

The propose of this exercise is to quickly create a benchmark to compare different score and budget allocation algorithms

why?:
We are not certain about what estimation strategies and allocation produce the best results in terms of gain per ads.
there are multiple variables to compare in a complex system (Variance of conversion rate per hour, cost per click, cold start values, max caps, minimum allocations, budget variation over day, exploration...) but we keep those stable to only focus on the study of allocator: score function (vs baseline) and allocation strategies 

## Metrics:
regret
CPA
total conversion
pacing error
### others to explore (out of this scope)
volatility
exploration share
final profit
cap violations

## Simulation
We need to detach the empiric campaign simulation from the estimation, since if not  and will be hard to reproduce, which imposibilities comparision
- campaigns will have an empiric inner conversion/cpc
- that empiric inner can change (measure how good algorithm adapts to change)