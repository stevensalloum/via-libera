# Status: Steven (@stevensalloum)

Only Steven's agent edits this file. Update it before every PR.

## Doing

- Risk-budget evacuation planner (#21).

## Done

- Sim v1 (#20): drifting wind (`Scenario.drift`, `sim.wind_at`), sensor dropouts (NaN rows)
  and stuck sensors in `readings`, `Factory.random(rng)` (connected random plants). All off
  by default, so existing libraries are unchanged. Baselines re-measured after the seed
  change: source 9.7 m, hazard Brier 0.091, exposed 5.9 vs 7.3 of 30. 500 drifting leaks x
  120 s in 1.0 s.
- M0 walking skeleton: contracts, simulator v0, baselines for source / hazard / evac,
  `pipeline.evaluate` and `uv run leak-demo`. Baselines (30 episodes, seed 0): source 9.4 m,
  hazard Brier 0.089, exposed 5.4 (plan) vs 7.0 (standard) of 30.

## Needs

- Nothing yet.

## Blocked

- Nothing.
