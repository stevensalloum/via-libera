# Status: Steven (@stevensalloum)

Only Steven's agent edits this file. Update it before every PR.

## Doing

- Evacuation step 2: compare planners (A*, time-dependent, many people), then the timing
  study (harm vs alarm time and update rate).

## Done

- Live one-person evacuation (`evac.live`, `uv run evac-live`): replans every `update_s`
  seconds; live vs plan-once vs nearest exit on the true gas. 60 leaks, default settings:
  exposed 9 / 9 / 11, harm-seconds 133 / 131 / 137, of which 98 before the alarm at 30 s.
  Alarm at 15 s: harm-seconds 38 / 38 / 42. Stand-ins `OracleLocator`, `OracleHazard`.
- Gas pipe network (`Factory.pipes`, `fittings`, `leak_prior`): leaks only on pipes, fittings
  5x likelier; 24 sites in the default plant (v0: 110). Baselines with pipes (30 episodes,
  seed 0): source 10.2 m, Brier 0.076, exposed 5.6 vs 7.2. `Factory.default(pipes=False)`
  reproduces v0. **Rayan, Ahmad: training libraries need regenerating once this merges.**

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
