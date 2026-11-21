# Status: Steven (@stevensalloum)

Only Steven's agent edits this file. Update it before every PR.

## Doing

- Waiting for Rayan's and Ahmad's pushes; then run `sync-check` on both TabPFN classes and
  merge. Next build: crowd model and risk-aware capacity-constrained planner
  (`docs/research/evacuation.md`), timing study on large plants.

## Done

- Crowds (`uv run evac-crowd`, `evac-crowd-live`): crowd simulator, exit fields, three crowd
  planners, scenario ladder A-E up to 200x120 m and 1,200 people, 50 held-out leaks each.
  With perfect maps the capacity-aware planner is as safe as risk-only (45-95 % less harm
  than nearest) and 32-38 % faster on big plants; with baseline maps the planners trade
  safety for speed, so hazard-map quality decides. About 1 leak in 50 is never detected.
- Detector alarm (first 5 s sample above 10 % LEL: median 10 s, max 25 s) and
  `uv run sync-check` (closed loop, every hand-over checked, latency per call). Conventions
  and heads-up list in `docs/sync.md`.
- Timing study (`uv run evac-timing`, 100 leaks): decide early (exposed 6 % at 10 s,
  14 % at 20 s, 22 % at 60 s); replanning rate 2-60 s makes no difference; routes must land
  within about 2 s (2 s keeps ~60 % of the gain, 10 s ~0); people should walk to the
  nearest exit until the route lands, never wait. **Rayan: the locator needs a 20-30 s
  version** (today it reads 20/40/60 s).
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

- Rayan, Ahmad: run `uv run sync-check --source/--hazard <your class>` before pushing; see
  `docs/sync.md` section 4 (decision at ~10-30 s, absolute forecast times, pipes).

## Blocked

- Nothing.
