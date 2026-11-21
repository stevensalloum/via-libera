# Project brief: gas-leak emergencies with TabPFN

Read this before writing code. Integrator-owned; ask for changes in your status file.

## The story (what the 3-minute video shows)

A pressurised gas line leaks in a factory. Within one minute, from **14 fixed sensors and
the wind**, our system:

1. **finds the leak** as a probability map, honest about ambiguity ("valve A *or* B");
2. **predicts the hazard**: where it will be toxic, and where gas meets a hot machine
   (fire risk), over the next two minutes;
3. **evacuates everyone on the safest route**, not the fixed nearest-exit plan, and
   flags exits that will be blocked; the robot then goes to the leak for maintenance.

Why TabPFN: no per-plant model training, answers in one forward pass, and **calibrated
probabilities**, which a safety decision needs. Each of the three stages is a different,
measurable TabPFN use. Pitch claim: *"fewer people exposed than the standard evacuation
plan, using sensors the plant already has, with no model training per site."*

## System (all code in `src/tabpfn_spatial/`)

```
sim.Factory + sim.Scenario --readings()--> SensorReadings
  -> source.locate()   -> SourceEstimate (prob map over the grid, rate quantiles)
  -> hazard.predict()  -> HazardMap (P(toxic), P(fire) per cell at future times)
  -> evac.plan()       -> EvacPlan (one route per person)
pipeline.run_episode / evaluate: runs all of it on random leaks and scores it on the truth
```

Interfaces live in `contracts.py`; implement your protocol, keep your baseline.

**Simulator facts (v0):** 60 m x 40 m hall, 2 m cells (30 x 20 grid), two halls joined by
two doors, 6 machines (furnace, generator, dryer are hot = ignition), 4 exits, 14 sensors.
Leaks start at t = 0 next to a machine, rate 0.5-2, wind 0.2-1.5 m/s in any direction.
The system acts at `T_OBS = 60 s` (sensors sampled every 5 s) and forecasts to 180 s.
Toxic above `TOXIC_LIMIT = 0.1` (~15 % of the floor), flammable between `LEL = 0.3` and
`UEL = 3.0`, fire only next to hot machines. 500 leaks x 120 s simulate in ~1 s.

**Simulator v1 options (all off by default; the team switches them on together):**
`sample_scenarios(..., drift=0.05)` makes the wind direction wander during the leak
(`sim.wind_at(scenario, times)` gives the true wind; readings report the window mean);
`readings(..., dropout=0.1, stuck=0.05)` adds dead sensors (rows of NaN) and stuck ones;
`Factory.random(rng)` builds a new connected plant layout (for the "any plant" claim).

## Baselines to beat (`uv run leak-demo --episodes 30`, seed 0)

| Metric | Baseline | Lower is better |
|---|---|---|
| `source_error_m` (most likely cell to true leak) | **9.7 m** (weighted sensor centroid) | yes |
| `hazard_brier` (P(harm) vs truth, all forecast times) | **0.091** (forward sim from the single best guess, 0/1) | yes |
| `exposed_plan` vs `exposed_standard` (of 30 people) | **5.9 vs 7.3** (risk-aware Dijkstra on the baseline hazard) | yes |

Compare your version with `pipeline.evaluate(Factory.default(), 30, seed=0, locator=...)`
(or `predictor=`, `planner=`); same seeds, so numbers are comparable across people.

## Area tasks

### Leak source detection (Rayan, `source/`)

- **Idea:** simulation-based inference. Simulate thousands of leaks, one row per leak:
  sensor features + wind -> where the leak was. TabPFN then returns the posterior for a new
  leak in one call (our prep work found this 10x better than "it's in the middle" for
  hidden object weights).
- **Rows:** per sensor, `log1p` of readings at a few times (e.g. 20, 40, 60 s), first time
  above a threshold (arrival time), plus wind u, v. ~14 x 4 + 2 features.
- **Targets, recommended:** classify the leak **zone** (cluster `factory.leak_sites` into
  <= 100 zones; TabPFN-3.5 supports up to 160 classes) -> `predict_proba` is the map,
  naturally multimodal. Spread zone probability over its cells. Optional second model:
  release rate as a regressor (`QuantileDist`) for `SourceEstimate.rate`.
- **Data:** `sim.sample_scenarios` + `sim.simulate` in batches (fast), with
  `sim.sensor_values` + the same noise as `sim.readings`. 2,000-5,000 training leaks.
- **Measure:** `source_error_m`, P(true zone) and top-3 accuracy vs baseline; calibration.
- **Budget:** fit once, locate many leaks in **one** batched call for benchmarks.

### Hazard prediction (Ahmad, `hazard/`)

- **Idea:** predict the concentration distribution per cell and time, then
  P(toxic) = P(c > TOXIC_LIMIT), P(fire) = P(LEL <= c <= UEL) x ignition. The baseline gives
  0/1 maps from one guess; TabPFN gives calibrated probabilities that also carry the
  source uncertainty.
- **Rows:** one per (cell, time): position relative to the leak in wind coordinates
  (downwind, crosswind distance), distance, time since leak, wind speed, release rate,
  open-floor fraction between leak and cell (walls block), plus sensor summary features.
- **Source uncertainty:** evaluate rows for the top-k source cells of `SourceEstimate` and
  mix the probabilities with their weights.
- **Data:** simulated leaks x subsampled cells x forecast times (true concentration as
  target). Regressor -> `QuantileDist` (exceedance from quantiles), or classifiers for
  toxic and flammable directly.
- **Measure:** `hazard_brier`, calibration (ECE), and AUC of P(toxic) vs truth.
- **Budget:** one batched predict per episode (~600 cells x 12 times x k sources rows).

### Simulator and evacuation (Steven, `sim/`, `evac/`)

- **Sim (first):** realism where it changes decisions: wind that shifts during the leak,
  sensor dropouts, random factory layouts (to claim "any plant, no training"); keep it fast
  and keep `contracts` stable.
- **Evac:** a **risk-budget** planner: shortest route whose P(harm) <= 1 %, else the safest;
  mark exits blocked when P(harm) there is high at arrival time; compare routes planned on
  the TabPFN hazard vs the mean-only hazard vs the fixed plan. Stretch: robot guide that
  leads groups, then drives to the leak (maintenance scene).
- **UI / video:** live 2D (or 3D, from the prep repo's three.js world) view of one
  emergency: plume growing, source map sharpening, routes switching.

## Milestones

| When | Milestone | Done when |
|---|---|---|
| now | M0 skeleton | this PR merged; everyone runs `uv run leak-demo` |
| +2 h | M1 first TabPFN win | each area: TabPFN class + test + one `evaluate` number vs baseline in status |
| +4 h | M2 integrated | pipeline runs with all three TabPFN parts; headline numbers on 30+ episodes |
| +6 h | M3 demo | UI shows one emergency end to end; figures for the video |
| end | M4 video | 3-minute video recorded and submitted |

## Rules of thumb

- Train on simulated leaks with **fixed seeds**; evaluate on **other** seeds.
- Every number names its baseline, episodes, seed, backend and tokens spent.
- Cloud (`backend="cloud"`) for training and benchmarks (batched); local
  (`uv sync --extra local`) only on a GPU machine for live loops.
