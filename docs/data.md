# Datasets and the closed loop

How we generate, split and pass data so that the three parts plug into one loop:
**readings -> source -> hazard -> evacuation**. Status: proposal for the team to approve
(2026-11-21); items marked *to build* are not in the code yet.

## 1. Which datasets

| Dataset | What it is | Used by | Status |
|---|---|---|---|
| **Simulated leaks** (`sim`) | Our simulator: random leaks next to machines, wind, noisy sensors, true concentration fields | Everyone: training tables and all evaluation | In use |
| Simulated plants (`Factory.random`) | New layouts never seen in training | Generalisation test, "any plant" claim | PR #23 |
| UCI "Gas sensor arrays in open sampling settings" (Vergara et al. 2013) | Real wind tunnel: 72 metal-oxide sensors at 6 distances from a gas source, 3 wind speeds, 10 gases | Optional real-data check of source distance estimation | Candidate, not started |

Everything we pitch is measured on the simulator; real data is a credibility bonus only if
time allows. We never commit generated tables: each area has a seeded function that
regenerates its table.

## 2. Seeds: training never sees evaluation leaks

| Seed range | Use | Rule |
|---|---|---|
| `0-99` | `pipeline.evaluate` (the shared numbers we report) | **Never train on these** |
| `500-999` | Each area's held-out benchmark | Never train on these |
| `>= 1000` | Training libraries | Each area picks its own and writes it in its status file |

Current use: Rayan trains on 12345, benchmarks on 777; Ahmad trains on 42 and evaluates on 0.
*To change:* Ahmad's training seed 42 is in the evaluation range; move it to >= 1000.

Note: the default plant has 110 leak sites, so test leaks share sites with training leaks
(different wind, rate and noise). That is fine for "this plant"; for "a new plant" we
evaluate on `Factory.random` layouts that were never in training.

## 3. One simulator setting for everyone *(to build)*

`sim.SETTINGS` (a frozen dataclass) holds `drift`, `dropout`, `stuck`, `t_obs`, sensor
interval and noise. Every library generator and `pipeline` read it, so switching the wind
drift on changes training and evaluation together. Version string `SETTINGS.version`
(e.g. `"v1-drift0.05"`) is written next to every reported number.

## 4. The closed loop: what each part consumes and produces

| Step | Input at run time (observable only) | Output | Trained on |
|---|---|---|---|
| Source | `SensorReadings` (readings up to 60 s, measured wind) | `SourceEstimate`: map + **release-rate quantiles** | Simulated readings -> true zone (and rate) |
| Hazard | `SensorReadings` + `SourceEstimate` | `HazardMap` at 70-180 s | Features from the **estimated** source and **measured** wind -> true future concentration |
| Evac | People + `HazardMap` | `EvacPlan` | No training (planner) |

Rules that make the parts compatible:

1. **Only observable inputs at run time.** Features may use readings, the measured wind and
   the `SourceEstimate`, never the scenario's true source, rate or wind. Truth is only the
   target and the score.
2. **Train downstream on upstream outputs.** Hazard is trained with the source as the
   locator would give it (best: run the source model on the training leaks; acceptable
   for M1: true source plus position noise matching the locator's error, ~1 m today).
   Otherwise hazard learns from perfect inputs and fails in the loop.
3. **Source must fill `rate`.** Hazard needs the release rate; without it hazard assumes 1.0
   (the median), losing a factor of up to 2 in concentration.
4. **Probabilities stay probabilities.** Hazard mixes over the source map (top-k cells
   weighted by probability); evac plans on P(harm), never on a thresholded mean.
5. **Harm is one number per cell** *(to fix, integrator)*: with our thresholds flammable
   implies toxic, so `P(harm) = max(P(toxic), P(fire))`, not the independent combination
   the pipeline uses today.

## 5. Evaluation protocol (what the video numbers mean)

- **Per part (M1):** each owner swaps only their part into `pipeline.evaluate(Factory.default(),
  30, seed=0)` and reports all four scores against the all-baseline run.
- **Closed loop (M2):** `pipeline.evaluate(..., locator=TabPFNLocator(...),
  predictor=TabPFNHazardPredictor(...), planner=RiskBudgetPlanner(...))` on seeds 0-99:
  the headline numbers. Also on `Factory.random` plants for generalisation.
- *To build:* batched evaluation (one TabPFN call per part for all episodes instead of one
  per episode). Today 30 episodes cost about 300k tokens per TabPFN part.
- *To build:* a closed-loop test in `tests/test_pipeline.py` that runs all three TabPFN
  classes with the offline `baseline` backend, so a contract break fails CI before merge.

## 6. Who builds what

| Item | Owner |
|---|---|
| `sim.SETTINGS`, seed constants, `max` harm fix, batched `evaluate`, closed-loop test | Steven (integrator) |
| Release-rate quantiles in `SourceEstimate` | Rayan |
| Hazard trained on estimated source and measured wind, training seed >= 1000 | Ahmad |
| Real UCI dataset check (optional) | Whoever finishes M1 first |
