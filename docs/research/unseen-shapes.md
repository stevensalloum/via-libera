# Unseen factory shapes: do the planners and the team's models hold up?

Steven, 2026-11-22. Command: `uv run evac-shapes` (`src/tabpfn_spatial/evac/shapes_study.py`).
Figures: `docs/figures/shapes/`.

## What was tested

- **24 plants nobody had seen:** 6 each of L-, U-, T-shaped halls and a campus of two halls
  joined by a corridor with offices. Seeds 100-105, while the templates use seeds 1-4. Every
  plant has its own outline, interior walls, machines, pipes, detectors (on walls, near
  the pipes), exits and AGV lanes (`plants.png`).
- **Crowd:** one worker per 20 m^2 of floor (179-304 people). The detector alarm starts the
  evacuation; plans are redone every 15 s.
- **AGVs:** 4 per plant, all emergency-stop at the alarm and block their cell.
- **Every planner on the same leak, alarm, model maps and people:** nearest exit, risk-aware,
  CCRP (capacity-aware).
- **Three model settings:**

  | setting | what it is | leaks |
  |---|---|---|
  | perfect | true leak and true future: isolates the planning | 4 per plant, 96 |
  | baseline | centroid locator + forward simulation | 4 per plant, 96 |
  | team | Rayan's locator + Ahmad's hazard predictor on cloud TabPFN, fitted on simulated data of each new plant | 1 per plant, 24 |

## Results (`gains.png`, `scores.png`, `locator.png`)

| models | planner | s in gas / person | vs nearest (paired) | exposed % | 95 % out (s) | runs |
|---|---|---|---|---|---|---|
| perfect | nearest | 0.57 | - | 5.6 | 47 | 96 |
| perfect | risk | 0.13 | -0.45 +/- 0.07 | 1.9 | 51 | 96 |
| perfect | ccrp | 0.13 | -0.44 +/- 0.07 | 2.0 | 46 | 96 |
| baseline | nearest | 0.57 | - | 5.6 | 47 | 96 |
| baseline | risk | 0.24 | -0.33 +/- 0.07 | 3.1 | 51 | 96 |
| baseline | ccrp | 0.29 | -0.29 +/- 0.07 | 3.5 | 46 | 96 |
| **team** | nearest | 0.73 | - | 6.8 | 44 | 24 |
| **team** | risk | **0.09** | **-0.64 +/- 0.21** | 2.0 | 51 | 24 |
| **team** | ccrp | **0.08** | **-0.65 +/- 0.21** | 2.0 | 44 | 24 |

No one was left inside in any run.

1. **The gas-aware planners transfer to unseen shapes.** On every shape and with every
   model setting they beat nearest exit. With the team's models they cut the time in gas
   per person by about 88 % (0.73 -> 0.08 s) and the share of people exposed from 6.8 % to
   2.0 %.
2. **CCRP is the one to use:** as safe as risk-aware, and as fast as nearest exit (95 %
   out at 44 s). Risk-aware detours cost 4-9 s (most on the campus, where the corridor
   is a bottleneck).
3. **The team's models are good enough for planning on new plants:** with them, the
   planners do as well as with the perfect models (0.08 vs 0.13 s). The locator's median
   error is 0 m (the exact cell) from the first update. The baseline locator is ~10 m off.
4. **T-plants are where it matters most** (nearest exit: 1.44 s in gas, 10 % exposed): the
   stem funnels people past the leak. The planners bring it down to 0.10 s and 3 %.

## Limits

- **Few runs with the team's models:** 24, one leak per plant, because every model update
  is two cloud calls. Their error bars are about three times wider than the others.
- **Small numbers in absolute terms:** the dense, pipe-side detectors raise the alarm at
  5 s (median), so the leaks are caught early. Larger leaks or sparser detectors would
  give bigger exposures.
- **Gas is 2D;** the height zone is not used by the planners yet (next step).
- **Cost:** about 11M cloud tokens for this run, plus 2.8M lost in a first attempt. That
  attempt crashed on the cloud's limit of 60 predict requests per minute. Every TabPFN call
  now retries on that limit (`core.models.with_retry`), and the study saves each run as it
  finishes and resumes without repeating any.
