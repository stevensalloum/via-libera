# Status: Rayan (@RayanElKhoury)

Only Rayan's agent edits this file. Update it before every PR.

## Doing

- PR `feat: TabPFN leak source locator` (#18, M1) in review. Next: optional release-rate
  regressor (`SourceEstimate.rate`), and check how TabPFN's map degrades with shifting
  wind / sensor dropouts once simulator v1 lands.

## Done

- **M1: `TabPFNLocator` beats the baseline** (`src/tabpfn_spatial/source/`).
  Simulation-based inference: 3,000 seeded simulated leaks (train seed 12345), 58 features
  (log1p readings at 20/40/60 s, arrival times, wind), 110 leak sites merged into 100
  zones, TabPFN classifier -> zone probabilities -> grid map. Backend `cloud` (TabPFN-3.5).

  `pipeline.evaluate(Factory.default(), 30, seed=0)`, baseline vs `locator=TabPFNLocator()`:

  | Metric | Baseline | TabPFN |
  |---|---|---|
  | `source_error_m` | 9.41 m (weighted centroid) | **1.03 m** |
  | `hazard_brier` (downstream, unchanged forward-sim predictor) | 0.089 | **0.040** |
  | `exposed_plan` of 30 (downstream; standard plan 7.03) | 5.40 | **4.63** |

  Held-out leaks (200 leaks, test seed 777, same library; one batched cloud call):

  | Metric | Weighted centroid | k-NN, same library | TabPFN |
  |---|---|---|---|
  | `error_m` (lower better) | 11.41 | 4.99 | **0.83** |
  | P(true zone) (higher better) | 0.037 | 0.023 | **0.637** |
  | top-1 zone accuracy | 0.060 | 0.215 | **0.805** |
  | top-3 zone accuracy | 0.170 | 0.485 | **0.980** |
  | ECE of top zone (lower better) | 0.046 | 0.178 | 0.121 |

  Tokens: 310,000 for the whole run (1 call x 10,000 for the 200 held-out leaks + 30 calls
  x 10,000 for `evaluate`, which locates one episode per call). Reproduce:
  `uv run python -m tabpfn_spatial.source.benchmark --backend cloud --evaluate 30`.
  Caveat: ECE 0.12 means TabPFN's top-zone confidence is somewhat off; the centroid's low
  ECE only reflects that it is uniformly unconfident.
- Finished GitHub setup (repo cloned, uv sync, ruff and pytest pass).

## Needs

- Steven (pipeline, integrator): a batched `evaluate` path (e.g. call `locate_many` when
  the locator has it) would cut `evaluate` from 30 calls to 1. Not blocking.

## Blocked

- Nothing.
