# Status: Rayan (@RayanElKhoury)

Only Rayan's agent edits this file. Update it before every PR.

## Doing

- Waiting for review of PR #25. Next ideas: extend the table past alarm + 30 s; check
  drifting wind and sensor dropouts.

## Done

- **Locator answers at the alarm and every 5 s after it** (PR #25, `source/`), on the
  pipe plant from #23. Clock starts at the first alarm (a real plant does not know when
  the leak started). One row per update, 31 columns for 14 sensors: per sensor the latest
  reading (log1p) and the arrival time relative to the alarm, plus wind u, v and seconds
  since the alarm. Table: 1,200 simulated leaks (seed 12345) x 7 moments (alarm + 0..30 s)
  = 8,393 rows. 24 pipe cells = 24 zones. Before any alarm: the plant's leak prior.
  Dead sensors = missing values.

  Location, 199 held-out leaks (seed 777) x 7 moments, one batched call per method,
  backend `cloud` (TabPFN-3.5), default plant:

  | Seconds since alarm | 0 | 5 | 10 | 20 | 30 |
  |---|---|---|---|---|---|
  | Prior only (likeliest fitting, no sensors): error | 17.1 m | 17.1 | 17.1 | 17.1 | 17.1 |
  | Weighted centroid (baseline): error | 6.2 m | 6.5 | 7.1 | 8.5 | 9.1 |
  | k-NN, same table: error | 2.6 m | 2.3 | 2.4 | 2.5 | 2.7 |
  | **TabPFN: error** | **1.3 m** | **0.7** | **0.7** | **0.7** | **0.7** |
  | TabPFN: exact pipe cell is top guess | 69 % | 76 % | 78 % | 76 % | 78 % |

  Calibration (ECE of the top cell, all moments): TabPFN 0.03, centroid 0.06,
  k-NN 0.30. Mean P(true cell): TabPFN 0.64, centroid 0.12.

- **Leak size (`SourceEstimate.rate`)**: a TabPFN regressor on the same row predicts the
  log release rate; 19 quantiles (5-95 %). Same 199 x 7 held-out rows:

  | Seconds since alarm | 0 | 10 | 20 | 30 |
  |---|---|---|---|---|
  | Prior only (median 1.0, what hazard assumed): mean error | 38 % | 38 % | 38 % | 38 % |
  | k-NN, same table | 36 % | 24 % | 19 % | 18 % |
  | **TabPFN** | **26 %** | **13 %** | **9 %** | **8 %** |
  | TabPFN: true size inside its 80 % interval | 86 % | 88 % | 89 % | 92 % |

- **Latency** (cloud, one live update): map + size in parallel 3.25 s, over the ~2 s
  budget, so `TabPFNLocator(live=True)` now returns the map first: **1.69 s**, with the
  size of the previous update of the same emergency (the size call finished 0.25 s
  later, ready for the next update). Batch of 1,393 rows: 2.06 s.
- `uv run sync-check --source tabpfn_spatial.source:TabPFNLocator` (offline k-NN):
  0 problems, 0 warnings, default and random plant, t_obs 10/20/30/60 s.
- Tokens for these results: 70,000 (7 cloud calls x 10,000). Reproduce:
  `uv run python -m tabpfn_spatial.source.benchmark --backend cloud`.
- Earlier (old 110-site plant, 60 s after leak start): 0.8 m vs centroid 11.4 m. Superseded.
- Finished GitHub setup.

## Needs

- Nothing blocking. Note for Steven: `pipeline.evaluate` / `sync-check` at t_obs 60 s can
  ask for moments beyond alarm + 30 s, which the table does not cover yet (works, but less
  accurate there).

## Blocked

- Nothing.
