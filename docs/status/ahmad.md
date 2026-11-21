# Status: Ahmad (@Asorus)

Only Ahmad's agent edits this file. Update it before every PR.

## Doing

- Hazard predictor (#19): source-weighted concentration forecasts; larger cloud
  benchmark awaiting approval. Results below are preliminary, not proof of a win.

## Done

- Paired held-out benchmark, default factory, 3 episodes, evaluation seed 0,
  training seed 42, cloud backend, 100 simulated training leaks, 20 sampled cells,
  4 forecast times, top 3 source cells: harm Brier 0.05663 vs forward-simulation
  baseline 0.05907; toxic Brier 0.05639 vs 0.05907; toxic ECE 0.05637 vs 0.05907;
  toxic AUC 0.85534 vs 0.79890. The cloud calls used 30,000 credits vs 0 for the
  offline baseline (quoted and consistent with account usage). These are the same
  episode scores used by `pipeline.evaluate`, calculated on paired episodes.

## Needs

- Approval for the 30-episode cloud benchmark (estimated 300,000 credits vs 0 for
  the baseline) to verify the comparison on the brief's benchmark size.
- Pipeline scoring combines toxic and fire as independent probabilities, although
  fire is a subset of toxic under the current thresholds; this can overstate harm.

## Blocked

- Automatic review blocked the larger cloud run because of its credit cost.
