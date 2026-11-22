# Status: Ahmad (@Asorus)

Only Ahmad's agent edits this file. Update it before every PR.

## Doing

- Prototype a 3D hazard interface for explicit `(x, y, z)` sources and machine
  geometry. Waiting for Rayan's actual 3D source format and Steven's 3D layout /
  planner contract before connecting it to the integrated application.
- Multi-layout verification (#39): 117 paired case variations from 28 base leaks in
  two layouts completed before the requested time limit; groups are incomplete
  and uneven. Full settings, route baselines and limitations are in
  `hazard/MULTIPLANT_RESULTS.md`.
- Sensor-history ablation (#31): optional features and exploratory pilots were merged in
  PR #32; results do not justify changing the seven-feature default.
- Review the merged hazard predictor (#19), source selection and geometry with
  Ahmad. Steven's pipe workflow and Rayan's source locator are now on main; the
  existing comparisons are preliminary simulation results.

## Done

- Experimental transient 3D hazard volume and forward baseline: explicit 3D
  source hypotheses, voxel walls, axis-aligned 3D machines and hot zones; output
  axes `(time, z, y, x)`. Small paired cloud benchmark, one synthetic layout,
  12 test leaks seed 20261211, 80 training leaks seed 20261210, 4,800 context
  rows, three times, known true 3D source/rate/wind, no source estimator: mean
  toxic Brier forward / TabPFN 0.00994 / 0.01414, recall at 0.5 91.34% /
  76.91%, false-positive rate 0.57% / 0.87%; cloud quote 120,000 credits vs 0.
  Same simulator family for training, baseline and truth; 3D units are relative,
  with no HVAC or buoyancy. This is not real-world or integrated validation.
  One repeated cloud case after bounded query batching reproduced its Brier and
  counts for an additional quoted 10,000 credits.
  Details: `hazard/VOLUME3D.md`.
- Partial multi-layout cloud study, evaluation seeds 20261121/20261122/20261123,
  hazard context seed 1001, 8,000 rows, Rayan source seed 12345, top 3, open
  cells, five forecast times. Forward / centroid-context / Rayan-context toxic
  Brier: standard 0.02719 / 0.05825 / 0.03628 (28 cases); changing wind 0.04107 /
  0.05988 / 0.04080 (25); sensor faults 0.04664 / 0.06014 / 0.04151 (25);
  slower mixing 0.03320 / 0.05359 / 0.03755 (19); faster mixing 0.03200 /
  0.04492 / 0.02745 (10); combined stress 0.04917 / 0.05001 / 0.03677 (10).
  Pooled recall at 0.5 in the same order: 0.8990 / 0.5642 / 0.7196; 0.8492 /
  0.5663 / 0.7095; 0.8117 / 0.5230 / 0.6706; 0.8718 / 0.5785 / 0.6792;
  0.8810 / 0.7630 / 0.8779; 0.7739 / 0.6717 / 0.7440. At least 1,070,000
  credits were quoted for completed cloud calls including pilot and batch checks;
  additional interrupted calls and actual provider billing are unknown. Forward
  and offline tests use zero. Unequal conditions and repeated scenarios mean no
  general ranking; source and target both use the same simulator.
- Cloud sensor-history pilots, five simulated pipe leaks, evaluation seed 0,
  8,000 context rows seed 1001, forecasts 0/10/20/30 s after observation, top 3:
  centroid source forward / A spatial / B time / C sensors Brier 0.05806 /
  0.04173 / 0.04006 / 0.03609 and recall 0.52153 / 0.46411 / 0.45096 / 0.54306;
  Rayan source and rate Brier 0.02786 / 0.03291 / 0.03468 / 0.03499 and recall
  0.76435 / 0.50000 / 0.49163 / 0.48804. False-positive rates centroid 0.02092 /
  0.01363 / 0.00803 / 0.01131, Rayan 0.00951 / 0.00475 / 0.00792 / 0.00655.
  Hazard median seconds centroid 0.00247 / 2.28886 / 2.19408 / 3.12874; Rayan
  0.00239 / 2.39598 / 2.36941 / 2.79001. Immediate replay exposed routes
  centroid 0/5 / 1/5 / 1/5 / 1/5, Rayan 0/5 for all. Quoted credits centroid
  0 / 50,000 / 50,000 / 50,000; Rayan 0 / 50,000 / 50,000 / 50,000 plus
  40,000 shared source calls. Rayan source activated for 4/5 evaluation cases;
  the first uses a pipe prior. The five cases cannot prove redundancy or
  establish safety. Guard amounts are quotes; timings omit source and routing.
- Sensor-history offline pilot: five pipe-plant leaks, seed 0, 640 shared context
  rows seed 1001, centroid source, simulated alarm at leak start, open floor,
  forecast horizons 0/10/20/30 s. Forward vs offline k-NN spatial A / time B /
  sensor C toxic Brier 0.05806 / 0.07700 / 0.07679 / 0.06681, recall at 0.5
  0.52153 / 0.72847 / 0.47608 / 0.35646, false-positive rate 0.02092 / 0.10207 /
  0.10366 / 0.03339; median hazard seconds 0.0016 / 0.1086 / 0.0889 / 0.1849.
  All route replays had 0/5 exposed. All used 0 cloud credits; outcomes are
  correlated and this small pilot cannot establish a general result. A/B/C used
  exactly the same generated context rows and evaluation observations.
- Added `hazard/RESULTS_GUIDE.md` and a credit-free plotting script to explain
  the recorded experiments and reproduce nine presentation figures. The plots
  use previously documented aggregate results; they are not new measurements.
- In a temporary checkout of Steven's PR #28 stack (head `d124666`), ran the
  `sync-check` CLI with `--hazard tabpfn_spatial.hazard:TabPFNHazardPredictor`,
  offline backend, default centroid source: 24/24 cases passed hard checks across
  two plants, four decision times and three seeds; forward baseline is the CLI's
  reference and does not use cloud credits. The two warnings are that the centroid
  source does not keep 99% probability on pipes and supplies no rate. The offline
  k-NN query chunk was reduced to 128 for memory; its timing is not a cloud test.
- Open-cell query pilot: five paired early-alarm pipe leaks, seed 0, same centroid
  source, 8,000 context rows seed 1001. Full-grid vs open-cell cloud Brier 0.04198
  vs 0.04198, source + hazard + one-plan median 3.6796 vs 3.4496 s; forward
  Brier 0.05395 and median 0.0082 s. Each cloud option exposed 1/5 people, as did
  forward and nearest exit. Largest cell probability difference 6e-17; full vs
  open queries 23,400 vs 20,085 rows. Credits 50,000 vs 50,000; forward 0. The
  five-case sample is too small to establish speed or accuracy benefits.
- Repeated the same 30 pipe-leak cases with a first-route hybrid replay: baseline
  vs cloud toxic Brier 0.07186 vs 0.05731 overall and 0.06239 vs 0.04352 for the
  first 30 s. Source + hazard + one-plan median 0.0066 vs 3.9187 s, p95 0.0091 vs
  4.9136 s. Exposed: nearest 4/30, baseline 4/30, cloud 2/30, hybrid 1/30; mean
  harmful seconds 0.87, 1.17, 0.47 and 0.33 respectively. Model calls start from
  the same readings; hybrid switches from nearest to baseline to the cloud route.
  This is simulated replay, not an integrated live feature. Paired Brier, latency
  and exposure baselines are above; repeat used 300,000 cloud credits vs 0 for
  baseline. A separate 10-case hybrid pilot used 100,000; all four policies had
  0/10 exposed, so it did not distinguish routes. Details and limitations below.
- Early-alarm pipe comparison: 30 paired default-plant leaks, evaluation seed 0,
  simulated detector median alarm 10 s, centroid source for both, hazard context
  seed 1001, 8,000 rows, noisy wind, sampled estimated sources, top 3 candidates,
  open-floor toxic outcomes over 0-120 s after alarm. Forward baseline vs cloud
  TabPFN toxic Brier 0.07186 vs 0.05731, ECE 0.07186 vs 0.00645, AUC 0.79755 vs
  0.94516. Measured source + hazard + one-plan median 0.0100 vs 3.8503 s, p95
  0.0172 vs 5.3120 s. One-person route replay: 4/30 vs 3/30 people exposed,
  mean harmful seconds 1.17 vs 0.50; nearest exit 4/30 and 0.87 s. Delays
  rounded up to whole seconds. Forward uses 0 credits, cloud 300,000; an earlier
  three-case pilot used another 30,000. Timing excludes model setup and uses the
  centroid, not Rayan's locator. The centroid does not apply the pipe prior.
  One simulated plant and correlated cells/times; no real-world claim. Tested in
  a temporary copy of Steven's unmerged PR #28 stack (head `d124666`).
- Cache/Fast experiment: 5 default-factory cases, seed 0, source maps from Rayan,
  8,000 context rows seed 1001, top 3, open-floor toxic outcomes. Forward baseline
  Brier 0.04405; v3.5 / v3.5 cached / v3.5-Fast / Fast cached Brier 0.07029 /
  0.07034 / 0.06827 / 0.06825, median hazard seconds 3.65 / 4.68 / 2.74 / 4.32,
  credits per forecast 10,000 / 30,000 / 10,000 / 30,000 (forward 0 credits).
  Cached forecasts needed three requests due to the server's 10,000-row limit.
  Successful comparison cost 400,000 credits; five cases and variable setup time
  cannot establish a general latency or quality ranking. Cache and non-cache
  used different batching, so observed output differences cannot be attributed.
- Paired geometry experiment, 30 default-factory episodes, evaluation seed 0,
  Rayan source branch `331bbe3` (context seed 12345), hazard context seed 1001,
  8,000 rows, top 3, open-floor toxic outcomes, cloud TabPFN: forward baseline
  vs original vs optional path features Brier 0.04638 / 0.05832 / 0.05600; ECE
  0.04638 / 0.03397 / 0.03450; AUC 0.91150 / 0.95749 / 0.96200; provisional mean
  exposed 4.63 / 4.20 / 4.27 vs standard plan 7.03. Hazard calls cost 0 /
  300,000 / 300,000 credits; source-map batch cost an additional 10,000 vs 0
  for weighted centroid. Path minus original Brier -0.00233, exploratory paired
  bootstrap 95% interval -0.00413 to -0.00077. Same simulator and one factory;
  model version not pinned and route risk combination remains provisional.
- Source-selection comparison on the same 30 scenarios: centroid vs Rayan locator
  top-3 retained mass 7.06% vs 91.34% and true-source coverage 1/30 vs 28/30.
  With Rayan's maps, forward top 1 vs TabPFN top 3 / top 10 / adaptive 90%
  yielded toxic Brier 0.04638 vs 0.05832 / 0.05830 / 0.05841. The three TabPFN
  variants shared one top-10 batch per episode, 557,580 quoted credits total
  vs 0 for forward; adaptive cap 10 bound in 0/30 cases. Exploratory paired
  intervals for selection differences include zero. Results are simulation-only.
- Paired backend comparison: 5 default-factory episodes, evaluation seed 0,
  context seed 12345, 8,000 context rows, top 3 source cells, weighted-centroid
  locator, open-floor toxic outcomes. Forward simulation vs offline k-NN vs cloud
  TabPFN: Brier 0.09081 / 0.10857 / 0.08850; ECE 0.09081 / 0.09152 / 0.08216;
  AUC 0.75245 / 0.80376 / 0.80814; cloud credits 0 / 0 / 50,000. Hazard call
  times were 0.01-0.02 / 4.16-4.45 / 3.49-4.98 s respectively; offline k-NN
  used a smaller query chunk for memory. Small simulated sample, cloud version
  and fit mode not explicitly selected; not a general performance claim.
- Offline diagnostics, 0 cloud credits for both conditions: a constructed pair of
  14 m x 10 m layouts produced identical target features but simulator concentration
  0.13408 vs 0.00882 at 120 s (near vs far door, threshold 0.1). Separately, on 30
  default-factory episodes, seed 0, toxic Brier on open floor was 0.09288 vs
  0.07345 for forward-simulation mixtures using top 3 vs 90% retained source mass
  (3 vs 83.4 mean candidate cells). The latter won 28/30 vs 2/30 episodes. Neither
  diagnostic evaluates TabPFN or validates the simulator against real data.
- Offline source-map diagnostic, default factory, 30 episodes, seed 0,
  weighted-centroid baseline, 60-second readings and 0 cloud credits: top 3 cells
  retained 7.06% mean probability vs at least 90% for an adaptive selection;
  the true simulated source was included in 1/30 vs 22/30 cases. The 90% selection
  used 83.4 cells on average vs 3. This is a source-map diagnostic, not a hazard
  benchmark; Rayan's TabPFN source locator was not evaluated.
- Paired held-out benchmark, default factory, 3 episodes, evaluation seed 0,
  training seed 42, cloud backend, 100 simulated training leaks, 20 sampled cells,
  4 forecast times, top 3 source cells: harm Brier 0.05663 vs forward-simulation
  baseline 0.05907; toxic Brier 0.05639 vs 0.05907; toxic ECE 0.05637 vs 0.05907;
  toxic AUC 0.85534 vs 0.79890. The cloud calls used 30,000 credits vs 0 for the
  offline baseline (quoted and consistent with account usage). These are the same
  episode scores used by `pipeline.evaluate`, calculated on paired episodes.

## Needs

- Rayan/Steven: agree on a 3D source and hazard contract and supply explicit
  source heights, 3D wall/machine geometry and ignition metadata. Do not infer
  heights from the existing 2D map; validate 3D physics and units separately.
- Expand the Rayan-source sensor-history pilot beyond five simulated cases and
  one plant before drawing a general conclusion about its additional value.
- Validate early-alarm accuracy and route outcomes with independent simulation,
  drift, sensor faults, varied plants and unknown real leak-start times; measure
  actual source + hazard + route latency with the selected deployment backend.
- Review whether to keep the optional source-mass and path features after looking
  at their quality, runtime and limitations; defaults have not been changed.
- Earlier centroid-based comparisons used the rate fallback 1.0; the new
  Rayan-source pilot used his estimated rate quantiles.
- Pipeline scoring combines toxic and fire as independent probabilities, although
  fire is a subset of toxic under the current thresholds; this can overstate harm.

## Blocked

- None.
