# Sensor-history experiment

Status: **exploratory offline and cloud pilots**. The seven-feature predictor
remains the default; the new options are experimental.

## Question and implementation

Can gas readings and their recent changes improve a hazard forecast beyond the
existing source estimate, wind and spatial features? Three configurations use
the same predictor, the same source-map mixture and the same forecast targets:

| Version | Configuration | Feature count without geometry |
| --- | --- | ---: |
| A | Existing spatial features | 7 |
| B | `include_time_context=True`: A + observation time and forecast horizon | 9 |
| C | `include_sensor_history=True`: B + sensor summaries | 37 |

`include_sensor_history=True` also turns on time context. `include_geometry=True`
adds the existing two geometry features to any configuration. The public
`HazardPredictor` protocol, the default predictor behavior and the existing source
selection remain unchanged. A direct comparison against the previous revision
confirmed identical default context rows, targets and fake-model hazard maps.

The 28 sensor columns are constructed in `sensor_history.py`:

- For each of the three sensors geometrically nearest to the forecast cell:
  distance, latest available value, rate of change per second, value-available
  flag and trend-available flag (15 columns).
- For the two sensors nearest to the candidate source: the same five values
  (10 columns). The source-specific part is recomputed for each hypothesis.
- Maximum and mean available sensor value, and fraction of available sensors
  (3 columns).

Distances use metres. Latest and previous **finite** samples at or before the
observation time give the trend, using their actual time difference. Missing
readings, a missing trend and padded sensor slots use zeros with availability
flags; thus unknown is distinguishable from a valid reading or trend of zero.
Fewer than five sensors, missing values and one-sample histories are supported.
Global values use only available sensors. Equal distances retain sensor order.
Optional forecasts that precede the latest observation raise an error. The
shared contract requires the caller to pass only currently available readings;
the predictor has no separate decision-time argument to identify future samples.

The simulator experiment deliberately assumes the alarm and leak start coincide
at T=0, as requested. `observation_times` is an optional training-data argument;
the pilot uses it to include T=0 alongside later moments without changing legacy
defaults. This forced initial event is not an actual sensor-triggered alarm: the
simulator begins with zero gas, so the first reading contains only sensor noise.
The resulting startup cases are a simplification, not a detector evaluation.

## Five-scenario offline pilot

Run:

```bash
uv run python tests/hazard/compare_sensor_history.py --episodes 5
```

The script makes no cloud calls by default. It generates **one** set of 640
context rows (20 simulated leaks x 8 sampled cells x 4 forecast times, seed 1001)
and obtains A and B as column subsets of C; all versions therefore use exactly
the same underlying examples and targets. Five held-out pipe-plant leaks use
evaluation seed 0, one shared centroid source estimate per leak, and observation
times 0, 5, 10, 15 and 20 s. Forecasts are evaluated at 0, 10, 20 and 30 s after
each observation on open floor; the corresponding 10,300 cell/time outcomes include
836 toxic observations and are correlated within each leak. The centroid did not
apply the pipe prior or supply a rate; its top-three candidates retained 6.66%
mean source probability. All A/B/C models used the offline k-NN regressor, not
TabPFN; k-NN chunk size was 64. Forward is the existing simulator baseline.

| Metric | Forward | A: spatial | B: + time | C: + sensor history |
| --- | ---: | ---: | ---: | ---: |
| Toxic Brier, lower better | 0.05806 | 0.07700 | 0.07679 | 0.06681 |
| Toxic ECE, lower better | 0.05806 | 0.08188 | 0.07096 | 0.05323 |
| Toxic AUC, higher better | 0.75030 | 0.89279 | 0.83502 | 0.75000 |
| Toxic recall at 0.5, higher better | 0.52153 | 0.72847 | 0.47608 | 0.35646 |
| False-positive rate at 0.5, lower better | 0.02092 | 0.10207 | 0.10366 | 0.03339 |
| Precision at 0.5, higher better | 0.68770 | 0.38667 | 0.28861 | 0.48534 |
| Brier within 6 m of a sensor | 0.05761 | 0.08508 | 0.08198 | 0.06494 |
| Brier farther than 6 m | 0.05889 | 0.06197 | 0.06715 | 0.07030 |
| Hazard-call median seconds | 0.0016 | 0.1086 | 0.0889 | 0.1849 |
| Simulated routes exposed, no latency | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| Cloud credits | 0 | 0 | 0 | 0 |

Sensor history reduced Brier versus A on this sample, but its recall and AUC
were lower. It improved Brier near sensors and worsened it farther away relative
to A. The apparent tradeoff is worth investigating, but five leaks cannot prove
any improvement or general ranking. The 0.5 threshold is a preselected test
threshold, not a safety recommendation. Route scores are one-person, immediate
planning replays; they exclude prediction delay and use the shared planner's
provisional toxic/fire combination. For arrival after the 30-second horizon,
the planner uses the nearest available map, i.e. the last forecast. Zero exposed
in every case is inconclusive.
Hazard timings exclude source detection and routing, and do not estimate cloud
latency. The context and evaluation use the same simplified simulator.

The accompanying figure is
![Offline sensor-history pilot](../../../docs/figures/ahmad/01-sensor-history-pilot.png).
Recreate it without model calls:

```bash
uv run python tests/hazard/plot_sensor_history.py
```

## Five-scenario cloud pilots

For five scenarios and 8,000 context rows (100 leaks x 20 cells x four times),
with three source hypotheses, open-cell-only queries and four forecast times,
the free quote on the current default factory was **10,000 credits per call**
for each of A (7 columns), B (9) and C (37): 15 calls and **150,000 quoted
credits total**. The centroid source uses no cloud calls. This quote covers the
intended prediction calls; it is not a guarantee of billing, setup or future
service pricing. Quote again before execution. The model configuration uses the
cloud backend's default version rather than a pinned version.

```bash
uv run python tests/hazard/compare_sensor_history.py --quote-only --episodes 5 \
  --train-scenarios 100 --cells-per-scenario 20
# Run only after approving the quote:
uv run python tests/hazard/compare_sensor_history.py --backend cloud --run-cloud \
  --episodes 5 --train-scenarios 100 --cells-per-scenario 20
```

Both pilots were run on the same five leak scenarios, seed 0, using 8,000 hazard
context rows from 100 simulated leaks, seed 1001. Each A/B/C model in a pilot
used identical context examples, evaluation readings, source estimates and
targets. The forward simulator used the same source estimates. C differs from B
only by the 28 sensor columns; it is the direct test of additional sensor
information. All metrics below are toxic predictions over open cells and four
forecast times 0/10/20/30 s after the observation, with 836 positive outcomes
among 10,300 correlated cell/time outcomes.

| Source and metric | Forward | A: spatial | B: + time | C: + sensors |
| --- | ---: | ---: | ---: | ---: |
| Centroid: Brier (lower better) | 0.05806 | 0.04173 | 0.04006 | 0.03609 |
| Centroid: recall at 0.5 | 0.52153 | 0.46411 | 0.45096 | 0.54306 |
| Centroid: false-positive rate at 0.5 | 0.02092 | 0.01363 | 0.00803 | 0.01131 |
| Centroid: hazard median seconds | 0.00247 | 2.28886 | 2.19408 | 3.12874 |
| Centroid: exposed routes (no latency) | 0/5 | 1/5 | 1/5 | 1/5 |
| Rayan: Brier (lower better) | 0.02786 | 0.03291 | 0.03468 | 0.03499 |
| Rayan: recall at 0.5 | 0.76435 | 0.50000 | 0.49163 | 0.48804 |
| Rayan: false-positive rate at 0.5 | 0.00951 | 0.00475 | 0.00792 | 0.00655 |
| Rayan: hazard median seconds | 0.00239 | 2.39598 | 2.36941 | 2.79001 |
| Rayan: exposed routes (no latency) | 0/5 | 0/5 | 0/5 | 0/5 |

The centroid comparison consumed 150,000 quoted credits (50,000 per cloud
configuration; zero for centroid and forward). The Rayan comparison consumed
190,000 quoted credits: 150,000 for hazard plus 40,000 for the batched source
classifier and rate regressor (two calls each). These guard totals are estimates,
not verified billing. Default cloud model versions were not pinned. Source
context used Rayan's default 1,200 leaks, seed 12345, and a rate model. Of 100
hazard training observations, 63 activated Rayan's locator; 4/5 evaluation
observations did. The others used the pipe prior and prior rate. The first
evaluation case is the artificially assumed alarm at zero concentration.
Mean top-three retained mass was 6.66% with centroid and 78.28% with Rayan.
The training contexts differ between source estimators, so assess B versus C
within each pilot rather than attributing cross-source changes solely to the
source model.

With centroid, C improved Brier and recall versus B, with more false positives.
With Rayan, C's Brier and recall were slightly worse than B, with fewer false
positives; forward had a better Brier and recall than either TabPFN configuration
on this tiny sample. There is no compelling evidence here to enable the added
sensor inputs by default. Five correlated scenarios cannot establish that sensor
history is redundant or prove any ranking. Hazard timings exclude source,
routing and setup; batched source timings do not represent a live call. The
planner's toxic/fire combination is provisional, and replay excludes latency.
The simulator is also the source of both examples and targets.

Reproduce the Rayan-source pilot (cloud calls spend credits):

```bash
uv run python tests/hazard/compare_sensor_history.py --quote-only --source rayan \
  --episodes 5 --train-scenarios 100 --cells-per-scenario 20
uv run python tests/hazard/compare_sensor_history.py --backend cloud --run-cloud \
  --source rayan --episodes 5 --train-scenarios 100 --cells-per-scenario 20
```

![Cloud sensor-history pilots](../../../docs/figures/ahmad/02-sensor-history-cloud.png)

Redraw the recorded aggregates without model calls:

```bash
uv run python tests/hazard/plot_sensor_history_cloud.py
```

The offline 640-row and cloud 8,000-row pilots use different settings and are not
paired comparisons. Future tests need more leaks and layouts, sensor faults,
independent simulation and end-to-end latency. C does not isolate whether local
readings, source-neighborhood readings, global context or trends caused a change.
