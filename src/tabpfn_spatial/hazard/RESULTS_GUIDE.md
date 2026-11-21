# Hazard proof of concept: results and presentation guide

This guide brings together the experiments we recorded for the hazard predictor.
It is intended for explaining the work and choosing figures for a presentation.
The detailed methods, additional measurements, caveats and reproduction notes are
in [PROOF_OF_CONCEPT.md](PROOF_OF_CONCEPT.md); the raw per-case results were not
retained for every experiment. The figures reproduce the recorded **aggregate**
numbers and do not add new experiments or uncertainty estimates.

**Scope:** all forecast and route results here use synthetic leaks in a simplified
factory simulator. They do not establish real-factory accuracy, calibrated
probabilities, safety, or a statistically reliable improvement. Much of the data
used to prepare TabPFN and all the test outcomes came from the same simulator the
forward baseline uses. Most comparisons use one factory and a small number of
leak scenarios; many scored cells and times come from each scenario and are not
independent observations.

## Getting the figures

From the repository root, run:

```bash
uv run python -m tabpfn_spatial.hazard.plot_results
```

This reads the rounded aggregate values recorded in the plotting script and
writes nine figures to `outputs/hazard_results/`, each as a high-resolution PNG
and an editable, scalable SVG. **It does not run TabPFN or spend cloud credits.**
PNG is convenient for slides; SVG stays sharp when resized. The `outputs/` folder
is ignored by Git, so figures must be generated locally or shared separately;
the source for recreating them is [plot_results.py](plot_results.py).

Each figure's footer identifies its experimental setup. Keep that footer or put
the same context on the slide. The first three figures use the **early-alarm pipe
setup**; the later figures use an **older 60-second observation setup** unless
they say otherwise. Their scores should not be treated as one directly paired
experiment.

## The measures, in plain language

- **Toxic Brier score:** average squared error between the predicted toxic
  probability and the simulated 0/1 outcome for each cell and time. Lower is
  better. A lower score does not by itself establish calibrated probabilities.
- **Toxic AUC:** how often hazardous observations receive a higher score than
  non-hazardous observations, accounting for ties. Higher is better. AUC does
  not say how many hazards an alarm threshold will actually detect.
- **ECE:** a bin-based summary of the difference between predicted probabilities
  and observed frequencies. Lower is better, but a low sample ECE alone does not
  establish calibration.
- **Exposure:** the number of simulated people who enter a truly hazardous cell,
  or the time they spend there. These are simulated route outcomes, not injuries.
- **Median / p95:** the middle measured time and the value at or below which
  95% of that sample's measured times fall. Neither guarantees future timing.

## Early alarms and pipe leaks

These experiments used a temporary copy of Steven's proposed PR #28 stack at
`d124666`; that stack was not merged into the hazard branch when tested. Leaks
started on the proposed pipe network. The alarm was triggered by the clean
simulated detector and occurred at a median of 10 seconds in the 30-case run;
the model readings were noisy. Both hazard methods received the **same
weighted-centroid source estimate**, which did not apply the pipe prior and was
not Rayan's source model. The hazard context used seed 1001, 100 scenarios,
20 cells per scenario, four forecast times per scenario, estimated source
hypotheses and measured wind. Evaluation used seed 0 and the open-floor mask.

### Forecast quality

![Early alarm hazard forecast quality](../../../outputs/hazard_results/01_early_forecast_quality.png)

| Paired toxic score, 30 cases | Forward baseline | Cloud TabPFN |
| --- | ---: | ---: |
| Brier, all 0-120 s after alarm | 0.07186 | 0.05731 |
| Brier, first 30 s after alarm | 0.06239 | 0.04352 |
| ECE, all forecast times | 0.07186 | 0.00645 |
| AUC, all forecast times | 0.79755 | 0.94516 |

In this particular sample TabPFN scored better on these measures. The earlier
60-second experiment below produced a different Brier ordering, so this is not a
general claim about which method always wins. A 30-case cloud comparison used
300,000 quoted credits, versus zero for the forward baseline. A repeat on the
same scenarios used another 300,000; its forecast scores were identical.

### Time to a first route

![Early alarm route timing](../../../outputs/hazard_results/02_early_route_timing.png)

| Source + hazard + one route plan | Forward baseline | Cloud TabPFN |
| --- | ---: | ---: |
| Median | 0.0066 s | 3.9187 s |
| p95 | 0.0091 s | 4.9136 s |

The two-second line is the target suggested by Steven's **simulated timing
study**, not a universal safety threshold. The measurements include the centroid
source, hazard call and one route-planning call; they exclude model preparation
and do not include Rayan's source-model latency. An earlier repeat measured
forward vs TabPFN medians of 0.0100 vs 3.8503 s and p95s of 0.0172 vs 5.3120 s.

### Route replay

![Early alarm route exposure](../../../outputs/hazard_results/03_early_route_exposure.png)

| Route policy, one simulated person in each of 30 leaks | Exposed | Mean harmful seconds |
| --- | ---: | ---: |
| Nearest exit | 4 / 30 | 0.87 |
| Forward hazard route | 4 / 30 | 1.17 |
| TabPFN hazard route | 2 / 30 | 0.47 |
| Baseline route, then TabPFN update | 1 / 30 | 0.33 |

These are **first-route replays**, not a deployed real-time system. At the alarm
the person starts toward the nearest exit; a model-informed route is applied
after the measured delay rounded up to whole seconds. The hybrid assumes the
baseline and cloud calls start from the same readings and switches routes when
each arrives. It is an experiment in the hazard test area, not a modification
to Steven's live planner. In the earlier repeat, TabPFN exposed 3/30 with 0.50
mean harmful seconds; measured latency varied. Small counts cannot establish a
reliable difference. The shared planner's toxic/fire combination is provisional.

### Other early-alarm checks

- **Open-cell query, five paired leaks:** forwarding only open cells reduced
  query rows from 23,400 to 20,085. Full vs open-only cloud Brier was 0.04198 vs
  0.04198; median source + hazard + one-plan time was 3.6796 vs 3.4496 s.
  The forward baseline scored 0.05395 and took 0.0082 s. Each cloud variant,
  forward and nearest exit exposed 1/5 people. The maximum per-cell probability
  difference between cloud variants was about 6e-17. Each variant used 50,000
  credits versus zero for the baseline. Five cases cannot establish a dependable
  speed improvement; neither cloud median met the proposed two-second target.
- **Hybrid pilot, ten different leaks:** nearest, forward, TabPFN and hybrid all
  had 0/10 exposed, so the pilot did not distinguish them. The TabPFN forecasts
  used 100,000 credits; the forward baseline used zero.
- **Earlier three-case pilot:** before the context was expanded to cover 5-second
  decisions, forward vs TabPFN Brier was 0.08414 vs 0.09210 and median measured
  source + hazard + one-plan time was 0.0122 vs 3.9230 s. It used 30,000 cloud
  credits versus zero for forward. This used a prior context configuration.
- **Integration:** offline `sync-check` passed 24/24 cases on two proposed plants
  at decision times 10, 20, 30 and 60 seconds, three seeds each. The centroid
  source had two warnings: mass outside pipe sites and no release rate. The k-NN
  query chunk was reduced to 128 for memory. The check validates interface
  behavior, not cloud response time or prediction quality.

The early-alarm cloud runs above used 830,000 quoted credits in total: two
30-case comparisons, the ten-case and three-case pilots, and the five-case
two-variant open-cell comparison. The underlying 30 cases were repeated, not
60 independent cases.

## Earlier 60-second observation experiments

These used the original default factory without the new pipe workflow, observed
readings through 60 seconds and evaluated future times 70-180 seconds. Unless
otherwise stated, the scores use open-floor cells. Evaluation seed was 0; the
context and source seeds are specified below. They used the earlier hazard
context design with the true simulated source, wind and rate in training.

### Geometry feature comparison

![Earlier geometry comparison](../../../outputs/hazard_results/04_previous_geometry.png)

On 30 paired cases, the same Rayan source-model snapshot (`331bbe3`) was supplied
to each hazard approach. Hazard context seed 1001 supplied 8,000 rows; the source
model's context seed was 12345. The original and optional path-feature TabPFN
models each used 300,000 credits; forward used zero. The source maps had been
created in one cloud batch for an additional 10,000 credits.

| Metric | Forward baseline | Original TabPFN | TabPFN + path |
| --- | ---: | ---: | ---: |
| Toxic Brier | 0.04638 | 0.05832 | 0.05600 |
| Toxic ECE | 0.04638 | 0.03397 | 0.03450 |
| Toxic AUC | 0.91150 | 0.95749 | 0.96200 |
| Mean people exposed, provisional | 4.63 | 4.20 | 4.27 |
| Hazard-call median / p95 | 0.015 / 0.025 s | 3.75 / 5.39 s | 3.65 / 5.05 s |

The zero-hazard standard plan exposed 7.03 people on average. The path feature
improved Brier compared with the original TabPFN in this sample, but both had
worse Brier than the forward baseline. The paired path-minus-original mean Brier
difference was -0.00233, with an exploratory 95% episode-bootstrap interval of
[-0.00413, -0.00077]. These interval values are an exploratory result, not a
real-world performance guarantee. Path length represents grid connectivity; it
is not a model of airflow.

### Source-map diagnostics

![Earlier source map diagnostics](../../../outputs/hazard_results/05_source_map_diagnostics.png)

| 30 simulated cases | Weighted centroid | Rayan snapshot |
| --- | ---: | ---: |
| Mean source probability in top 3 cells | 7.06% | 91.34% |
| True source in top 3 | 1 / 30 | 28 / 30 |
| Mean cells needed to cover at least 90% | 83.4 | 3.0 |
| True source in 90% selection | 22 / 30 | 30 / 30 |

The centroid required no cloud credits; Rayan's 30 maps were generated in one
10,000-credit batch. Rayan's top-three retained mass was below 90% in 9/30
cases. These are diagnostics of this sample, not proof that either probability
map is calibrated or that the same source accuracy applies to pipe leaks.

### Source-selection comparisons

![Earlier source selection comparisons](../../../outputs/hazard_results/06_source_selection.png)

There were **two separate studies** in this figure:

1. With Rayan's source maps, forward top one vs TabPFN top three, top ten and
   adaptive 90% selection had toxic Brier 0.04638 vs 0.05832, 0.05830 and
   0.05841. The TabPFN selections reused the same top-ten prediction batch;
   that batch used 557,580 quoted credits total, versus zero for forward.
   The top-ten and adaptive differences relative to top three had exploratory
   episode-bootstrap intervals including zero. The top-ten batch's median
   hazard call was 8.36 s, with a 7.87-34.15 s range; it is not a timing for
   separate top-three calls.
2. In a different offline diagnostic using **centroid maps and forward-simulator
   mixtures only**, top three vs a 90% retained-mass selection scored 0.09288 vs
   0.07345 Brier. The 90% selection used 83.4 sources on average versus 3 and
   scored better in 28/30 cases. Neither variant used cloud credits. This is not
   an evaluation of TabPFN or Rayan's locator.

### Offline and cloud model pilot

![Earlier offline and cloud backend comparison](../../../outputs/hazard_results/07_backend_pilot.png)

On five paired cases using centroid source estimates, the forward simulator,
offline k-nearest-neighbour backend and cloud TabPFN had Brier scores of 0.09081,
0.10857 and 0.08850; ECE 0.09081, 0.09152 and 0.08216; and AUC 0.75245,
0.80376 and 0.80814. The two learned-model backends used 8,000 context rows,
seed 12345, and top three source cells. The observed hazard-call ranges were
0.01-0.02, 4.16-4.45 and 3.49-4.98 seconds. The offline k-NN chunk was reduced
to 128 to limit memory. Cloud used 50,000 credits; the two offline approaches
used zero. Five cases cannot establish a general ranking.

### Cache and Fast model pilot

![Earlier cache and Fast model comparison](../../../outputs/hazard_results/08_cache_fast_pilot.png)

Five cases used Rayan's older source maps and 8,000 context rows, seed 1001.
The forward baseline Brier was 0.04405 at zero cloud credits.

| Cloud configuration | Toxic Brier | Median hazard call | Credits / forecast |
| --- | ---: | ---: | ---: |
| v3.5, preprocessing | 0.07029 | 3.65 s | 10,000 |
| v3.5, cache, 3 chunks | 0.07034 | 4.68 s | 30,000 |
| v3.5-Fast, preprocessing | 0.06827 | 2.74 s | 10,000 |
| v3.5-Fast, cache, 3 chunks | 0.06825 | 4.32 s | 30,000 |

The successful comparison used 400,000 credits total. The prior unsplit attempt
completed ten non-cached calls for another 100,000 credits; cached calls failed
because the request exceeded the service's 10,000-row cached limit. Successful
cached forecasts were split into three requests. Since both caching and batch
splitting changed, differences cannot be attributed to caching alone. Setup time
was variable, and no result establishes a real-time guarantee.

### Constructed geometry counterexample

![Constructed geometry counterexample](../../../outputs/hazard_results/09_geometry_counterexample.png)

Two constructed layouts produced the same original seven-feature row for one
target cell, but the simulated concentrations at 120 seconds were 0.13408 and
0.00882, versus a relative toxic threshold of 0.1. The doors were at different
locations. This is an example of information missing from those features, not a
measurement of how often the issue occurs and not a fitted TabPFN test.

### First three-case benchmark

The original default-factory comparison used three paired scenarios, context seed
42 and evaluation seed 0. It included wall cells. Forward vs cloud TabPFN had
harm Brier 0.05907 vs 0.05663, toxic Brier 0.05907 vs 0.05639, toxic ECE 0.05907
vs 0.05637, and toxic AUC 0.79890 vs 0.85534. It used zero vs 30,000 cloud credits.
Three cases are too few to establish a reliable advantage; the combined-harm
calculation is provisional.

## Explaining the result responsibly

A suitable presentation statement is:

> We built a simulation-based proof of concept that uses a leak-source estimate
> to predict a factory hazard map. In a small early-alarm pipe-leak experiment,
> TabPFN improved the recorded hazard scores, while the forward baseline was much
> faster. We are testing the full workflow and plan independent simulation and
> real-data evaluation.

Avoid saying the model is validated for real emergencies, that the probabilities
are calibrated, or that the hybrid route has been deployed. The next comparison
should use the merged pipe workflow, Rayan's updated early-time source model,
measured end-to-end latency, varied plants and an independent simulator or real
measurements. The true leak start is known in these simulations but generally
unknown in a real installation.
