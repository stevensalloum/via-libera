# Hazard training with varied spreading

This is a controlled simulation experiment, not validation of a real factory.
The default predictor is unchanged. Passing `diffusivity_range=(low, high)` to
`TabPFNHazardPredictor.fit` samples one log-uniform diffusivity per simulated
training leak. That value generates both the sensor readings and the future
concentration labels. It is **not** a feature given to the prediction model.
The separate random stream keeps leak scenarios and noise paired with the fixed
training context. With no range, the existing fixed setting remains in use.

The range 0.4-1.6 m²/s is an exploratory choice around the simulator's 0.8
default; it is not a calibration from the 3D study. It and the five evaluation
values 0.2, 0.4, 0.8, 1.6 and 3.2 were specified before this run. The first and
last values are outside the range. Prior experiments had already explored some
of these conditions, so the comparison is exploratory, not preregistered.
Here "training" means generating the labelled examples supplied as TabPFN's
context; it does not mean updating the pretrained TabPFN model weights.

For each factory, both TabPFN variants use 100 paired training leaks, 20 cells
and four times (8,000 rows) with source estimates from Rayan's locator. The
source model trains on its own fixed-spread simulated data. Evaluation uses 20
held-out simulated leaks per layout, repeated under all five diffusion values;
both TabPFN versions and the forward baseline receive the same observations
and source estimate in each test case. Sensor readings and evaluation truth
also use the same condition. The contexts, source model and test leaks have
different seeds (1001, 12345 and 20261129). No test outcome chooses the training
range or model. Each factory's context uses that factory's layout; this tests
new leaks on known layouts, not generalisation to an unseen factory without
generating its own context. The two layouts share random episode seeds, so their
results can be correlated.

Forecasts are at 0, 10, 30, 60 and 120 seconds after observation, with
observation times cycling between 10, 20 and 30 seconds after simulated leak
start. An observation before the source alarm uses the source model's pipe
prior. This is not a test of an alarm-triggered scheduling policy. All cells
and times are correlated within a leak. Brier is averaged per case; recall and
false-positive rate pool cell-time counts using a 0.5 probability threshold.
The exploratory paired bootstrap resamples episode indices jointly across the
two fixed layouts and does not quantify uncertainty across new factories.

## Results: cloud TabPFN, 2026-11-22

Each condition has 40 case variations (20 base leaks in each of two layouts).
The five conditions repeat those leaks; there are not 200 independent leaks.
The columns give forward simulation / fixed-context TabPFN / varied-context
TabPFN, on the same cases. Lower Brier and false-positive rate are better;
higher recall is better.

| Test diffusivity m²/s | Toxic Brier | Toxic recall at 0.5 | False-positive rate at 0.5 | Source alarmed |
|---:|---|---|---|---:|
| 0.2 | 0.05724 / 0.04173 / 0.04190 | 0.7695 / 0.6144 / 0.5914 | 0.0394 / 0.0207 / 0.0189 | 27/40 |
| 0.4 | 0.04781 / 0.03977 / 0.04037 | 0.7947 / 0.6643 / 0.6355 | 0.0290 / 0.0177 / 0.0168 | 31/40 |
| 0.8 | 0.04030 / 0.04076 / 0.04183 | 0.8388 / 0.6896 / 0.6617 | 0.0238 / 0.0183 / 0.0175 | 33/40 |
| 1.6 | 0.04065 / 0.04155 / 0.04155 | 0.8365 / 0.7149 / 0.6924 | 0.0228 / 0.0235 / 0.0195 | 36/40 |
| 3.2 | 0.06484 / 0.05020 / 0.04791 | 0.8230 / 0.7272 / 0.7232 | 0.0503 / 0.0441 / 0.0405 | 40/40 |

Varied training did not consistently improve Brier over fixed training, and its
recall was lower at the chosen threshold in every condition. Its false-positive
rate was also lower in every condition. At 3.2, outside the training range,
its Brier improved by 0.00229 relative to fixed training, with an exploratory
paired episode-bootstrap 95% interval of [-0.00396, -0.00069] for varied minus
fixed. The other four intervals included zero. These intervals are conditional
on the two layouts and are not adjusted for five comparisons. In this
simulation, forward simulation had higher recall in every condition, while
the TabPFN methods sometimes had better Brier and fewer false positives.
There is no general winner established, and no reason from this experiment to
enable the varied context by default.

The 200 variants included 167 with an active source alarm and 33 using the
pipe prior. The complete run's guard recorded 882,720 quoted cloud credits:
442,720 for the default factory and 440,000 for random7, including both source
and hazard calls. Forward simulation and offline tests used zero cloud credits;
actual provider billing can differ. The experiment is batched, so it does not
provide live call latency. See the presentation figure and redraw it without
cloud calls:

```bash
uv run python tests/hazard/summarise_diffusivity.py --recorded \
  --figure docs/figures/ahmad/04-diffusivity-study.png
```

### Team heads-up: 33 source estimates were pre-alarm snapshots

This benchmark asked for one forecast at fixed simulated times 10, 20 or 30 s,
even when no measured sensor had alarmed. In 33/200 case variations, Rayan's
locator therefore returned the factory's pipe-location prior and prior rate
quantiles instead of a sensor-derived location and rate. These are repeated
conditions from 13 layout/leak pairs, not 33 independent emergencies. This was
the benchmark's scheduling choice, not proof that the intended system always
starts with a prior. The original benchmark made no later calls, so it did not
measure whether these source estimates subsequently changed.

A follow-up kept those 33 exact initial reading histories and generated *new,
hypothetical* future sensor noise, sampling every 5 s until 120 s after the
simulated leak started. Twenty-six of the 33 variants then alarmed (median 5 s
after the original snapshot; range 5-50 s); seven variations from three
layout/leak pairs did not alarm by 120 s.
Using the cloud source model at each of the 26 later first alarms, median
distance of the top-ranked source changed from 17.0 m using the pipe prior to
1.0 m using the updated model. The top source was closer in 16 cases, farther
in two, and at the same distance in eight. These are small correlated simulated
groups, not measured real incidents; a source update is not guaranteed to
improve. The source calls were quoted at 40,000 credits; no hazard calls were
made. Reproduce the continuation (new output filename required):

```bash
uv run python tests/hazard/prior_diagnostic.py --quote-only --output outputs/prior-check.jsonl
uv run python tests/hazard/prior_diagnostic.py --backend cloud --run-cloud \
  --output outputs/prior-check.jsonl
```

Before the sensor-alarm alignment, the live simulation determined the system
alarm from clean simulated fields (`sim.first_alarm`) but gave Rayan a noisy
sensor history; Rayan independently checked whether the *noisy* readings crossed the alarm
threshold. Close to the cutoff, the system could start while his locator still
returned its prior. The live and crowd evacuation simulations now use the same
noisy history to define the alarm and the source input, as this threshold
experiment did. An explicitly supplied fixed alarm remains a simulation override.

Once an alarm is present in the supplied history, a fresh `locate` call can
produce a sensor-based location. The ordinary batch mode used in these
experiments also waits for the current rate estimate. In Rayan's optional
`live=True` mode the location returns first and the rate initially comes from
the prior; a subsequent call can retrieve a finished background rate for the
same emergency. Updating depends on the caller making a later call, and neither
the next location nor the next rate is guaranteed to be more accurate.

The source alarm threshold above is a sensor-concentration trigger. It is
separate from the hazard-probability threshold used to label a forecast cell
as toxic in our analysis. The existing planner consumes hazard probabilities
directly, so choosing a reporting threshold does not automatically change
route planning or the detector alarm.

To estimate the cost or reproduce the run (cloud mode spends credits):

```bash
uv run python tests/hazard/compare_diffusivity.py --quote-only --episodes 20 --batch 5
uv run python tests/hazard/compare_diffusivity.py --backend cloud --run-cloud \
  --episodes 20 --batch 5 --output outputs/diffusivity-study.jsonl
uv run python tests/hazard/summarise_diffusivity.py outputs/diffusivity-study.jsonl \
  --figure outputs/diffusivity-study.png
```

The output file must not already exist. Per-case aggregate outputs are ignored
by Git. Simulated context and evaluation truth share the same 2D transport
model, so results do not establish physical accuracy, real-world hazard recall,
route safety, or performance under 3D density and ventilation effects. The
cloud benchmark batches multiple cases; it is not a live latency measurement.
