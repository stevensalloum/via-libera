# Journey: Steven (factory, simulator, evacuation)

One entry per step, newest at the bottom. Each figure is made by a command in the repo.

## 2026-11-21 · Factory step 1: gas pipes, so leaks start where gas actually is

**What changed and why.** In v0, a leak could start on any open cell next to any machine
(110 sites, all equally likely), including machines that never carry gas, such as the press
line. Real releases come from the gas system, and mostly from its fittings (flanges, valves,
tees, machine connections), not from the middle of a straight pipe. The plant now has a pipe
network:

- **Default plant:** one gas main along y = 21 m runs from the tank farm west to compressor A
  and the furnace, and east through the partition wall to the generator and the dryer.
- **Leaks:** they start only on open pipe cells (24 sites). A fitting is 5 times likelier
  than a plain pipe cell (`FITTING_WEIGHT`). That number is an assumption to tune, not a
  measured value.
- **`Factory.leak_prior`:** the map of where a leak is likely before any reading. This is
  the prior Rayan's locator can start from.
- **`Factory.random`:** routes a pipe tree from a supply point on the outer wall to every hot
  machine plus one other machine. The routing prefers straight runs, with a penalty per
  bend, like real pipework.
- **`Factory.default(pipes=False)`:** the v0 plant, which reproduces the old numbers exactly.

![pipes](../figures/steven/01-pipes.png)

Figure: `uv run sim-viz` (file `1_layout.png`), copied to `docs/figures/steven/01-pipes.png`.

**Numbers** (all-baseline pipeline, `evaluate(..., 30, seed=0)`):

| plant | source error | hazard Brier | exposed (plan / nearest exit) |
|---|---|---|---|
| v0, no pipes | 9.7 m | 0.091 | 5.7 / 7.2 of 30 |
| with pipes | 10.2 m | 0.076 | 5.6 / 7.2 of 30 |

The baseline locator ignores the prior, so its error barely moves. A locator that uses
`leak_prior` should gain the most: 24 candidate cells instead of 110.

**What did not work.** My first default network had two separate headers, one on each side
of the tank farm, so the pipe network was in two pieces (the test caught it). Real tank
farms have a manifold, so the main now runs through the tanks.

## 2026-11-21 · Evacuation step 1: one person, replanned live

**What changed and why.** The plan used to be computed once, at a fixed time. Now one person
is followed second by second:

- **Before the alarm:** the leak starts at t = 0 and the person stands still until the alarm
  (`--alarm`, default 30 s).
- **After the alarm:** every `--update` seconds (default 5) the system reads the sensors,
  asks the source and hazard models for new maps, and replans from where the person stands.
- **Three people from the same spot:**
  - **live:** replanned at every update;
  - **once:** planned at the alarm only;
  - **nearest:** nearest exit, gas ignored.

  All three are scored on the true gas: dose breathed, and seconds spent in toxic or
  flammable cells.
- **Stand-ins until the real models are linked:**
  - `OracleLocator`: the true leak plus an error of `--sigma` m (default 3), on the pipes;
  - `OracleHazard`: the true future, a perfect model and upper bound.

  The baselines can be used instead (`--source centroid`, `--hazard forward`), and any
  `SourceLocator` or `HazardPredictor` plugs in.
- **Planner fix:** routes no longer squeeze diagonally between two solid corners.

![evac live](../figures/steven/02-evac-live.png)

Figure: `uv run evac-live --seed 30 --t-max 120 --story docs/figures/steven/02-evac-live.png`.

**What the figure shows.** This is leak seed 30, on the compressor/furnace branch, with the
person starting at the top left. The nearest exit is E0, on the left, and the plume drifts
there. The nearest-exit person walks into it (dose 0.7, 4 s in harm). The gas-aware person
goes up to E2 instead (dose 0.1, 0 s in harm). In the live window (`uv run evac-live`) you
can click anywhere to move the person.

**Numbers** over 60 leaks, each with one random starting point
(`uv run evac-live --compare 60`):

| setting | live (exposed, harm-s, dose) | once | nearest exit | harm-s before alarm |
|---|---|---|---|---|
| default: source error 3 m, forward-sim hazard, drift 0.05, alarm 30 s | 9, 133, 0.73 | 9, 131, 0.71 | 11, 137, 0.78 | 98 |
| perfect hazard, drift 0.15 | 7, 106, 0.71 | 7, 106, 0.72 | 10, 138, 0.88 | 87 |
| baseline centroid source, drift 0.15 | 8, 126, 1.00 | 8, 132, 1.02 | 10, 138, 0.88 | 87 |
| default, drift 0.15, **alarm 15 s** | 5, 38, 0.22 | 5, 38, 0.22 | 5, 42, 0.29 | 27 |

What we learned (honest, small sample):

1. **Planning around the gas helps:** 7 to 9 people exposed out of 60, against 10 to 11 for
   the nearest exit.
2. **Replanning every 5 s changed nothing here.** In none of the 60 runs did the live person
   end at a different exit than the plan-once person. The hall is small, so a walk takes
   15 to 45 s, and the plume barely changes within one walk. Replanning should start to
   matter with longer walks, faster wind changes, crowds, or a model that gets much better
   as readings come in.
3. **Most harm happens before the alarm:** 98 of the 133 harm-seconds come while people are
   still waiting. Moving the alarm from 30 s to 15 s cuts harm-seconds from 133 to 38. **The
   first decision time matters far more than the update rate.** This is the input for the
   timing study (step 4) and for Rayan and Ahmad: a usable estimate early beats a perfect
   one late.
4. **A bad source estimate can do worse than ignoring the gas.** With the baseline centroid
   locator, the gas-aware people breathe more (dose 1.00) than the nearest-exit people
   (0.88). The planner is only as good as the maps it gets.

**What did not work.** The first version of the still figure drew one legend per panel, and
they overlapped. It now has one legend for the whole figure.

## 2026-11-21 · Timing study: when to decide, how often to replan, how fast to answer

**What changed and why.** We had to pick the decision time (`t_obs`), the update rate, and
how fast TabPFN must answer, without guessing. `uv run evac-timing` runs the live
one-person evacuation on the same 100 leaks (seeds 0-99) under every timing.

- **Two model settings bracket what TabPFN can reach:**
  - **baseline:** today's centroid locator plus forward-simulation hazard;
  - **perfect:** true leak plus true future.
- **New in the live loop:** `latency`, the seconds from readings to route. Until the first
  route arrives, people either walk to the nearest exit (`before_plan="nearest"`) or wait
  where they are (`"wait"`).

![timing](../figures/steven/03-timing.png)

Figure: `uv run evac-timing --leaks 100 --workers 10` (3.5 min on 10 cores), copied from
`outputs/timing/timing.png`. Re-plot without rerunning: `--plot-only`.

**Numbers** (per person, mean of 100 leaks; harm-s = seconds in toxic or flammable gas):

| first decision `t_obs` | 10 s | 20 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|---|
| harm-s before the alarm | 0.3 | 1.0 | 2.0 | 5.9 | 15.0 |
| % exposed, nearest exit | 6 | 14 | 17 | 22 | 25 |
| % exposed, live, baseline models | 6 | 11 | 15 | 18 | 21 |
| % exposed, live, perfect models | 6 | 9 | 13 | 17 | 20 |
| baseline source error (m) | 5.9 | 7.4 | 8.6 | 9.7 | 10.4 |
| baseline hazard Brier | 0.078 | 0.084 | 0.094 | 0.103 | 0.115 |

| model latency (alarm 30 s, perfect models) | 0 s | 2 s | 5 s | 10 s | 20 s |
|---|---|---|---|---|---|
| harm-s, walk to the nearest exit meanwhile | 2.31 | 2.53 | 2.65 | 2.81 | 2.84 |
| harm-s, wait for the route | 2.31 | 2.57 | 3.00 | 3.70 | 5.04 |
| harm-s, nearest exit without any model | 2.84 | 2.84 | 2.84 | 2.84 | 2.84 |

Seconds between replans (2 to 60 s) changed nothing: 2.92-2.96 harm-s with the baseline
models, 2.23-2.24 with the perfect ones.

What we learned (2D simple physics, one person per leak, 60 x 40 m hall):

1. **Decide early.** Every extra 10 s of waiting costs about 1 to 1.5 harm-seconds per
   person, almost all of it while people are still standing still. Exposed people go from
   6 % at 10 s to 22 % at 60 s. No better estimate can buy that back: even perfect models
   at 60 s (17 %) do worse than the nearest-exit plan at 20 s (14 %).
2. **The baseline locator gets worse with time** (5.9 m at 10 s, 10.4 m at 120 s): the
   reading-weighted centroid drifts downwind as the plume spreads. A model that reads the
   time series should improve instead. **Rayan's locator is built for 60 s** (readings at
   20, 40 and 60 s). We need versions that answer at 20 to 30 s.
3. **Routing helps, and better models help more.** After the alarm (at 30 s), routing
   controls 0.88 harm-s per person with the nearest exit, 0.82 with baseline models and
   0.35 with perfect models. That gap is the room for TabPFN.
4. **Replanning rate does not matter here.** Walks are short (15-45 s) and the plume
   changes slowly, so one route at the alarm is enough. Replanning every 10-30 s keeps
   TabPFN calls (and tokens) low. It may matter later with crowds and longer routes.
5. **Answer fast, and never make people wait.** With people walking to the nearest exit
   meanwhile, a route that arrives after 2 s keeps about 60 % of the gain, after 5 s about
   35 %, after 10 s almost nothing. Waiting for the route is worse than the plain
   nearest-exit plan as soon as the model takes more than about 4 s. So the alarm should
   send people to the nearest exit at once, and a guide robot should redirect them when the
   route lands. This sets the target for step 5 (cache, fast model, local inference):
   **a route within about 2 s of the readings.**

**What did not work.** The exact source stand-in showed 0.32 m error: its blur was one cell
wide, so a neighbouring fitting (5 times likelier) sometimes outranked the true cell. The
blur is now half a cell, with a test over every leak site. In the first figure, the harm
the planner controls was hidden under the pre-alarm area, so it now has its own panel.
Caveats: the alarm here sounds at a fixed time. A real detector fires when gas reaches it,
which is the next thing to model.

## 2026-11-21 · Detector alarm and the sync check

**What changed and why.** The team agreed that the whole plant evacuates as soon as a
leak is detected. The default is the nearest exit, and a route replaces it when the models
say that exit is unsafe.

- **Detector alarm:** the alarm is now the first sensor sample (every 5 s, as Rayan said
  our sensors read) above `sim.ALARM_LEVEL` = 10 % of LEL. `run_live(t_alarm=None)` and
  `evac-live` use it by default. A fixed `t_alarm` still works for the timing study.
- **`uv run sync-check`:** runs the closed loop with any locator and predictor and checks
  every hand-over: shapes, the (iy, ix) grid, absolute times, probabilities, solid cells,
  ignition zones and routes. It also times each call. The teammates run it on their model
  before pushing.
- **`docs/sync.md`:** writes down the shared conventions, plus a heads-up list from
  reading their branches.

![sync timeline](../figures/steven/04-sync-timeline.png)

Figure: `uv run sync-check --timeline docs/figures/steven/04-sync-timeline.png`.

**Numbers:**
- **Alarm time** (default plant, 200 leaks, 14 sensors):
  - median 10 s, 90 % within 15 s, max 25 s with sensors read every 5 s;
  - median 6 s with every 1 s;
  - median 10 s with every 10 s.
- **Readings at the alarm:** 2 to 6 per sensor (median 3).
- **Walk to the nearest exit, no crowd:**
  - default plant: median 12 s, max 24 s;
  - random 200 x 100 m plant: median 39 s, 95th percentile 69 s, max 81 s.
- **Baselines through the sync check:** 0 problems at decision times 10/20/30/60 s on
  the default and a random plant. Warnings: the centroid map puts only 14-25 % of its
  mass on the pipes, and it gives no release rate.

What we learned:

1. **The models must answer at about 10 s, from about 3 readings.** Rayan's locator reads
   20/40/60 s and Ahmad's hazard is trained on 70-180 s: both are tuned for a moment when,
   on our plant, everyone is already outside.
2. **"Seconds since the leak started" is a simulation luxury.** A real plant only knows
   when the alarm went off, so features should be relative to the alarm.
3. **Large plants change the picture.** Walks of minutes make long forecasts and
   replanning matter, and crowds at the doors make the "fastest path" the wrong goal.
   `docs/research/evacuation.md` proposes capacity-constrained routing (CCRP with a risk
   cost) plus a cell-queue crowd model to score it.

**What did not work.** I test-merged the teammates' current branches locally. Rayan's
tests pick leak sites by fixed index and fail with 24 pipe sites; they all pass on
`Factory.default(pipes=False)`. This is a heads-up for their next push, not a merge
blocker: their branches are still in progress.
