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
