# Keeping the three parts in sync

The closed loop only works if source, hazard and evacuation agree on **when**, **where**
and **in which units**. This page fixes those conventions. `uv run sync-check` tests them
on any model before you push. Status: proposal by Steven (2026-11-21); built on PRs #23,
#27 and this PR.

## 1. Check your model before pushing

```bash
uv run sync-check                                   # the baselines (must pass)
uv run sync-check --source tabpfn_spatial.source:TabPFNLocator
uv run sync-check --hazard tabpfn_spatial.hazard:TabPFNHazardPredictor
uv run sync-check --times 10 20 30 --every 2 5 10   # other decision times / sensor periods
uv run sync-check --timeline docs/figures/steven/04-sync-timeline.png
```

- **How each class is built:** with `backend="baseline"` if its constructor takes a
  `backend`, so no API tokens are spent.
- **Fitting:** objects with `fit(factory)` are fitted once per plant.
- **The loop:** readings, then `locate`, then `predict`, then `plan`. This runs on the
  default plant and a `Factory.random` plant, at decision times 10/20/30/60 s, on 3 leaks.
- **What is checked:** every hand-over (section 3), and each call is timed.
- **Result:** exit code 1 if any hard problem; warnings are listed but allowed.

## 2. Conventions

### Time

| What | Convention |
|---|---|
| Clock | Seconds since the leak started (`SensorReadings.times`, `HazardMap.times`, `t_start`). In simulation we know this clock. In a real plant we do not; see the open question below. |
| Sensor period | 5 s (`readings(every=5.0)`). Must work for other periods: real datasets differ. `sync-check --every` tests that. |
| Alarm | First sensor sample above `sim.ALARM_LEVEL` = 0.03 (10 % of LEL). On the default plant: median 10 s, 90 % within 15 s, max 25 s (200 leaks). |
| Decision time `t_obs` | = the alarm. At that moment there are only **2 to 6 readings** per sensor (median 3). |
| What people do | The whole plant evacuates at the alarm in stages (Ahmad's hybrid). People start for the nearest exit at once; the forward-simulation baseline route replaces it within ~0.01 s; the TabPFN route replaces that when it lands. |
| Latency budget | **Measured** (Ahmad, 30 early-alarm leaks, cloud TabPFN, centroid source): source + hazard + one route takes a median of 3.9 s and 4.9 s at p95. The fast model is ~2.7 s for the hazard call alone. The 2 s target of the timing study is not reachable in the cloud, so we design for **~5 s**. The staged routes cover the gap: in his replay, 1/30 people were exposed with staged routes vs 2/30 with TabPFN alone and 4/30 with the nearest exit. Rayan's locator time still has to be added. |
| Replanning | Every 10-30 s is enough on this plant. |
| Hazard forecast | Absolute times `t_obs + 0, 10, ..., 120 s`. On the default plant a typical person is out by about 35 s (alarm ~10 s plus a walk of at most 24 s, no crowding), so the first 30-60 s of the forecast matter most. |

**Open question for the real-data phase:** the leak start is unknown in a real plant.
Features should be relative to the alarm (for example readings at alarm - 10 s, at the
alarm, and at alarm + 5 s), not to t = 0. We keep the shared clock for simulation, but
please build features so that switching to "seconds since the alarm" is a one-line
change.

### Space

| What | Convention |
|---|---|
| Positions | Metres, origin bottom-left, `(x, y)`. |
| Grid arrays | `[..., iy, ix]` (row = y), shape `(factory.ny, factory.nx)`; cell `(iy, ix)` covers `[ix*cell, (ix+1)*cell) x [iy*cell, (iy+1)*cell)`. `factory.cell_of(xy)` and `factory.centre(iy, ix)` convert. |
| Leak sites | `factory.leak_sites`: open cells on the gas pipes (24 on the default plant; it was 110 before the pipes). Index sites by position, never by a fixed number. |
| Leak prior | `factory.leak_prior`: (ny, nx) map that sums to 1, fittings 5x likelier. Start the source estimate from it. |
| Ignition zones | `factory.ignition`: `p_fire` must be 0 elsewhere. |
| Plants | Everything must run on `Factory.random(rng)` too (other sizes and layouts). |

### Units

Concentration is relative (1.0 = the source cell after 1 s at rate 1). The thresholds are
`TOXIC_LIMIT` = 0.1 and `LEL`/`UEL` = 0.3/3.0, with alarm 0.03. Wind is in m/s, release
rate in units/s (sampled 0.5-2, median 1).

## 3. What each hand-over must satisfy (checked by `sync-check`)

| Hand-over | Hard checks | Warnings |
|---|---|---|
| sim -> `SensorReadings` | positions = `factory.sensors`; times ascending, >= 0; values (sensors, times); wind finite (u, v); dead sensors are NaN rows | |
| source -> `SourceEstimate` | `prob` shape (ny, nx), finite, >= 0, sums to 1, 0 on solid cells; if `rate` is given: levels ascending in (0, 1), quantiles positive and non-decreasing | < 99 % of the mass on leak sites; no `rate` (hazard then assumes 1.0) |
| hazard -> `HazardMap` | `times` = the requested absolute times; `p_toxic`, `p_fire` (times, ny, nx) in [0, 1]; 0 on solid cells; `p_fire` 0 outside `factory.ignition` | |
| evac -> `EvacPlan` | one route per person; starts at the person's cell, ends at its exit; no solid cells, no jumps; risk in [0, 1] | |

Crashes count as problems. A model that only works at t_obs = 60 s fails at 10 s and is
reported, not hidden.

## 4. Heads-up from the branches as of 2026-11-21 (work in progress)

These come from reading `feat/source-tabpfn` and `feat/hazard-tabpfn` and test-merging
them locally with #23 and #27. Nothing was pushed. They are notes for the next push, not a
review of unfinished work.

| # | Part | What will not line up | Suggested change |
|---|---|---|---|
| 1 | source | Features read the sensors at 20, 40 and 60 s, and the library is built at `T_OBS = 60`. The alarm comes at ~10 s and routes must land within seconds of it. | Train at the decision time we will use (alarm, roughly 10-30 s), from the readings available then. Keep one feature set per `t_obs`, or make the times relative to the alarm. |
| 2 | source | The tests pick leak sites by index (`leak_sites[60]`, `[30]`, ...). With pipes there are 24 sites, so 5 tests fail after the merge; all of them pass on `Factory.default(pipes=False)`. | Pick sites by position or `rng.choice(len(f.leak_sites))`. |
| 3 | source | `leak_zones(n_zones=100)` merges 110 sites. With 24 sites no merging happens, which is fine, but the zone count now depends on the plant. | Nothing to do; just do not hard-code 100 classes. |
| 4 | source | No release rate in `SourceEstimate`. | Fill `rate` / `rate_levels` (data.md rule 3). |
| 5 | source | The library's measured wind is the initial wind. Since sim v1, `readings` reports the mean wind over the window (they differ when `drift` > 0). | Use `sim.readings`, or `wind_at(...).mean()`, so training matches run time. |
| 6 | hazard | **Done (#24):** context covers early (5 s and later) decisions, with forecasts over 0-120 s after the alarm. | |
| 7 | hazard | **Done (#24):** trained on sampled estimated sources and measured (noisy) wind, context seed 1001. | Rate still falls back to 1.0 until the source supplies it (#4). |
| 8 | hazard | **Measured (#24):** querying only open cells cut the rows from 23,400 to 20,085 and the median from 3.68 to 3.45 s (5 cases). The budget is now ~5 s (section 2). | The staged routes make the latency acceptable. Next levers: fewer forecast times, cells near people only, local GPU. |
| 9 | both | Features use "seconds since the leak started", which a real plant does not know. | See the open question in section 2. |

## 5. Merge order

1. **#23** (sim v1 + pipes), then **#27** (live evacuation + timing), then this PR. Each
   one is based on the one before.
2. **Teammates:** rebase on `main`, run `uv run sync-check` with your class, and push.
3. **Integrator:** runs `sync-check` with both TabPFN classes together, then merges.
