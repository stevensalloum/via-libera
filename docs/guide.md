# Codebase guide: what we have, what each part does, what to use it for

State of `main` on 2026-11-21 after PR #17, plus the open PRs at the end. For the plan
and targets read `docs/brief.md`; for team rules read `AGENTS.md`. This guide explains the
code. It is updated whenever a PR changes what a module does.

## 1. The system in one picture

```
 sim.Factory (the plant)            sim.Scenario (one leak: where, how strong, wind)
          \                              /
           sim.simulate  ---> concentration of gas in every cell, at any time  (the "truth")
                 |
           sim.readings  ---> SensorReadings: 14 sensors x every 5 s up to t_obs = 60 s + wind
                 |
   [source]  locate(readings)            -> SourceEstimate: P(leak is in this cell), map
                 |
   [hazard]  predict(readings, source)   -> HazardMap: P(toxic), P(fire) per cell, 70-180 s
                 |
   [evac]    plan(people, hazard)        -> EvacPlan: one route per person to an exit
                 |
 pipeline.run_episode  scores every step against the truth from sim.simulate
```

**Important: on `main`, TabPFN is not yet used anywhere in this chain.** Each step is a
simple baseline (no learning). The TabPFN versions are the milestone-1 work of each owner;
the first one is Ahmad's hazard predictor (open PR #24). The TabPFN tooling they build on
is ready in `core/`.

## 2. Commands

| Command | What it does |
|---|---|
| `uv sync` | Create / update `.venv` with all dependencies (add `--extra local` for local TabPFN + PyTorch, multi-GB) |
| `uv run pytest` | All tests (about 1.5 s, no network) |
| `uv run ruff check .` / `uv run ruff format .` | Lint / format |
| `uv run leak-demo [--episodes N --seed S]` | Run one emergency with the baselines, save `outputs/leak_demo.png`, print scores over N emergencies |
| `uv run tabpfn-check` | Verify your TabPFN key, show token usage and free cost quotes (spends 10,000 tokens) |
| `uv run spatial-example --backend baseline\|cloud\|local` | Toy demo of `core`: a full map from 200 sparse readings |
| `uv run sim-viz` (PR #23) | Draw the simulator for a visual check into `outputs/sim/` |

## 3. Every file

### Root and configuration

| File | Purpose |
|---|---|
| `README.md` | Project summary, setup (uv or pip), daily git flow, TabPFN key, dependency rules |
| `AGENTS.md` | Rules for every coding agent: ownership, communication, workflow, code style, budget, secrets |
| `CLAUDE.md` | One line, `@AGENTS.md`: Claude Code reads the same rules |
| `pyproject.toml` | Package metadata, dependencies, optional `local` extra, CLI commands, ruff and pytest settings |
| `uv.lock` | Exact versions of every dependency (generated; do not edit) |
| `requirements.txt` | Same pins for pip users, exported from `uv.lock`; CI fails if it drifts |
| `.env.example` | Template for `.env` (your `TABPFN_TOKEN`; `.env` itself is gitignored) |
| `.gitignore` | Ignores venvs, caches, `.env`, `/data/`, `/outputs/`, `/checkpoints/`, model and data files |
| `.gitattributes` | Forces LF line endings on every OS (Windows safety) |
| `.github/workflows/ci.yml` | CI on every PR and push to main: `uv sync --locked`, requirements check, ruff, pytest |
| `.github/pull_request_template.md` | PR form: Closes #, What, Why, How tested, checklist |

### Docs

| File | Purpose |
|---|---|
| `docs/brief.md` | The plan: story, system, simulator facts, baselines to beat, per-area TabPFN recipes, milestones |
| `docs/guide.md` | This file |
| `docs/decisions.md` | Design decisions with date, reason, alternatives, and measured numbers |
| `docs/status/{steven,rayan,ahmad}.md` | Each person's Doing / Done / Needs / Blocked; only that person's agent edits it |

### Package `src/tabpfn_spatial/`

| File | Owner | Purpose |
|---|---|---|
| `__init__.py` | Steven | Package docstring and version |
| `contracts.py` | Steven | The data types passed between areas and the thresholds (section 4) |
| `pipeline.py` | Steven | Runs and scores whole emergencies; `leak-demo` (section 8) |
| `env.py` | Steven | Loads `.env`, returns the TabPFN token or a clear error |
| `tabpfn_check.py` | Steven | The `tabpfn-check` command |
| `sim/` | Steven | Factory and gas simulator (section 5) |
| `source/` | Rayan | Leak source detection (section 6) |
| `hazard/` | Ahmad | Hazard prediction (section 6) |
| `evac/` | Steven | Evacuation planning (section 6) |
| `core/` | Steven | TabPFN backends and tools used by every area (section 7) |

### Tests `tests/`

Mirror the package: `tests/sim/`, `tests/source/`, `tests/hazard/`, `tests/evac/`,
`tests/core/`, plus `test_pipeline.py`, `test_env.py`, `test_tabpfn_check.py`,
`test_smoke.py` (every subpackage imports). They never call the TabPFN API: TabPFN is
replaced by small fakes, so CI is free and fast.

## 4. Contracts (`contracts.py`): the language between areas

| Name | Shape / fields | Meaning |
|---|---|---|
| `TOXIC_LIMIT = 0.1` | | Concentration above which a person is harmed (relative units) |
| `LEL, UEL = 0.3, 3.0` | | Flammable range; fire risk only next to hot machines |
| `SensorReadings` | `positions (k,2)`, `times (t,)`, `values (k,t)`, `wind (2,)`; `t_obs` | What the plant measured up to the decision time |
| `SourceEstimate` | `prob (ny,nx)` sums to 1; optional `rate` quantiles; `top_cells(n)` | Where the leak probably is |
| `HazardMap` | `times (h,)`, `p_toxic (h,ny,nx)`, `p_fire (h,ny,nx)`; `at(t)` | Probability of harm per cell at future times |
| `Route` | `cells (m,2)` grid path, `exit_index`, `risk` | One person's way out and the planner's own risk estimate |
| `EvacPlan` | `routes`, `blocked_exits` | All routes |
| `SourceLocator` / `HazardPredictor` / `EvacPlanner` | one method each | What each area must implement |

Grid convention everywhere: arrays are `[iy, ix]` (row = y), positions are metres `(x, y)`.

## 5. The simulator (`sim/`)

### `sim/factory.py`

- `Machine(name, x0, y0, x1, y1, hot)`: a solid rectangle; `hot=True` means it can ignite gas.
- `Factory`: the plant as a grid. Fields: `width`, `height`, `cell` (2 m), `walls` (solid
  cells, machines included), `machines`, `exits (e,2)`, `sensors (k,2)`.
  - `ny`, `nx`: grid size (20 x 30 for the default plant).
  - `cell_of(xy)` -> `(iy, ix)`; `centre(iy, ix)` -> metres; `centres()` -> all centres.
  - `ignition`: open cells within 2 cells (4 m) of a hot machine.
  - `leak_sites`: open cells within 1 cell of any machine (where leaks are sampled).
  - `Factory.default()`: 60 x 40 m, two halls split by a wall at x = 30-32 m with doors at
    y = 6-12 m and 26-32 m; machines: compressor A, furnace (hot), tank farm, press line,
    generator (hot), dryer (hot); 4 exits; 14 sensors.
  - `Factory.random(rng)` (PR #23): random machines, wall and doors, one exit per side,
    16 sensors; guaranteed that every open cell can reach an exit.

### `sim/plume.py`

- `Scenario(source, rate, wind, drift=0, seed=0)`: one leak. `drift` (PR #23) makes the
  wind direction wander (random walk, reproducible by `seed`).
- `wind_at(scenario, times)` (PR #23): the true wind vector at each time.
- `simulate(factory, scenarios, times)` -> `(B, T, ny, nx)` concentrations.
  **Physics:** 2D advection-diffusion. Each step, gas moves with the wind (upwind scheme)
  and spreads by turbulent mixing (`DIFFUSIVITY = 0.8 m^2/s`); the leak adds `rate` per
  second into its cell; walls and machines block gas (no flux); the outer boundary lets gas
  out (ventilation). Many leaks run at once: 500 leaks x 120 s take about 1 s.
  **Known limitation (v0/v1):** the wind is one vector for the whole hall; it does not bend
  around machines. Gas blown into a machine face piles up there instead of flowing around,
  which creates artificial high-concentration spots next to machines (visible in
  `sim-viz`). Fix proposed: a precomputed flow field around obstacles.
- Units are relative: 1.0 is what the leak cell holds after 1 s at rate 1.

### `sim/scenarios.py`

- `sample_scenarios(factory, n, rng, drift=0)`: random leaks next to machines, rate 0.5-2
  (log-uniform), wind 0.2-1.5 m/s from any direction, one seed per leak.
- `sample_people(factory, n, rng)`: random positions on open floor.
- `sensor_values(factory, fields)`: read the clean values at the sensor cells.
- `readings(factory, scenario, t_obs, rng, ...)`: the noisy `SensorReadings` the system sees:
  10 % multiplicative + 0.001 additive noise, wind with 0.1 m/s noise; PR #23 adds dead
  (`dropout`, NaN rows) and stuck (`stuck`) sensors and reports the mean wind over the window.

### `sim/viz.py` (PR #23)

`plot_layout`, `plot_timelapse`, `plot_readings`, `plot_random_layouts`; `uv run sim-viz`.

## 6. The three application areas (baselines today)

| Area | Baseline class | What it does | Its score |
|---|---|---|---|
| Source (`source/baseline.py`) | `WeightedCentroidLocator(sigma=6)` | Average of sensor positions weighted by their last reading, spread as a 6 m Gaussian blob | `localisation_error(factory, est, true)`: metres from the most likely cell to the true leak |
| Hazard (`hazard/baseline.py`) | `ForwardSimPredictor` | Takes the single most likely source cell, reruns the simulator with the measured wind, thresholds: P = 0 or 1 | `brier(prob, truth)`: mean squared error of probabilities vs what really happened |
| Evac (`evac/planner.py`) | `RiskAwarePlanner(risk_weight=200, speed=1.3)` | Dijkstra on the grid (8 directions); each step costs its length + 200 m x P(harm) of that cell at the moment the person arrives; with zero hazard it is the standard nearest-exit plan | `route_exposed(...)`: was any cell on the route truly harmful when the person got there |

Also `hazard.hazard_from_fields` turns concentrations into toxic / fire masks (used for
the truth too).

## 7. Shared TabPFN tooling (`core/`)

| Module | What it gives | When to use it |
|---|---|---|
| `core/models.py` | `make_regressor(backend)` / `make_classifier(backend)` with backends `"cloud"` (TabPFN-3.5 on Prior Labs servers via `tabpfn-client`), `"local"` (`tabpfn` on your GPU/CPU), `"baseline"` (k-nearest-neighbours in numpy), `"auto"` (local GPU, else cloud if a key is set, else baseline). Regressors return a `QuantileDist`; classifiers give `predict_proba` | Every TabPFN model in the project goes through this, so backends can be swapped and tests can use the baseline |
| `core/dist.py` | `QuantileDist`: 19 quantiles (5 %-95 %) per row, `median`, `interval(0.9)`, `width()`, `mean`, `mode` | Turn TabPFN regression into probabilities, e.g. P(c > limit) |
| `core/budget.py` | `CostGuard(max_tokens_per_call=200_000)`: asks the server for a free quote before every cloud call, refuses expensive ones, counts tokens; `cloud_quote` | Automatically on for the cloud backend |
| `core/grid.py` | `Grid2D`: a 2D grid of query points as one batch, back to an image | Predict a whole map in one call |
| `core/metrics.py` | `crps` (quality of a distribution), `coverage` (do 90 % intervals hold 90 %), `ece` (calibration of class probabilities) | Every number we pitch |
| `core/viz.py` | `plot_map`: truth / mean / uncertainty heatmaps | Quick figures |
| `core/example.py` | `uv run spatial-example`: full map from 200 readings | Reference for how to use the above |

**TabPFN facts we measured:** each key has 50M credits; every cloud call costs at least
10,000 tokens (more with many test rows); TabPFN-3.5 handles up to 160 classes; on the Mac
GPU the local model answers 100-class problems in about 1 s. Tests never call the API.

## 8. Pipeline (`pipeline.py`)

- Constants: `T_OBS = 60` s (decision time), forecasts every `STEP = 10` s up to
  `HORIZON = 120` s later (70-180 s).
- `run_episode(factory, scenario, people, locator=, predictor=, planner=, rng=)` runs one
  emergency with any implementations (baselines by default) and returns an `Episode` with
  everything plus the scores: `source_error_m`, `hazard_brier`, `exposed_plan`,
  `exposed_standard` (people exposed with our plan vs the nearest-exit plan).
- `evaluate(factory, n_episodes, n_people, seed, locator=..., ...)`: mean scores over random
  leaks; the same seed gives the same leaks, so different implementations are comparable.
- `plot_episode(factory, episode, path)`: three panels (truth, source map, hazard + routes).
- **Known issue (flagged in PR #24):** the pipeline combines toxic and fire as independent
  (`1 - (1 - p_toxic)(1 - p_fire)`), but with these thresholds flammable gas is always also
  toxic, so the combined risk is overstated. Fix: use `max(p_toxic, p_fire)` or define harm
  directly; integrator task.

## 9. Baselines today (30 emergencies, seed 0)

Source error 9.4 m (9.7 m after PR #23's seed change), hazard Brier 0.089 (0.091), people
exposed 5.4 vs 7.0 of 30 (5.9 vs 7.3).

## 10. Open work

| PR / issue | Owner | Status |
|---|---|---|
| PR #23, issue #20: sim v1 (drifting wind, sensor faults, random plants, `sim-viz`) | Steven | Open, waiting for the visual review |
| PR #24, issue #19: TabPFN hazard predictor (`hazard/features.py`, `library.py`, `tabpfn_predictor.py`, `benchmark.py`) | Ahmad | Open; first result on 3 emergencies: Brier 0.0566 vs 0.0591, AUC 0.855 vs 0.799, 30k tokens; 30-emergency run pending |
| Issue #18: TabPFN source locator | Rayan | In progress |
| Issue #21: risk-budget evacuation planner | Steven | Not started (after the simulator review) |
| Issue #22: demo UI | Steven | After milestone 1 |

## 11. What came from the prep repo (yesterday)

Ideas and tooling, not files: the `core/` starter (backends, quantiles, metrics, budget
guard) was built fresh for this repo. Reused techniques: simulation-based inference with a
library of simulated cases ("X-ray by touch", now the source recipe), risk-budget decisions
(the tower experiment, now the evacuation recipe), deciding by probability mass instead of
the mean. The prep repo's 3D three.js world (`app/world.html`) is a candidate for the demo
UI; it is only on Steven's laptop.

## Glossary

- **Brier score:** mean of (predicted probability - outcome)^2; 0 is perfect.
- **Calibration / ECE:** do 70 % predictions come true 70 % of the time; 0 is perfect.
- **CRPS:** how close a predicted distribution is to the truth; lower is better.
- **Simulation-based inference:** learn "observations -> hidden cause" from many simulations.
- **Advection / diffusion:** gas carried by the wind / spreading by mixing.
- **LEL / UEL:** lower / upper explosive limit; between them a mixture can burn.
