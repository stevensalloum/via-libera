<p align="center"><img src="docs/brand/via-libera-logo.webp" alt="Via Libera" width="420"></p>

# Via Libera

**Find the leak, predict the gas, route everyone out.** When gas leaks in a factory, today's
answer is a fixed green sign to the nearest exit, even when that exit is inside the cloud.
Via Libera reads the gas detectors, locates the leak in 3D, predicts where the gas will be
harmful, and gives every person their own way out around it, updated as the gas moves.
Each step is a TabPFN model fitted in seconds on simulations of *that* plant: no training,
no real leak data needed.

Built at the Berlin AIxRobotics Hackathon (Prior Labs track) by Steven, Rayan and Ahmad.
The Python package is still called `tabpfn_spatial` (the project's working name).

## Results (final 3D test, cloud TabPFN)

Same leaks, same people, 9 plants (5 known layouts, 4 never seen before), 3D gas, alarm from
noisy detectors, models updated every 15 s ([table](docs/results/final-table.md)):

| | known plants | unseen plants |
|---|---|---|
| seconds in harmful gas per person: nearest exit (today) | 0.23 s | 0.84 s |
| ... Via Libera (CCRP + TabPFN) | **0.14 s** | **0.09 s** |
| ... with perfect knowledge of the gas | 0.14 s | 0.01 s |
| people exposed: nearest exit / Via Libera | 3.7 % / 2.3 % | 6.2 % / 2.4 % |
| 95 % of people out: nearest exit / Via Libera | 41 s / 42 s | 43 s / 40 s |
| leak position / height error (median) | 1.1 m / 0.95 m | 0.6 m / 0.58 m |

With Rayan's 6 further unseen plants (15 plants, 10 never seen, one leak each): seconds in gas
per person 0.50 s → **0.09 s (−83 %)**, on unseen plants 0.63 s → **0.06 s (−90 %)**, people
exposed 5.0 % → **1.8 % (−64 %)**, as good as perfect knowledge of the gas.

Against a classic k-NN in TabPFN's place (same 9 plants and leaks, `--backend baseline`):
TabPFN locates the leak 3.5x closer (median 0.63 m vs 2.20 m; height 0.63 m vs 1.63 m). The
k-NN hazard map is broader (catches 76 % of harmful cells vs 26 %, with more false alarms),
which helps routing on unseen plants (0.02 s vs 0.09 s in gas per person; known plants: TabPFN
0.14 s vs k-NN 0.21 s). Widening TabPFN's hazard map is the next step.

On real gas measurements TabPFN locates the leak better than a classic k-NN on the same rows
(Red:Vapor 0.38 m vs 0.64 m; SRGDM 0.83 vs 1.50 cells) and ranks hot zones at least as well
(AUC 0.88 / 0.62 / 0.98 vs 0.88 / 0.60 / 0.94 on Red:Vapor, Örebro, UCI).
Figures: [docs/figures/final](docs/figures/final).

## Demo (for the presentation)

**One command, nothing computed, no tokens:**

```bash
uv run via-libera              # the story page + live map, "Enter the factory" for 3D (saved cloud run)
```

The viewer loads Three.js from a CDN: have internet (or a phone hotspot) on the demo laptop.
Other commands:

```bash
uv sync                                                   # once
uv run via-libera --list                                  # saved demo runs (demo/)
uv run world --replay --rung T --leak 500 --backend cloud # a new 3D run (about 1M tokens)
uv run world --keep                                       # replay the last run again, free
uv run world --replay --rung T --leak 500 --backend baseline  # offline, no tokens
uv run evac-pipeline --models team --backend baseline     # the 2D view: one person, live
uv run gallery Final                                      # every figure + video, in pitch order
uv run evac-compare                                       # redraw the "why TabPFN" figures
```

In the 3D viewer (http://127.0.0.1:8770):
- **Play**, speed, and which crowd to show (nearest exit, CCRP planned once / replanned live);
- **click a person**: their three paths drawn live (green replanned, orange planned once, red
  nearest exit), seconds in gas, and the gas they breathe over time;
- **Follow** (camera behind them) or **Person view** (through their eyes, green route ahead);
- **cinematic camera** + **Record video** (saved into the run's `videos/` folder);
- the map (bottom right): toxic gas, predicted danger, leak guesses, alarming detectors,
  people, the selected person's paths; click it to fly there;
- **Walk inside**: W/S walk, A/D step aside, Q/E or the arrows turn, Shift runs.

Reproduce the final results (cloud credits: about 1M tokens per plant and leak):

```bash
uv run evac-final --backend cloud --leaks 1 --unseen 1 --n-train 100   # 9 plants
uv run evac-final --backend baseline --leaks 1 --unseen 1 --n-train 100  # k-NN, offline
uv run evac-final --backend cloud --stress --leaks 1 --unseen 1 --n-train 100  # hard case
uv run evac-compare                                       # figures -> docs/figures/final
```

Full presentation guide (pitch timing, shot list, backups): [docs/presentation.md](docs/presentation.md).

### More views

| Command | What it shows |
|---|---|
| `uv run world` | The 3D factory: orbit, fly to machines, walk inside |
| `uv run world --layout L-plant` | A shaped plant (`L-plant`, `U-plant`, `T-plant`, `campus`), or an unseen one: `--layout U --seed 7` |
| `uv run sim-viz` | The plant: machines, gas pipes, sensors, exits; gas spreading over time |
| `uv run evac-live` | One person evacuated live from the detector alarm (click to move them) |
| `uv run evac-crowd-live --rung B` | A whole crowd, three planners side by side |
| `uv run evac-shapes` | The unseen-shapes study (2D) |
| `uv run sync-check` | Do the source, hazard and evacuation parts agree on time, space, units? |

How each part evolved, with figures: `docs/journey/`.

## Team

- Steven ([@stevensalloum](https://github.com/stevensalloum)), integrator
- Rayan ([@RayanElKhoury](https://github.com/RayanElKhoury))
- Ahmad ([@Asorus](https://github.com/Asorus))

## Setup (once per laptop)

Requires Python 3.12+. Recommended: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12
itself if missing). Clone outside iCloud / OneDrive folders.

```bash
gh repo clone stevensalloum/via-libera
cd via-libera
uv sync                       # creates .venv from uv.lock
uv run pytest                 # expect all tests to pass
cp .env.example .env          # Windows: copy .env.example .env ; fill in keys, never commit
```

Without uv (plain pip), use `requirements.txt`, which mirrors `uv.lock`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate     # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
pytest
```

## TabPFN API key

Each person puts their own key in `.env` as `TABPFN_TOKEN=...` (never commit it), then:

```bash
uv run tabpfn-check           # verifies the key, shows usage and free cost quotes
```

Costs (measured 2026-11-21, TabPFN-3.5, 8 estimators): **10,000 tokens minimum per call**,
growing mainly with test rows (1k train + 100k test x 10 cols = 26.6k; 1M test rows = 2M).
Batch your queries into few large `predict` calls; quote first with
`tabpfn_client.estimate_cost` (free, sends only shapes).

## Shared starter (`tabpfn_spatial.core`)

Every application builds on the same pieces:

```python
from tabpfn_spatial.core.grid import Grid2D
from tabpfn_spatial.core.metrics import coverage, crps
from tabpfn_spatial.core.models import make_regressor

model = make_regressor("cloud")  # "local", "baseline" or "auto"
dist = model.fit(X_readings, y_readings).predict_dist(Grid2D((0, 10), (0, 10), 60, 60).points())
dist.mean, dist.median, dist.width(0.9)  # per-cell prediction and uncertainty
```

| Module | What it gives you |
|---|---|
| `dist` | `QuantileDist`: quantiles, `interval(0.9)`, `width`, mean, mode |
| `models` | `make_regressor` / `make_classifier` over TabPFN cloud, TabPFN local, k-NN baselines |
| `budget` | `CostGuard`: free quote before every cloud call, refuses calls above 200k tokens |
| `grid` | `Grid2D`: a whole map as one batched query, and back to an image |
| `metrics` | `crps`, `coverage`, `ece`: the numbers to pitch |
| `viz` | `plot_map`: truth / mean / uncertainty heatmaps |
| `example` | `uv run spatial-example --backend cloud`: full map from 200 sparse readings |

Local TabPFN (PyTorch, GPU if available, multi-GB download): `uv sync --extra local`.

## Daily flow

1. Start of every work block: `git switch main && git pull`
2. Pick your GitHub Issue (`gh issue list --assignee @me`) and branch from fresh main:
   `git switch -c feat/<topic>`
3. Test first, then code. Before pushing:
   `uv run ruff check . && uv run ruff format --check . && uv run pytest`
4. Update `docs/status/<name>.md`, commit (`feat: ...`), push, then `gh pr create`
   and fill in the template: `Closes #<issue>`, What, Why, How tested. Agents pass it
   with `--title` and `--body`; do not use `--fill`, which drops the template.
5. After Steven merges: `git switch main && git pull && git branch -d <branch>`

## Checks

```bash
uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff format .          # format
```

## Dependencies

Add or change them only in a PR the integrator reviews:

```bash
uv add <package>              # updates pyproject.toml and uv.lock
uv export --format requirements-txt --no-hashes --group dev --no-emit-project --quiet -o requirements.txt
```

CI fails if `requirements.txt` is out of sync with `uv.lock`.

## Repository layout

```
src/tabpfn_spatial/
  sim/          the plants (2D grid + 3D: plant3d, gas3d), gas physics, scenarios  (Steven)
  source/       find the leak: position, size, height (TabPFN)                     (Rayan)
  hazard/       predict the gas: toxic / fire probability, 2D and 3D (TabPFN)      (Ahmad)
  evac/         route everyone: crowd, planners (nearest, risk, CCRP), the loop,
                final 3D study (final_study, final_compare)                        (Steven)
  study3d/      real data vs simulation, physics ladder up to CFD (3D study)
  world/        the 3D viewer (Three.js) and its server
  core/         TabPFN backends (cloud / k-NN stand-in), budget guard, metrics
  contracts.py  the interfaces between the three steps
tests/          mirrors src/ (runs in CI on every PR)
docs/
  brand/        the Via Libera logo
  results/      final tables (the numbers behind the pitch)
  figures/      final/ (pitch figures) · shapes/ · 3d-study/ · steven/ · rayan/ · ahmad/
  research/     final-factory.md (3D inputs), unseen-shapes.md, 3D study notes
  journey/      how each part evolved;  status/  per-person status
outputs/        (git-ignored) runs: final/, world/ (replays + videos), pipeline/ (2D)
milestones/     (git-ignored) galleries from `uv run gallery <name>`
```

Contribution rules (branches, PRs, ownership, code style) are in [AGENTS.md](AGENTS.md).

## Note on tooling

We used AI coding assistants (Claude Code, Codex) while building this project. All code was
reviewed, tested and integrated by the team.
