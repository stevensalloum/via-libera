# tabpfn-spatial

TabPFN for spatial awareness in robotics. Built at the Berlin AIxRobotics Hackathon
(Prior Labs track).

> Project description, results and demo instructions will be filled in during the hackathon.

## Team

- Steven ([@stevensalloum](https://github.com/stevensalloum)), integrator
- Rayan ([@RayanElKhoury](https://github.com/RayanElKhoury))
- Ahmad ([@Asorus](https://github.com/Asorus))

## Setup (once per laptop)

Requires Python 3.12+. Recommended: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12
itself if missing). Clone outside iCloud / OneDrive folders.

```bash
gh repo clone stevensalloum/tabpfn-spatial
cd tabpfn-spatial
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

## How to run

```bash
uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff format .          # format
```

Entry points for the demo will be added here.

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
  data/      simulation and datasets
  models/    TabPFN wrappers, features, evaluation
  demo/      agent, visuals, demo
tests/       pytest + hypothesis suite (fast, runs in CI on every PR)
docs/
  decisions.md   design decisions log
  status/        per-person status files
```

Contribution rules (branches, PRs, ownership, code style) are in [AGENTS.md](AGENTS.md).

## Note on tooling

We used AI coding assistants (Claude Code, Codex) while building this project. All code was
reviewed, tested and integrated by the team.
