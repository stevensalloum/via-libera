# AGENTS.md: shared rules for every coding agent on this team

This file applies to every agent we use (Claude Code reads it via `CLAUDE.md`, Codex reads it
directly). Read it at the start of every session. Only the integrator (Steven) edits it;
propose changes in your status file or a PR description instead.

## Project

**Gas-leak emergencies in a factory, solved with TabPFN** (Berlin AIxRobotics Hackathon,
Prior Labs track). A leak starts; from sparse sensor readings and the wind we (1) locate the
leak, (2) predict where it becomes toxic or flammable, (3) route every person out on the
safest path. Full brief, targets and milestones: **`docs/brief.md`** (read it first); what every file
and function does: **`docs/guide.md`**; datasets, seeds and the closed loop: **`docs/data.md`**.
Python >= 3.12, package `tabpfn_spatial` under `src/`, tests under `tests/`.
Environment: `uv sync`; run everything through `uv run`. `uv run leak-demo` runs the whole
pipeline and draws `outputs/leak_demo.png`. Setup and daily flow: README.

## Ownership

Each person's agent stays inside its own folders unless its issue says otherwise.

| Area | Code | Tests | Owner |
|---|---|---|---|
| Simulator (factory, gas, scenarios) | `src/tabpfn_spatial/sim/` | `tests/sim/` | Steven |
| Leak source detection | `src/tabpfn_spatial/source/` | `tests/source/` | Rayan |
| Hazard prediction | `src/tabpfn_spatial/hazard/` | `tests/hazard/` | Ahmad |
| Evacuation planning | `src/tabpfn_spatial/evac/` | `tests/evac/` | Steven |
| Shared: `contracts.py`, `pipeline.py`, `core/`, `env.py`, `tabpfn_check.py`, `pyproject.toml`, `uv.lock`, `requirements.txt`, `AGENTS.md`, `CLAUDE.md`, `.claude/`, `.github/`, `docs/brief.md`, `docs/decisions.md` | | | Steven (integrator) |

- **Contracts are the API between areas.** `src/tabpfn_spatial/contracts.py` defines
  `SensorReadings`, `SourceEstimate`, `HazardMap`, `EvacPlan` and one protocol per area.
  Implement your protocol; never change another area's code. A contract change is its own
  small PR labelled `interface`, reviewed by the integrator, and announced in your status.
- **Build on `tabpfn_spatial.core`** (`make_regressor` / `make_classifier`, `QuantileDist`,
  `Grid2D`, metrics, `CostGuard`) instead of calling TabPFN directly.
- **Keep the baseline.** Your TabPFN version is a new class next to the baseline; the
  pipeline compares both on the same seeds (`pipeline.evaluate(..., locator=...)`).
- New dependencies go through the integrator (they touch shared files).

## How agents communicate

There is no chat between agents; GitHub and the repo are the channel. At the start of
every session: `git switch main && git pull`, read `docs/brief.md`, every
`docs/status/*.md`, and your open issues (`gh issue list --assignee @me`).

- **Need something from another area** (a field in a contract, a simulator feature, a
  bug): open an issue assigned to that owner with label `needs`, describing exactly what
  and why; add it under "Needs" in your status file. Work around it with a stub meanwhile.
- **Status file** `docs/status/<name>.md` (Doing / Done / Needs / Blocked, newest first,
  with numbers): update it in every PR and at least every two hours. Your agent writes
  only its own person's file.
- **Results** (every measured number) go in your status file and the PR description:
  metric, baseline value, your value, settings (episodes, seed, backend, tokens spent).
- **Blocked for more than 20 minutes**: say so in your status and ping Steven.

## Workflow

- `main` is protected: never push to it directly. Every change lands via a pull request.
- One GitHub Issue per task, assigned to its owner. The PR links it (`Closes #12`).
- Short-lived branches from fresh `main`: `feat/<area>-<topic>` (e.g. `feat/source-tabpfn`),
  `fix/...`, `docs/...`. Merge within 1 to 2 hours, then delete the branch.
- Pull `main` at least every hour (`git pull --rebase origin main`) and before every PR.
- The integrator reviews and merges every 1 to 2 hours or when a PR is ready.
  Only the integrator merges. PRs are squash-merged (the PR title becomes the commit,
  so give it a conventional title) and the branch is deleted on GitHub automatically.
  After a merge: `git switch main && git pull && git branch -d <branch>`.
- GitHub enforces this: pushes to `main` are rejected and the `ruff + pytest` check
  must pass on an up-to-date branch before merging.

### Before opening a PR

1. `git fetch origin && git rebase origin/main`
2. `uv run ruff check . && uv run ruff format --check . && uv run pytest`
3. Update your own `docs/status/<name>.md`.
4. Fill in the PR template (what, why, how tested) and link the issue.

## Code rules

- Test-driven development: write or extend the test first, then the code.
  The suite stays green and fast (whole run under about 30 s). Mark anything slow
  or needing model weights / GPU so it is skipped by default.
- Modern, typed, concise Python: full type hints, dataclasses or typed dicts for
  structured data, no bloated functions (split anything over ~40 lines).
- State time and compute complexity in the docstring where it matters
  (e.g. `O(n log n)` in samples, TabPFN context size limits, MPS vs CPU).
- Portable across macOS, Linux and Windows: pick the torch device at runtime
  (CUDA, then MPS, then CPU); never hard-code `"mps"` or `"cuda"`. Use `pathlib`, not
  string paths with `/` or `\\`.
- Document every module and public function (docstrings are enforced by ruff `D`).
  Important design decisions go in `docs/decisions.md` (date, decision, why, alternatives).

## Commits and PRs

- Small commits in conventional format: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`,
  `chore:`, `ci:`. Written as a human developer would write them.
- Keep AI co-author trailers: commits written with an agent end with its
  `Co-Authored-By:` line (Claude Code adds it by default; do not turn that off).
  If your agent does not add one (check Codex), add it by hand. Squash merges keep the
  trailers in `main`. No "generated by" banners inside source code.

## TabPFN API budget

- Every cloud call costs at least 10,000 tokens (about 5,000 calls per 50M key); cost grows
  mainly with test rows. Batch queries into few large `predict` calls, never one call per
  point in a loop. Quote big runs first with `tabpfn_client.estimate_cost` (free).
- Tests and CI never call the API: put cloud calls behind a small interface and test with
  a fake (see `tests/test_tabpfn_check.py`). Mark any opt-in live test so it is skipped
  by default.

## Secrets and data

- Never commit secrets, API keys, datasets, model weights or large binaries.
- Keys go in `.env` (gitignored); document variable names in `.env.example`.
- Data lives in `data/`, generated artifacts in `outputs/`, weights in `checkpoints/`;
  all gitignored. Training tables come from the simulator: write the function that
  generates them (with a seed), never commit the table.
