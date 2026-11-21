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

## Plan (agreed by the team, 2026-11-21)

### Phases

1. **Proof of concept (done):** the current simple simulator (2D advection-diffusion, one
   uniform wind vector) with baselines and the first TabPFN parts. It stays runnable as the
   baseline every later step is compared against.
2. **Realistic simulator (next):** move to a stronger gas simulator (GADEN or similar; the
   team chooses after reading). Rayan and Ahmad build on it once it is chosen. Until then
   each person reads into their own problem and the whole task.
3. **Real data only to validate:** fluid dynamics has no closed-form answer, so labelled
   training cases come from simulation; real datasets (`docs/data.md`) check the method.
4. **Factory:** randomly generated 2D outlines (`Factory.random`) for all computation. A 3D
   view is optional for the final presentation only, unless height matters (heavy gases,
   ceiling sensors).

### Steven's steps, in order

1. **Factory** simulated and viewable; navigating through it comes later.
2. **Path finder:** people positions + source map (Rayan) + hazard map (Ahmad) -> the best
   path for each person to an exit. Uses simulated stand-ins for the source and hazard
   outputs until the real parts are linked. Compare algorithms: risk-cost Dijkstra (today),
   A*, time-dependent search, many people with corridor capacity. In a smart factory the
   same planner routes robots.
3. **How people are guided out:** phones are often banned on the floor; floor LEDs and
   screens are limited; drones and ceiling rails are too costly to sell. Current choice:
   small ground guide robots with cameras (fixed in the hall and on the robot) that find
   people and lead them out, several robots so nobody is left behind. Later: how many
   robots and sensors and where, adapted to what a plant already has (retrofit or new
   build) to keep the cost down.
4. **Timing study (for the whole team):** how prediction quality changes with how long we
   wait for readings (`t_obs`) and how often we predict. The output is curves of source
   error, hazard Brier and people exposed against `t_obs` from 10 to 120 s, plus TabPFN
   latency per call. Steven simulates them; the team picks the accepted decision time:
   not so early that it is wrong, not so late that people are already exposed.
5. **TabPFN inference mode (input for Rayan and Ahmad):** measure latency and tokens of
   `fit_mode="fit_with_cache"` (the server keeps a cache of the training set, so later
   predictions skip refitting), the fast model `model_path="v3.5-fast_default"` (4
   ensemble members instead of 8), fewer `n_estimators`, and local inference on GPU/MPS.
   Source and hazard owners choose using the timing study.

## Document as you go

Every step is part of the story we present (and may publish), so record it while you
work, not at the end. Each person keeps **`docs/journey/<name>.md`** (only your own file,
newest entry at the bottom). One entry per step:

- **Date and step:** what changed and why (e.g. "uniform wind -> wind field around
  machines, because gas piled up against machine faces").
- **Figure:** `docs/figures/<name>/<step>.png`, made by a script in the repo; write the
  command next to it. Keep each PNG small (under about 500 kB).
- **Numbers:** metric, previous value, new value, settings (episodes, seed, backend, tokens).
- **What did not work** and what you learned from it.

Keep the old version runnable (a flag or a baseline class) so the before/after figures can
be redrawn at any time.

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
3. Update your own `docs/status/<name>.md` and `docs/journey/<name>.md`.
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

- Never commit secrets, API keys, datasets, model weights or large binaries. The only
  binaries allowed are the small journey figures in `docs/figures/`.
- Keys go in `.env` (gitignored); document variable names in `.env.example`.
- Data lives in `data/`, generated artifacts in `outputs/`, weights in `checkpoints/`;
  all gitignored. Training tables come from the simulator: write the function that
  generates them (with a seed), never commit the table.
