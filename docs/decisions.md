# Design decisions

Newest first. Each entry: date, decision, why, alternatives considered.

## 2026-11-21: Protected main, squash merges

- **Decision:** `main` requires a PR and a passing, up-to-date `ruff + pytest` check,
  enforced for admins too. Squash merge only; head branches auto-delete.
- **Why:** one reviewed, green commit per task keeps `main` always demo-ready.
- **Alternatives:** required approvals (blocks the integrator's own PRs in a 3-person team).

## 2026-11-21: Repository and tooling

- **Decision:** single Python package `tabpfn_spatial` (src layout), uv for environments,
  pytest + hypothesis for tests, ruff for lint and format, GitHub Actions CI on every PR.
- **Why:** one environment for the whole team, fast installs, one command to check a PR.
- **Alternatives:** separate packages per area (more friction at a 1-day hackathon);
  pip + requirements.txt (slower, no lock file).

## 2026-11-21: Ownership by folder

- **Decision:** `data/`, `models/` and `demo/` subpackages, one owner each; shared files
  are integrator-owned and change via reviewed PRs.
- **Why:** three people working in parallel with Claude Code; folder ownership keeps
  merge conflicts rare.
