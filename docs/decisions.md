# Design decisions

Newest first. Each entry: date, decision, why, alternatives considered.

## 2026-11-21: TabPFN cloud API via tabpfn-client, budget-aware

- **Decision:** use `tabpfn-client` (cloud, TabPFN-3.5) as the default backend; key in
  `TABPFN_TOKEN` via `.env`; `uv run tabpfn-check` verifies each laptop.
- **Measured:** limit shows 50M credits per key (resets 2026-12-01); every call costs at
  least 10,000 tokens; 20k+20k x 50 = 11.1k, 50k+50k x 100 = 37.8k, 1k+100k x 10 = 26.6k,
  1k+1M x 10 = 1.97M. So batch queries; avoid per-point loops.
- **Why:** no GPU or multi-GB install needed on any laptop (Rayan: Windows, Ahmad: MacBook Air).
- **Alternatives:** local `tabpfn` with PyTorch (decide after the idea is fixed).

## 2026-11-21: Keep AI co-author trailers

- **Decision:** commits keep the agent's `Co-Authored-By:` trailer; Claude Code's default
  attribution stays on. Reverses the earlier no-attribution rule and its CI check.
- **Why:** the team prefers transparent credit for agent-written code over hiding it.
- **Alternatives:** no attribution with a CI guard (dropped).

## 2026-11-21: AGENTS.md as the single rulebook

- **Decision:** team rules live in `AGENTS.md`; `CLAUDE.md` only imports it (`@AGENTS.md`).
  `.gitattributes` forces LF line endings; torch device is chosen at runtime.
- **Why:** the team uses Claude Code and Codex (which reads `AGENTS.md`), on macOS and Windows.
  One file avoids drift; LF keeps `ruff format --check` stable across OSes.
- **Alternatives:** two copies of the rules (drift); a symlink (breaks on Windows checkouts).

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
