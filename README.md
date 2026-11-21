# tabpfn-spatial

TabPFN for spatial awareness in robotics. Built at the Berlin AIxRobotics Hackathon
(Prior Labs track).

> Project description, results and demo instructions will be filled in during the hackathon.

## Team

- Steven ([@stevensalloum](https://github.com/stevensalloum)), integrator
- Rayan ([@RayanElKhoury](https://github.com/RayanElKhoury))
- Ahmad ([@Asorus](https://github.com/Asorus))

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/stevensalloum/tabpfn-spatial.git
cd tabpfn-spatial
uv sync
cp .env.example .env   # then fill in any keys; .env is gitignored
```

## How to run

```bash
uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff format .          # format
```

Entry points for the demo will be added here.

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

Contribution rules (branches, PRs, ownership, code style) are in [CLAUDE.md](CLAUDE.md).

## Note on tooling

We used AI coding assistants (Claude Code) while building this project. All code was
reviewed, tested and integrated by the team.
