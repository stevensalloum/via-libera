Closes #

## What

<!-- One or two sentences: what changes. -->

## Why

<!-- The problem or task this solves. -->

## How tested

<!-- New or changed tests; output of `uv run pytest`; manual checks if any. -->

## Checklist

- [ ] Rebased on latest `main`
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pytest` passes
- [ ] Stayed inside my own folder, or this PR changes a shared interface (say which)
- [ ] Updated my `docs/status/<name>.md`
- [ ] No secrets, datasets or large files
