from pathlib import Path

import numpy as np

from tabpfn_spatial.pipeline import evaluate, plot_episode, run_episode
from tabpfn_spatial.sim import Factory, Scenario, sample_people


def test_episode_end_to_end_with_baselines(tmp_path: Path) -> None:
    f = Factory.default()
    rng = np.random.default_rng(0)
    sc = Scenario(np.array([35.0, 21.0]), 1.5, np.array([0.6, 0.3]))
    ep = run_episode(f, sc, sample_people(f, 12, rng), rng=rng)
    assert ep.source_error_m >= 0 and 0 <= ep.hazard_brier <= 1
    assert len(ep.plan.routes) == len(ep.standard.routes) == 12
    assert 0 <= ep.exposed_plan <= 12 and 0 <= ep.exposed_standard <= 12
    path = plot_episode(f, ep, tmp_path / "ep.png")
    assert path.exists() and path.stat().st_size > 5_000


def test_evaluate_aggregates_episodes() -> None:
    summary = evaluate(Factory.default(), n_episodes=3, n_people=8, seed=1)
    assert set(summary) >= {"source_error_m", "hazard_brier", "exposed_plan", "exposed_standard"}
    assert summary["episodes"] == 3
