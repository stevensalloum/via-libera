"""End-to-end emergency: leak -> readings -> source -> hazard -> evacuation, scored on truth.

Plug in any implementation of the three ``contracts`` protocols; defaults are the
baselines. ``evaluate`` averages the scores over random leaks; ``uv run leak-demo``
prints them and draws one episode. Integrator-owned.

Scores (lower is better except where noted):
- ``source_error_m``: metres from the most likely cell to the true leak
- ``hazard_brier``: Brier score of P(toxic or fire) vs the true simulation, all forecast times
- ``exposed_plan`` / ``exposed_standard``: people whose route crosses a truly harmful cell
  when they reach it, with our plan vs the conventional nearest-exit plan
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import (
    EvacPlan,
    EvacPlanner,
    HazardMap,
    HazardPredictor,
    SensorReadings,
    SourceEstimate,
    SourceLocator,
)
from tabpfn_spatial.evac import RiskAwarePlanner, route_exposed
from tabpfn_spatial.hazard import ForwardSimPredictor, brier, hazard_from_fields
from tabpfn_spatial.sim import (
    Factory,
    Scenario,
    readings,
    sample_people,
    sample_scenarios,
    simulate,
)
from tabpfn_spatial.source import WeightedCentroidLocator, localisation_error

T_OBS = 60.0  # seconds of sensor data before the system acts
HORIZON = 120.0  # forecast this far past T_OBS
STEP = 10.0


@dataclass(frozen=True)
class Episode:
    """Everything one emergency produced, plus its scores."""

    scenario: Scenario
    people: np.ndarray
    readings: SensorReadings
    source: SourceEstimate
    hazard: HazardMap
    plan: EvacPlan
    standard: EvacPlan
    truth_times: np.ndarray
    truth_harm: np.ndarray  # (T, ny, nx) bool
    source_error_m: float
    hazard_brier: float
    exposed_plan: int
    exposed_standard: int


def run_episode(
    factory: Factory,
    scenario: Scenario,
    people: np.ndarray,
    locator: SourceLocator | None = None,
    predictor: HazardPredictor | None = None,
    planner: EvacPlanner | None = None,
    rng: np.random.Generator | None = None,
    t_obs: float = T_OBS,
) -> Episode:
    """Run and score one emergency. Cost: two simulations + the plugged-in models."""
    rng = rng or np.random.default_rng()
    locator = locator or WeightedCentroidLocator()
    predictor = predictor or ForwardSimPredictor()
    planner = planner or RiskAwarePlanner()
    r = readings(factory, scenario, t_obs, rng)
    est = locator.locate(factory, r)
    times = t_obs + np.arange(STEP, HORIZON + STEP, STEP)
    hz = predictor.predict(factory, r, est, times)
    truth_times = np.concatenate([[t_obs], times])
    toxic, fire = hazard_from_fields(factory, simulate(factory, [scenario], truth_times)[0])
    harm = toxic | fire
    plan = planner.plan(factory, people, hz, t_obs)
    zero = np.zeros_like(hz.p_toxic)
    standard = RiskAwarePlanner().plan(factory, people, HazardMap(times, zero, zero), t_obs)

    def exposed(p: EvacPlan) -> int:
        return sum(route_exposed(factory, rt, truth_times, harm, t_obs) for rt in p.routes)

    danger = 1 - (1 - hz.p_toxic) * (1 - hz.p_fire)
    return Episode(
        scenario,
        people,
        r,
        est,
        hz,
        plan,
        standard,
        truth_times,
        harm,
        localisation_error(factory, est, scenario.source),
        brier(danger, harm[1:]),
        exposed(plan),
        exposed(standard),
    )


def evaluate(
    factory: Factory, n_episodes: int = 20, n_people: int = 30, seed: int = 0, **components
) -> dict[str, float]:
    """Mean scores over random leaks (same seeds -> comparable across implementations)."""
    rng = np.random.default_rng(seed)
    eps = [
        run_episode(factory, sc, sample_people(factory, n_people, rng), rng=rng, **components)
        for sc in sample_scenarios(factory, n_episodes, rng)
    ]
    keys = ("source_error_m", "hazard_brier", "exposed_plan", "exposed_standard")
    out = {k: float(np.mean([getattr(e, k) for e in eps])) for k in keys}
    out["episodes"] = n_episodes
    return out


def plot_episode(factory: Factory, ep: Episode, path: Path) -> Path:
    """Three panels: true plume + sensors, source map, hazard + routes. Saves a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = (0, factory.width, 0, factory.height)
    solid = np.ma.masked_where(~factory.walls, factory.walls)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    panels = [
        ("truth at t_obs: harmful cells", ep.truth_harm[0].astype(float), "Reds"),
        ("source estimate P(leak here)", ep.source.prob, "viridis"),
        (
            f"P(harm) at t={ep.hazard.times[5]:.0f}s, routes",
            1 - (1 - ep.hazard.p_toxic[5]) * (1 - ep.hazard.p_fire[5]),
            "magma",
        ),
    ]
    for ax, (title, img, cmap) in zip(axes, panels, strict=True):
        ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0)
        ax.imshow(solid, origin="lower", extent=ext, cmap="gray_r", alpha=0.9)
        ax.scatter(*ep.scenario.source, marker="*", s=220, c="cyan", edgecolors="k", zorder=5)
        ax.set_title(title, fontsize=10)
    axes[0].scatter(*factory.sensors.T, marker="s", s=25, c="white", edgecolors="k")
    for m in factory.machines:
        axes[0].text(
            (m.x0 + m.x1) / 2,
            (m.y0 + m.y1) / 2,
            m.name + (" *" if m.hot else ""),
            ha="center",
            va="center",
            fontsize=6,
            color="black",
        )
    for plan, colour in ((ep.standard, "red"), (ep.plan, "lime")):
        for rt in plan.routes:
            xy = (rt.cells[:, ::-1] + 0.5) * factory.cell
            axes[2].plot(*xy.T, c=colour, lw=1, alpha=0.7)
    axes[2].scatter(*ep.people.T, s=10, c="white", edgecolors="k", zorder=6)
    axes[2].scatter(*factory.exits.T, marker="D", s=50, c="yellow", edgecolors="k", zorder=6)
    fig.suptitle(
        f"source error {ep.source_error_m:.1f} m | hazard Brier {ep.hazard_brier:.3f} | "
        f"exposed: plan {ep.exposed_plan} vs standard {ep.exposed_standard} "
        f"of {len(ep.people)} (green = plan, red = standard)",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> int:
    """CLI: ``uv run leak-demo [--seed S] [--episodes N]``."""
    p = argparse.ArgumentParser(description="Run the leak pipeline with the baselines.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()
    f = Factory.default()
    rng = np.random.default_rng(args.seed)
    sc = sample_scenarios(f, 1, rng)[0]
    ep = run_episode(f, sc, sample_people(f, 30, rng), rng=rng)
    print("figure:", plot_episode(f, ep, Path("outputs/leak_demo.png")))
    print("baselines over", args.episodes, "episodes:", evaluate(f, args.episodes, seed=args.seed))
    return 0
