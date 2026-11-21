"""Compare forward simulation, offline k-NN and cloud TabPFN on paired scenarios.

Requires ``--run-cloud`` because the cloud predictions spend API credits. The
offline k-NN chunk size is reduced to bound memory; this can affect its timing.
"""

import argparse
import copy
import time

import numpy as np

import tabpfn_spatial.core.models as model_helpers
from tabpfn_spatial.hazard import ForwardSimPredictor, TabPFNHazardPredictor
from tabpfn_spatial.hazard.benchmark import binary_metrics
from tabpfn_spatial.pipeline import run_episode
from tabpfn_spatial.sim import Factory, sample_people, sample_scenarios


class TimedPredictor:
    """Measure each predictor's complete hazard call, including feature building."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.elapsed = []

    def predict(self, factory, readings, source, times):
        start = time.perf_counter()
        result = self.predictor.predict(factory, readings, source, times)
        self.elapsed.append(time.perf_counter() - start)
        return result


def main():
    """Run five paired episodes after the caller explicitly opts in to cloud use."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-cloud", action="store_true", help="allow paid cloud calls")
    args = parser.parse_args()
    if not args.run_cloud:
        parser.error("cloud calls spend credits; pass --run-cloud to opt in")

    factory = Factory.default()
    model_helpers._CHUNK = 128  # Same k-NN calculation, lower peak memory than default.
    models = {"forward": TimedPredictor(ForwardSimPredictor())}
    for backend in ("baseline", "cloud"):
        start = time.perf_counter()
        predictor = TabPFNHazardPredictor(backend=backend, top_k=3).fit(factory, seed=12345)
        print(
            f"{backend} context generation + fit seconds: {time.perf_counter() - start:.2f}",
            flush=True,
        )
        models[backend] = TimedPredictor(predictor)

    rng = np.random.default_rng(0)
    scenarios = sample_scenarios(factory, 5, rng)
    truths = []
    predictions = {name: [] for name in models}
    briers = {name: [] for name in models}
    open_cells = ~factory.walls
    for index, scenario in enumerate(scenarios):
        people = sample_people(factory, 30, rng)
        state = copy.deepcopy(rng.bit_generator.state)
        episodes = []
        for name, model in models.items():
            episode_rng = np.random.default_rng()
            episode_rng.bit_generator.state = copy.deepcopy(state)
            episode = run_episode(factory, scenario, people, predictor=model, rng=episode_rng)
            episodes.append(episode)
            prob = episode.hazard.p_toxic[:, open_cells]
            truth = episode.truth_harm[1:, :, :][:, open_cells]
            predictions[name].append(prob.ravel())
            briers[name].append(float(np.mean((prob - truth) ** 2)))
        for episode in episodes[1:]:
            np.testing.assert_array_equal(episode.readings.values, episodes[0].readings.values)
            np.testing.assert_array_equal(episode.source.prob, episodes[0].source.prob)
        rng.bit_generator.state = episode_rng.bit_generator.state
        truths.append(episodes[0].truth_harm[1:, :, :][:, open_cells].ravel())
        print(
            "episode",
            index + 1,
            "toxic Brier",
            {k: round(v[-1], 5) for k, v in briers.items()},
            flush=True,
        )

    y = np.concatenate(truths)
    for name, timed in models.items():
        metrics = binary_metrics(np.concatenate(predictions[name]), y)
        print(
            name,
            "metrics:",
            metrics,
            "hazard seconds:",
            np.round(timed.elapsed, 2).tolist(),
            flush=True,
        )
    guard = models["cloud"].predictor.model.guard
    print("cloud quoted credits:", guard.spent, "calls:", guard.calls, flush=True)


if __name__ == "__main__":
    main()
