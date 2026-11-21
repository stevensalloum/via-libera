"""Paired hazard benchmark; cloud mode spends credits on each episode.

Run ``uv run python -m tabpfn_spatial.hazard.benchmark --episodes 3``.
The default cloud setup costs approximately 10,000 credits per episode.
"""

import argparse
import copy

import numpy as np

from tabpfn_spatial.hazard.baseline import ForwardSimPredictor, hazard_from_fields
from tabpfn_spatial.hazard.tabpfn_predictor import TabPFNHazardPredictor
from tabpfn_spatial.pipeline import run_episode
from tabpfn_spatial.sim import Factory, sample_people, sample_scenarios, simulate


def binary_metrics(prob: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Brier, ten-bin ECE and tie-aware AUC for binary probabilities; O(n log n)."""
    prob, truth = np.asarray(prob).ravel(), np.asarray(truth, dtype=bool).ravel()
    bins = np.minimum((prob * 10).astype(int), 9)
    ece = sum(
        np.mean(bins == i) * abs(prob[bins == i].mean() - truth[bins == i].mean())
        for i in range(10)
        if np.any(bins == i)
    )
    order = np.argsort(prob)
    sorted_prob, y = prob[order], truth[order]
    _, first, counts = np.unique(sorted_prob, return_index=True, return_counts=True)
    positives = np.add.reduceat(y.astype(float), first)
    negatives = counts - positives
    pairs = np.sum(positives * (np.cumsum(negatives) - negatives / 2))
    denominator = positives.sum() * negatives.sum()
    return {
        "brier": float(np.mean((prob - truth) ** 2)),
        "ece": float(ece),
        "auc": float(pairs / denominator) if denominator else float("nan"),
    }


def run_benchmark(episodes: int = 3, seed: int = 0, backend: str = "cloud") -> None:
    """Print paired harm Brier and toxic scores; cloud calls use simulated data only."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    factory = Factory.default()
    predictor = TabPFNHazardPredictor(backend=backend, top_k=3).fit(factory, seed=42)
    rng = np.random.default_rng(seed)
    scenarios = sample_scenarios(factory, episodes, rng)
    predictions: list[list[np.ndarray]] = [[], []]
    truth = []
    scores: list[list[float]] = [[], []]
    for scenario in scenarios:
        people = sample_people(factory, 30, rng)
        state = copy.deepcopy(rng.bit_generator.state)
        pair = []
        for model in (ForwardSimPredictor(), predictor):
            paired_rng = np.random.default_rng()
            paired_rng.bit_generator.state = copy.deepcopy(state)
            pair.append(run_episode(factory, scenario, people, predictor=model, rng=paired_rng))
        rng.bit_generator.state = paired_rng.bit_generator.state
        fields = simulate(factory, [scenario], pair[0].hazard.times)[0]
        toxic, _ = hazard_from_fields(factory, fields)
        truth.append(toxic.ravel())
        for i, episode in enumerate(pair):
            predictions[i].append(episode.hazard.p_toxic.ravel())
            scores[i].append(episode.hazard_brier)
    target = np.concatenate(truth)
    for name, pred, harm in zip(("forward", backend), predictions, scores, strict=True):
        print(
            name,
            "harm Brier",
            float(np.mean(harm)),
            "toxic",
            binary_metrics(np.concatenate(pred), target),
        )
    guard = getattr(predictor.model, "guard", None)
    print("quoted credits:", getattr(guard, "spent", 0))


def main() -> None:
    """Run a paired benchmark; defaults to three cloud calls."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["cloud", "baseline", "local"], default="cloud")
    args = parser.parse_args()
    run_benchmark(args.episodes, args.seed, args.backend)


if __name__ == "__main__":
    main()
