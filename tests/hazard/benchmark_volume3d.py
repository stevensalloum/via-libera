"""Small paired 3D benchmark with known synthetic sources; opt-in cloud spending.

This tests neither Rayan's future 3D locator nor real factory safety. The training
and evaluation use separate random scenarios but the same synthetic geometry and
transport model. No thresholds are calibrated from these test outcomes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tabpfn_spatial.contracts import TOXIC_LIMIT
from tabpfn_spatial.core.budget import cloud_quote
from tabpfn_spatial.env import load_env
from tabpfn_spatial.hazard import (
    ForwardVolumePredictor,
    MachineBox3D,
    TabPFNVolumePredictor,
    VolumeGrid,
)
from tabpfn_spatial.hazard.volume3d import simulate_volume

TRAIN_SEED = 20261210
TEST_SEED = 20261211
TIMES = np.array([5.0, 15.0, 30.0])


def example_layout() -> tuple[VolumeGrid, np.ndarray]:
    """One explicitly synthetic 24 x 16 x 4 m room and two-height source sites."""
    walls = np.zeros((4, 8, 12), bool)
    machines = [MachineBox3D((10.0, 4.0, 0.0), (14.0, 10.0, 2.5), hot=True)]
    grid = VolumeGrid.from_layout(walls, machines, cell=2.0, dz=1.0, ignition_radius=1.5)
    candidates = np.zeros_like(walls)
    candidates[[0, 2], 2, 2] = True
    candidates[[0, 2], 5, 8] = True
    return grid, candidates


def make_cases(grid, candidates, count: int, seed: int) -> list[dict]:
    """Draw independent test scenarios; the true 3D source is supplied as an oracle."""
    if seed == TRAIN_SEED:
        raise ValueError("test seed must differ from training seed")
    rng = np.random.default_rng(seed)
    choices = grid.centres()[candidates]
    cases = []
    for index in range(count):
        source = choices[rng.integers(len(choices))]
        rate = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        speed, angle = rng.uniform(0.2, 1.5), rng.uniform(0, 2 * np.pi)
        wind = speed * np.array([np.cos(angle), np.sin(angle)])
        diffusion = float(np.exp(rng.uniform(np.log(0.4), np.log(1.6))))
        cases.append(dict(index=index, source=source, rate=rate, wind=wind, diffusivity=diffusion))
    return cases


def score(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Brier and toxic counts at the preselected 0.5 comparison threshold."""
    predicted, truth = np.asarray(predicted), np.asarray(truth, dtype=bool)
    toxic = predicted >= 0.5
    return dict(
        brier=float(np.mean((predicted - truth) ** 2)),
        tp=int((toxic & truth).sum()),
        fn=int((~toxic & truth).sum()),
        fp=int((toxic & ~truth).sum()),
        tn=int((~toxic & ~truth).sum()),
    )


def run(args) -> None:
    """Fit once and compare paired cases; each cloud prediction spends credits."""
    if args.output.exists():
        raise FileExistsError(args.output)
    grid, candidates = example_layout()
    model = TabPFNVolumePredictor(backend=args.backend, top_k=1).fit(
        grid,
        candidates,
        n_scenarios=args.train,
        cells_per_scenario=args.cells,
        times=TIMES,
        seed=TRAIN_SEED,
        diffusivity_range=(0.4, 1.6),
    )
    forward = ForwardVolumePredictor(diffusivity=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for case in make_cases(grid, candidates, args.cases, TEST_SEED):
            source, wind, rate = case["source"], case["wind"], case["rate"]
            truth = (
                simulate_volume(grid, source, rate, wind, TIMES, case["diffusivity"]) > TOXIC_LIMIT
            )
            predictions = {}
            for name, predictor in (("forward", forward), ("tabpfn", model)):
                start = time.perf_counter()
                result = predictor.predict(grid, source[None], np.array([1.0]), wind, rate, TIMES)
                elapsed = time.perf_counter() - start
                predictions[name] = {**score(result.p_toxic[:, ~grid.solid], truth[:, ~grid.solid]),
                                     "seconds": elapsed}  # fmt: skip
            record = dict(
                index=case["index"],
                source=source.tolist(),
                rate=rate,
                wind=wind.tolist(),
                diffusivity=case["diffusivity"],
                train_seed=TRAIN_SEED,
                test_seed=TEST_SEED,
                backend=args.backend,
                train_scenarios=args.train,
                cells_per_scenario=args.cells,
                forward=predictions["forward"],
                tabpfn=predictions["tabpfn"],
            )
            print(json.dumps(record), file=stream, flush=True)
            print(json.dumps(record), flush=True)
    guard = getattr(model.model, "guard", None)
    print("quoted credits", getattr(guard, "spent", 0), flush=True)


def main() -> None:
    """Benchmark only on explicit invocation; quote-only sends shapes and no data."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("baseline", "cloud"), default="baseline")
    p.add_argument("--run-cloud", action="store_true")
    p.add_argument("--quote-only", action="store_true")
    p.add_argument("--train", type=int, default=80)
    p.add_argument("--cells", type=int, default=20)
    p.add_argument("--cases", type=int, default=12)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.train, args.cells, args.cases) < 1:
        p.error("train, cells and cases must be positive")
    if args.quote_only:
        load_env()
        grid, _ = example_layout()
        per = cloud_quote(
            args.train * args.cells * len(TIMES), int((~grid.solid).sum()) * len(TIMES), 9
        )
        print(f"per cloud call {per:,}; {args.cases} calls {per * args.cases:,}")
        return
    if args.backend == "cloud" and not args.run_cloud:
        p.error("cloud predictions spend credits; pass --run-cloud")
    load_env()
    run(args)


if __name__ == "__main__":
    main()
