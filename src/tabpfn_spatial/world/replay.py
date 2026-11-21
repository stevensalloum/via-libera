"""A whole closed-loop run packed for the 3D viewer (``uv run world --replay``).

One leak, one crowd, the same alarm and model updates, every planner. Packed small:
- gas: true concentration every ``gas_every`` s as log10 in [GAS_LOG_MIN, GAS_LOG_MAX]
  quantised to one byte per cell (base64);
- updates: at each model update, the danger map the planners used (max predicted P(harm)
  over the next 60 s, one byte per cell) and the locator's top 3 guesses;
- people: for every planner and every second, each person's cell (int16 pairs) and state
  (0 walking or waiting for the alarm, 1 queueing, 2 standing in harmful gas, 3 out).
"""

import base64

import numpy as np

from tabpfn_spatial.evac.crowd import CrowdParams, sample_workers
from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.crowd_run import make_models, prepare, run_crowd
from tabpfn_spatial.evac.crowd_study import Rung
from tabpfn_spatial.sim import sample_scenarios
from tabpfn_spatial.world.export import scene

GAS_LOG_MIN, GAS_LOG_MAX = -3.0, 1.0  # 0.001 .. 10 (toxic limit 0.1 = -1)


def encode_u8(a: np.ndarray) -> str:
    """Base64 of an array cast to uint8."""
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def decode_u8(s: str, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of ``encode_u8``."""
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).reshape(shape)


def _gas_byte(field: np.ndarray) -> np.ndarray:
    lg = np.log10(np.maximum(field, 10**GAS_LOG_MIN))
    q = (lg - GAS_LOG_MIN) / (GAS_LOG_MAX - GAS_LOG_MIN) * 255
    return np.clip(np.round(q), 0, 255) * (field > 10**GAS_LOG_MIN)


def _track(prep, frames: list[dict], k_end: int) -> tuple[str, str]:
    """(cells int16 base64, state uint8 base64) for seconds 0..k_end."""
    cells = np.stack([fr["cells"] for fr in frames[: k_end + 1]]).astype(np.int16)
    state = np.zeros(cells.shape[:2], dtype=np.uint8)
    for k, fr in enumerate(frames[: k_end + 1]):
        in_gas = prep.harm[fr["k"]][fr["cells"][:, 0], fr["cells"][:, 1]]
        state[k] = np.where(~fr["active"], 3, np.where(in_gas, 2, np.where(fr["waiting"], 1, 0)))
    return base64.b64encode(cells.tobytes()).decode(), encode_u8(state)


def make_replay(
    rung: Rung, seed: int, models: str = "tabpfn", backend: str = "baseline", gas_every: float = 2.0
) -> dict:
    """Run every planner on one leak and pack it. Cost: one closed-loop run per planner."""
    f = rung.plant()
    rng = np.random.default_rng(seed)
    sc = sample_scenarios(f, 1, rng)[0]
    people = sample_workers(f, rung.people, rng)
    fitted = make_models(f, sc, models, backend) if models == "tabpfn" else None
    prep = prepare(f, sc, models, rung.t_max, rung.update_s, seed, fitted=fitted, backend=backend)
    params = CrowdParams(door_m=rung.door_m)
    runs = {p: run_crowd(f, prep, people, p, params, record_every=1) for p in PLANNERS}
    ends = [r.metrics["out_all_s"] for r in runs.values() if np.isfinite(r.metrics["out_all_s"])]
    k_end = int(min(rung.t_max, (max(ends) if ends else rung.t_max) + 10))
    gas_t = np.arange(0.0, k_end + 1e-9, gas_every)
    updates = [
        {
            "t": float(t),
            "danger": encode_u8(np.round(d * 255)),
            **{k: v for k, v in log.items() if k != "t"},
        }
        for t, d, log in zip(prep.replan_times, prep.danger, prep.log, strict=True)
        if t <= k_end
    ]
    planners = {}
    for p, r in runs.items():
        cells, state = _track(prep, r.frames, k_end)
        last = r.frames[min(k_end, len(r.frames) - 1)]
        planners[p] = {"cells": cells, "state": state, "metrics": r.metrics,
                       "exit": last["exit"].tolist()}  # fmt: skip
    return {
        "scene": scene(f),
        "rung": f"{rung.name}: {rung.label}",
        "seed": seed,
        "models": models if models != "tabpfn" else f"tabpfn ({backend})",
        "leak": {"xy": sc.source.tolist(), "rate": float(sc.rate)},
        "people": len(people),
        "t_alarm": prep.t_alarm,
        "t_end": float(k_end),
        "times": list(range(k_end + 1)),
        "gas": {
            "every": gas_every,
            "times": gas_t.tolist(),
            "log_min": GAS_LOG_MIN,
            "log_max": GAS_LOG_MAX,
            "frames": [encode_u8(_gas_byte(prep.fields[int(t)])) for t in gas_t],
        },
        "updates": updates,
        "planners": planners,
    }
