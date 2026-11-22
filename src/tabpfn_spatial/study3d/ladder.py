"""Run the physics ladder on every real case and score it (``uv run python -m ...ladder``).

Writes ``outputs/study3d/ladder_<dataset>.json`` (metrics per case and level) and
``outputs/study3d/fields_<dataset>.npz`` (each level's prediction at the observation points,
used later as the simulated side of the transfer workflows).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tabpfn_spatial.study3d import cases as cs
from tabpfn_spatial.study3d import physics as ph
from tabpfn_spatial.study3d.evaluate import score

OUT = Path("outputs/study3d")


def run_case(c: cs.Case, k_factor: float = 1.0) -> dict[str, np.ndarray]:
    """Predictions of L1, L2, L3 at the case's observation points.

    ``k_factor`` scales the turbulence (sigma_v, sigma_w) for the calibration study.
    """
    f = c.flow
    if k_factor != 1.0:
        f = ph.Flow(f.direction, f.z, f.speed, f.sigma_v * k_factor, f.sigma_w * k_factor,
                    f.updraft, f.l_max, f.open_faces)  # fmt: skip
    out = {"L1": ph.plume(c.xyz, c.source, f)}
    w0 = ph.base_wind(c.grid, c.solid, f)
    out["L2"] = ph.sample(c.grid, ph.transport(c.grid, c.solid, f, w0, c.source), c.xyz)
    w = ph.mass_consistent(c.grid, c.solid, f, w0)
    out["L3"] = ph.sample(c.grid, ph.transport(c.grid, c.solid, f, w, c.source), c.xyz)
    return out


def main() -> int:
    """CLI: ``--dataset redvapor`` (default) ``--k-factor 1.0``."""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="redvapor")
    p.add_argument("--k-factor", type=float, default=1.0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {"redvapor": cs.redvapor_cases}[a.dataset]()
    rows, fields = [], {}
    for c in cases:
        t = time.time()
        pred = run_case(c, a.k_factor)
        for level, m in pred.items():
            fields[f"{c.name}_{level}"] = m.astype(np.float32)
            rows.append({"case": c.name, "level": level, **c.info,
                         **score(c.xyz, c.obs, m, c.source, c.flow.direction)})  # fmt: skip
        print(c.name, f"{time.time() - t:.0f}s",
              " ".join(f"{r['level']} auc={r['auc_hot']:.2f} prof={r['profile_r']:.2f} "
                       f"w={r['width_ratio']:.2f}" for r in rows[-3:]), flush=True)  # fmt: skip
    tag = f"{a.dataset}" + ("" if a.k_factor == 1.0 else f"_k{a.k_factor:g}")
    (OUT / f"ladder_{tag}.json").write_text(json.dumps(rows, indent=1))
    np.savez_compressed(OUT / f"fields_{tag}.npz", **fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
