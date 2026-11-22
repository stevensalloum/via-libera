"""Task S on Red:Vapor: where is the leak, from fixed detectors? Both transfer workflows.

Detectors: a 5 x 5 horizontal grid (x, y in -0.9..0.9 m) at two heights (0.10 and 0.42 m,
i.e. near the ground and at the release level of most equipment), 50 in all; detectors
inside a building read NaN.

- **Real rows** (8, one per experiment): each detector reads the PID excess of the nearest
  measured voxel (within 0.10 m horizontally, 0.06 m vertically; NaN otherwise).
- **Simulated rows:** 600 random leaks per experiment at equipment heights (0.08-0.40 m),
  same detectors, read from the source-receptor sensitivity (``physics.adjoint``) of the
  calibrated L3 model with that experiment's measured wind.

Features: readings / the strongest reading (release rates are unknown, so only the pattern
counts) + wind speed. Targets: leak x and y (m). Score: horizontal error (m).

Workflows (leave-one-setup-out): ``real->real`` (context: the real rows of the other
setups, at most 6 rows from 3 leak positions), ``A real->sim`` (same context, tested on the
held-out setup's simulated leaks), ``B same-site`` (context: simulated leaks of the target
site), ``B other-sites`` (simulated leaks of the other setups). References without TabPFN:
``strongest detector`` (its position) and ``physics inversion`` (the simulated leak of the
target site whose detector pattern is most similar, cosine).
"""

import argparse
import json

import numpy as np

from tabpfn_spatial.core.models import make_regressor
from tabpfn_spatial.study3d import cases as cs
from tabpfn_spatial.study3d import physics as ph
from tabpfn_spatial.study3d.ladder import OUT

XY = np.linspace(-0.9, 0.9, 5)
DETECTORS = np.array([(x, y, z) for z in (0.10, 0.42) for x in XY for y in XY])
LEAK_Z = (0.08, 0.40)
N_LEAKS = 600
K_CAL = 8.0  # turbulence factor chosen by the leave-one-setup-out calibration


def real_readings(c: cs.Case) -> np.ndarray:
    """(50,) PID excess at each detector from the nearest measured voxel, NaN if none."""
    out = np.full(len(DETECTORS), np.nan)
    for k, d in enumerate(DETECTORS):
        dh = np.hypot(*(c.xyz[:, :2] - d[:2]).T)
        dz = np.abs(c.xyz[:, 2] - d[2])
        m = (dh <= 0.10) & (dz <= 0.06)
        if m.any():
            out[k] = c.obs[m][np.argmin(dh[m] + dz[m])]
    return out


def simulated(c: cs.Case, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(N_LEAKS, 50) readings and (N_LEAKS, 3) leak positions for a case's site and wind."""
    f = c.flow
    f = ph.Flow(f.direction, f.z, f.speed, f.sigma_v * K_CAL, f.sigma_w * K_CAL, f.updraft,
                f.l_max, f.open_faces)  # fmt: skip
    w = ph.mass_consistent(c.grid, c.solid, f, ph.base_wind(c.grid, c.solid, f))
    adj = ph.adjoint(c.grid, c.solid, f, w, DETECTORS)
    g = c.grid
    xs, ys, zs = g.axis(0), g.axis(1), g.axis(2)
    cand = np.argwhere(~c.solid)
    keep = ((zs[cand[:, 2]] >= LEAK_Z[0]) & (zs[cand[:, 2]] <= LEAK_Z[1])
            & (np.abs(xs[cand[:, 0]]) <= 1.0) & (np.abs(ys[cand[:, 1]]) <= 1.0))  # fmt: skip
    cand = cand[keep][rng.choice(keep.sum(), N_LEAKS, replace=False)]
    readings = adj[:, cand[:, 0], cand[:, 1], cand[:, 2]].T
    pos = np.c_[xs[cand[:, 0]], ys[cand[:, 1]], zs[cand[:, 2]]]
    return readings, pos


def features(readings: np.ndarray, speed: float) -> np.ndarray:
    """Readings / the row's strongest reading (NaN kept), plus the wind speed."""
    r = np.clip(readings, 0, None)
    peak = np.nanmax(r, axis=1, keepdims=True)
    return np.c_[r / np.where(peak > 0, peak, 1.0), np.full(len(r), speed)]


def _cosine_inversion(lib_x: np.ndarray, lib_pos: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    a = np.nan_to_num(lib_x[:, ok])
    b = np.nan_to_num(x[ok])
    sim = a @ b / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-12)
    return lib_pos[np.argmax(sim), :2]


def run(backend: str, seed: int = 0) -> list[dict]:
    """Every workflow for every held-out experiment; returns rows with errors (m)."""
    rng = np.random.default_rng(seed)
    cases = cs.redvapor_cases()
    speed = {c.name: float(c.flow.at("speed", c.source[2])) for c in cases}
    real = {c.name: features(real_readings(c)[None], speed[c.name])[0] for c in cases}
    sim = {}
    for c in cases:
        r, pos = simulated(c, rng)
        sim[c.name] = (features(r, speed[c.name]), pos)
        print("simulated", c.name, flush=True)
    rows = []

    def tabpfn(xc: np.ndarray, pc: np.ndarray, xt: np.ndarray) -> np.ndarray:
        out = [make_regressor(backend).fit(xc, pc[:, k]).predict_dist(xt).mean for k in (0, 1)]
        return np.c_[out[0], out[1]]

    for setup in sorted({c.info["setup"] for c in cases}):
        held = [c for c in cases if c.info["setup"] == setup]
        others = [c for c in cases if c.info["setup"] != setup]
        x_real = np.vstack([real[o.name] for o in others])
        p_real = np.vstack([o.source[None] for o in others])
        x_test_real = np.vstack([real[c.name] for c in held])
        x_test_sim = np.vstack([sim[c.name][0][:100] for c in held])
        p_test_sim = np.vstack([sim[c.name][1][:100] for c in held])
        pred = tabpfn(x_real, p_real, np.vstack([x_test_real, x_test_sim]))
        pr, ps = pred[: len(held)], pred[len(held) :]
        x_oth = np.vstack([sim[o.name][0] for o in others])
        p_oth = np.vstack([sim[o.name][1] for o in others])
        po = tabpfn(x_oth, p_oth, x_test_real)
        err_a = np.hypot(*(ps - p_test_sim[:, :2]).T)
        for k, c in enumerate(held):
            truth = c.source[:2]
            ps_same = tabpfn(*sim[c.name], real[c.name][None])[0]
            peak = int(np.nanargmax(np.nan_to_num(real[c.name][:-1], nan=-1)))
            res = {
                "real->real": pr[k],
                "B same-site": ps_same,
                "B other-sites": po[k],
                "strongest detector": DETECTORS[peak, :2],
                "physics inversion": _cosine_inversion(
                    sim[c.name][0][:, :-1], sim[c.name][1], real[c.name][:-1]
                ),
            }
            for name, p in res.items():
                rows.append({"case": c.name, "setup": setup, "workflow": name,
                             "backend": backend if "->" in name or "B " in name else "-",
                             "error_m": float(np.hypot(*(p - truth))),
                             "pred": [float(v) for v in p]})  # fmt: skip
            a_rows = err_a[k * 100 : (k + 1) * 100]
            rows.append({"case": c.name, "setup": setup, "workflow": "A real->sim",
                         "backend": backend, "error_m": float(np.median(a_rows)),
                         "error_p90_m": float(np.percentile(a_rows, 90))})  # fmt: skip
            print(c.name, " ".join(f"{r['workflow']}={r['error_m']:.2f}" for r in rows[-6:]),
                  flush=True)  # fmt: skip
    return rows


def main() -> int:
    """CLI: ``--backend baseline|cloud``."""
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="baseline")
    a = p.parse_args()
    rows = run(a.backend)
    out = OUT / f"source_redvapor_{a.backend}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
