"""The two transfer workflows on Red:Vapor, task H (hazard: concentration at a point).

Leave-one-setup-out: for each experiment, every context comes from the *other* setups
(A, B, C, D), except where a workflow is explicitly allowed to simulate the target site.

Rows: one per observation point. Features (physical, shared by real and simulated rows):

| feature | meaning |
|---|---|
| s | distance downwind of the leak (m) |
| n | distance crosswind (m) |
| z | height (m) |
| dz | height above the release (m) |
| u_src | measured wind speed at release height (m/s) |
| blocked | 1 if the straight line leak -> point passes through a building |

Target: log10(c / c99 + 1e-3), c99 = 99th percentile of that field's positive values (scale-
free: relative PID and simulated units become comparable; 1e-3 = 0.1 % of the peak level).

Settings (``WORKFLOWS``):
- ``real->real``: context real, other setups; test real (the reference: what real data alone
  gives on an unseen site).
- ``physics``: the simulator's own field, no TabPFN.
- ``B same-site``: context simulated (level L) **of the target experiment** (you can
  simulate your own factory), test real.
- ``B other-sites``: context simulated of the other setups, test real.
- ``hybrid``: context real (other setups) + simulated (target site), with an ``is_sim`` flag;
  test real (``is_sim`` = 0).
- ``A real->sim``: context real (other setups), test **simulated** target (truth =
  simulator): does a model built on real data agree with the physics on a new site?
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tabpfn_spatial.core.models import make_regressor
from tabpfn_spatial.study3d import cases as cs
from tabpfn_spatial.study3d.evaluate import score
from tabpfn_spatial.study3d.geometry import Grid

OUT = Path("outputs/study3d")
FEATURES = ["s", "n", "z", "dz", "u_src", "blocked"]
EPS = 1e-3


def blocked(
    g: Grid, solid: np.ndarray, src: np.ndarray, xyz: np.ndarray, steps: int = 40
) -> np.ndarray:
    """1.0 where the segment from the leak to the point crosses a solid cell."""
    t = np.linspace(0.05, 0.95, steps)[None, :, None]
    pts = src[None, None, :] + t * (xyz[:, None, :] - src[None, None, :])
    i = g.index(pts.reshape(-1, 3))
    return solid[i[:, 0], i[:, 1], i[:, 2]].reshape(len(xyz), steps).any(1).astype(float)


def features(c: cs.Case) -> np.ndarray:
    """(n, 6) feature rows of a case's observation points."""
    d = np.asarray(c.flow.direction)
    rel = c.xyz - c.source
    s, n = rel[:, :2] @ d, rel[:, :2] @ np.array([-d[1], d[0]])
    u = np.full(len(rel), float(c.flow.at("speed", c.source[2])))
    return np.c_[s, n, c.xyz[:, 2], rel[:, 2], u, blocked(c.grid, c.solid, c.source, c.xyz)]


def target(c: np.ndarray) -> np.ndarray:
    """log10 of the concentration relative to the field's 99th percentile."""
    pos = c[c > 0]
    ref = np.percentile(pos, 99) if pos.size else 1.0
    return np.log10(np.clip(c, 0, None) / ref + EPS)


def _sub(x: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> tuple:
    if len(y) <= n:
        return x, y
    i = rng.choice(len(y), n, replace=False)
    return x[i], y[i]


def run(backend: str, level: str, k_tag: str, max_context: int, seed: int = 0) -> list[dict]:
    """All workflows for every held-out experiment; returns score rows.

    Calls are shared where the context and test rows are identical: ``real->real`` and
    ``A real->sim`` are one prediction (same context, same test points, different truth),
    and the real context of a held-out setup serves all its experiments.
    """
    cases = cs.redvapor_cases()
    fields = np.load(OUT / f"fields_redvapor{k_tag}.npz")
    X = {c.name: features(c) for c in cases}
    y_real = {c.name: target(c.obs) for c in cases}
    y_sim = {c.name: target(fields[f"{c.name}_{level}"].astype(float)) for c in cases}
    rng = np.random.default_rng(seed)
    rows = []
    for setup in sorted({c.info["setup"] for c in cases}):
        held = [c for c in cases if c.info["setup"] == setup]
        others = [o.name for o in cases if o.info["setup"] != setup]
        xr = np.vstack([X[o] for o in others])
        yr = np.concatenate([y_real[o] for o in others])
        ys_other = np.concatenate([y_sim[o] for o in others])
        xt_all = np.vstack([X[c.name] for c in held])
        split = np.cumsum([len(X[c.name]) for c in held])[:-1]

        def fit_predict(xc: np.ndarray, yc: np.ndarray, xt: np.ndarray) -> np.ndarray:
            xc, yc = _sub(xc, yc, max_context, rng)
            return make_regressor(backend).fit(xc, yc).predict_dist(xt).mean

        p_real = np.split(fit_predict(xr, yr, xt_all), split)
        p_other = np.split(fit_predict(xr, ys_other, xt_all), split)
        n_ctx = min(len(yr), max_context)
        for c, pr, po in zip(held, p_real, p_other, strict=True):
            nr = min(len(yr), max_context // 2)
            ar = _sub(xr, yr, nr, rng)
            asim = _sub(X[c.name], y_sim[c.name], max_context - nr, rng)
            xh = np.vstack(
                [np.c_[ar[0], np.zeros(len(ar[1]))], np.c_[asim[0], np.ones(len(asim[1]))]]
            )
            yh = np.concatenate([ar[1], asim[1]])
            preds = {
                "real->real": (pr, "real", n_ctx),
                "A real->sim": (pr, "sim", n_ctx),
                "B other-sites": (po, "real", n_ctx),
                "B same-site": (fit_predict(X[c.name], y_sim[c.name], X[c.name]), "real",
                                min(len(y_sim[c.name]), max_context)),
                "hybrid": (make_regressor(backend).fit(xh, yh).predict_dist(
                    np.c_[X[c.name], np.zeros(len(X[c.name]))]).mean, "real", len(yh)),
            }  # fmt: skip
            for name, (pred, truth, n_rows) in preds.items():
                ref = c.obs if truth == "real" else 10 ** y_sim[c.name]
                sc = score(c.xyz, ref, 10**pred, c.source, c.flow.direction)
                rows.append({"case": c.name, "setup": setup, "workflow": name, "level": level,
                             "k": k_tag or "k1", "backend": backend, "context_rows": n_rows,
                             **sc})  # fmt: skip
            phys = fields[f"{c.name}_{level}"].astype(float)
            rows.append({"case": c.name, "setup": setup, "workflow": "physics", "level": level,
                         "k": k_tag or "k1", "backend": "-", "context_rows": 0,
                         **score(c.xyz, c.obs, phys, c.source, c.flow.direction)})  # fmt: skip
            line = " ".join(f"{r['workflow']}: auc={r['auc_hot']:.2f} prof={r['profile_r']:.2f}"
                            for r in rows[-6:])  # fmt: skip
            print(c.name, line, flush=True)
    return rows


def main() -> int:
    """CLI: ``--backend baseline|cloud --level L1|L2|L3 --k 1|4|8 --max-context 3000``."""
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="baseline")
    p.add_argument("--level", default="L3")
    p.add_argument("--k", default="1")
    p.add_argument("--max-context", type=int, default=3000)
    a = p.parse_args()
    k_tag = "" if a.k == "1" else f"_k{a.k}"  # "cal": leave-one-setup-out calibrated
    rows = run(a.backend, a.level, k_tag, a.max_context)
    out = OUT / f"transfer_{a.backend}_{a.level}{k_tag or '_k1'}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
