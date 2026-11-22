"""Hazard-task workflows on the small datasets: Örebro (10 experiments) and UCI (3 fans).

Leave-one-case-out. Features as on Red:Vapor (``transfer.FEATURES``). The simulated side
is the L3 field (L1 for comparison); because a simulator gives any number of rows, the
simulated contexts sample ``N_SIM`` points from the whole simulated volume. Real and
simulated targets are normalised the same way: by the 99th percentile *at the real sensor
positions* (so both use the same reference points).

Workflows: ``real->real``, ``B same-site`` (simulated volume of the held-out case),
``B other-sites`` (simulated volumes of the other cases), ``hybrid`` (real other cases +
simulated held-out case), ``A real->sim`` (real other cases, truth = simulator at the held-out
sensors), and ``physics`` (the field itself).
"""

import argparse
import json

import numpy as np

from tabpfn_spatial.core.models import make_regressor
from tabpfn_spatial.study3d import cases as cs
from tabpfn_spatial.study3d import physics as ph
from tabpfn_spatial.study3d.evaluate import score
from tabpfn_spatial.study3d.ladder import OUT
from tabpfn_spatial.study3d.transfer import EPS, blocked

N_SIM = 3000


def _feat(c: cs.Case, xyz: np.ndarray) -> np.ndarray:
    d = np.asarray(c.flow.direction)
    rel = xyz - c.source
    s, n = rel[:, :2] @ d, rel[:, :2] @ np.array([-d[1], d[0]])
    u = np.full(len(rel), float(c.flow.at("speed", c.source[2])))
    return np.c_[s, n, xyz[:, 2], rel[:, 2], u, blocked(c.grid, c.solid, c.source, xyz)]


def _norm(v: np.ndarray, ref_vals: np.ndarray) -> np.ndarray:
    pos = ref_vals[ref_vals > 0]
    ref = np.percentile(pos, 99) if pos.size else 1.0
    return np.log10(np.clip(v, 0, None) / ref + EPS)


def _field(c: cs.Case, level: str) -> np.ndarray:
    f = c.flow
    if level == "L1":
        g = c.grid
        pts = np.stack(np.meshgrid(g.axis(0), g.axis(1), g.axis(2), indexing="ij"), -1)
        return ph.plume(pts.reshape(-1, 3), c.source, f).reshape(g.shape)
    w = ph.base_wind(c.grid, c.solid, f)
    if level == "L3":
        w = ph.mass_consistent(c.grid, c.solid, f, w)
    return ph.transport(c.grid, c.solid, f, w, c.source)


def run(dataset: str, backend: str, level: str, seed: int = 0) -> list[dict]:
    """All workflows, every held-out case of ``dataset``."""
    rng = np.random.default_rng(seed)
    cases = {"orebro": cs.orebro_cases, "uci": cs.uci_cases}[dataset]()
    sim = {}
    for c in cases:
        field = _field(c, level)
        g = c.grid
        cells = np.argwhere(~c.solid)
        pick = cells[rng.choice(len(cells), N_SIM, replace=False)]
        xyz = np.c_[g.axis(0)[pick[:, 0]], g.axis(1)[pick[:, 1]], g.axis(2)[pick[:, 2]]]
        at_sensors = ph.sample(g, field, c.xyz)
        vol = field[pick[:, 0], pick[:, 1], pick[:, 2]]
        sim[c.name] = (_feat(c, xyz), _norm(vol, at_sensors), at_sensors)
    real = {c.name: (_feat(c, c.xyz), _norm(c.obs, c.obs)) for c in cases}
    rows = []

    def fp(xc: np.ndarray, yc: np.ndarray, xt: np.ndarray) -> np.ndarray:
        return make_regressor(backend).fit(xc, yc).predict_dist(xt).mean

    for c in cases:
        others = [o.name for o in cases if o.name != c.name]
        xr = np.vstack([real[o][0] for o in others])
        yr = np.concatenate([real[o][1] for o in others])
        xs = np.vstack([sim[o][0] for o in others])
        ys = np.concatenate([sim[o][1] for o in others])
        xt = real[c.name][0]
        p_real = fp(xr, yr, xt)
        xh = np.vstack([np.c_[xr, np.zeros(len(xr))], np.c_[sim[c.name][0], np.ones(N_SIM)]])
        yh = np.concatenate([yr, sim[c.name][1]])
        preds = {
            "real->real": (p_real, c.obs),
            "A real->sim": (p_real, sim[c.name][2]),
            "B same-site": (fp(sim[c.name][0], sim[c.name][1], xt), c.obs),
            "B other-sites": (fp(xs, ys, xt), c.obs),
            "hybrid": (fp(xh, yh, np.c_[xt, np.zeros(len(xt))]), c.obs),
        }
        for name, (p, truth) in preds.items():
            rows.append({"dataset": dataset, "case": c.name, "workflow": name, "level": level,
                         "backend": backend,
                         **score(c.xyz, truth, 10**p, c.source, c.flow.direction)})  # fmt: skip
        phys = score(c.xyz, c.obs, sim[c.name][2], c.source, c.flow.direction)
        rows.append({"dataset": dataset, "case": c.name, "workflow": "physics", "level": level,
                     "backend": "-", **phys})  # fmt: skip
        print(c.name, " ".join(f"{r['workflow']}: rho={r['rho']:.2f} auc={r['auc_hot']:.2f}"
                               for r in rows[-6:]), flush=True)  # fmt: skip
    return rows


def main() -> int:
    """CLI: ``--dataset orebro|uci --backend baseline|cloud --level L1|L3``."""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="orebro")
    p.add_argument("--backend", default="baseline")
    p.add_argument("--level", default="L3")
    a = p.parse_args()
    rows = run(a.dataset, a.backend, a.level)
    out = OUT / f"transfer_{a.dataset}_{a.backend}_{a.level}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
