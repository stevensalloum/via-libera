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
    p.add_argument("--l4", action="store_true", help="add L4 (CFD) and L4G (GADEN) rows")
    a = p.parse_args()
    if a.l4:
        main_l4()
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {"redvapor": cs.redvapor_cases, "orebro": cs.orebro_cases,
             "uci": cs.uci_cases}[a.dataset]()  # fmt: skip
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


def calibrate(dataset: str = "redvapor", factors: tuple[str, ...] = ("", "_k4", "_k8")) -> Path:
    """Leave-one-setup-out choice of the turbulence factor, per level.

    For each held-out setup, the factor with the best mean score on the *other* setups
    (AUC of the hot 10 % + vertical profile r - |log width ratio|) is chosen, and the
    held-out experiments take that factor's fields. Writes ``fields_<dataset>_kcal.npz``
    and ``calibration_<dataset>.json`` (the chosen factor per setup and level).
    """
    runs = {f: json.loads((OUT / f"ladder_{dataset}{f}.json").read_text()) for f in factors}
    fields = {f: np.load(OUT / f"fields_{dataset}{f}.npz") for f in factors}

    def merit(r: dict) -> float:
        prof = r["profile_r"] if np.isfinite(r["profile_r"]) else 0.0
        return r["auc_hot"] + prof - abs(np.log(max(r["width_ratio"], 1e-3)))

    setups = sorted({r["setup"] for r in runs[factors[0]]})
    levels = sorted({r["level"] for r in runs[factors[0]]})
    out, chosen = {}, []
    for setup in setups:
        for level in levels:
            best = max(factors, key=lambda f: np.mean([merit(r) for r in runs[f]
                                                       if r["setup"] != setup
                                                       and r["level"] == level]))  # fmt: skip
            chosen.append({"setup": setup, "level": level, "factor": best or "_k1"})
            for r in runs[best]:
                if r["setup"] == setup and r["level"] == level:
                    out[f"{r['case']}_{level}"] = fields[best][f"{r['case']}_{level}"]
    np.savez_compressed(OUT / f"fields_{dataset}_kcal.npz", **out)
    (OUT / f"calibration_{dataset}.json").write_text(json.dumps(chosen, indent=1))
    return OUT / f"fields_{dataset}_kcal.npz"


CFD_CASES = {"A": "setupA_exp1", "B": "setupB_exp2", "C": "setupC_exp4", "D": "setupD_exp6"}


def run_l4(c: cs.Case, factor: float) -> dict[str, np.ndarray]:
    """L4a (CFD wind + CFD nu_t/Sc_t, our transport) and L4b (GADEN filaments in CFD wind).

    One CFD solve per setup; other wind speeds scale it (RANS around sharp-edged
    buildings is Reynolds-number independent: velocities and nu_t scale with U).
    ``factor`` multiplies the diffusivity (L4a) and the filament growth (L4b).
    """
    from tabpfn_spatial.study3d import cfd, gaden

    d = Path("outputs/study3d/cfd") / CFD_CASES[c.info["setup"]]
    ref = next(x for x in cs.redvapor_cases() if x.name == d.name.split("_")[1])
    scale = float(c.flow.at("speed", 0.5) / ref.flow.at("speed", 0.5))
    w = cfd.wind_on_grid(d, c.grid, c.solid, c.flow)
    w = ph.Faces(w.ux * scale, w.uy * scale, w.uz * scale)
    k = cfd.diffusivity_on_grid(d, c.grid) * scale * factor
    out = {"L4": ph.sample(c.grid, ph.transport(c.grid, c.solid, c.flow, w, c.source, k_field=k),
                           c.xyz)}  # fmt: skip
    root = gaden.write_config(Path(f"outputs/study3d/gaden/{c.name}_f{factor:g}"), c, d, scale,
                              60.0, gamma_factor=factor)  # fmt: skip
    out["L4G"] = gaden.run(root, t_end=60.0, t_start=10.0)
    return out


def main_l4(factors: tuple[float, ...] = (1.0, 4.0, 8.0)) -> None:
    """L4 rows for every Red:Vapor experiment and factor, appended as separate files."""
    cases = cs.redvapor_cases()
    for factor in factors:
        tag = "" if factor == 1.0 else f"_k{factor:g}"
        rows = json.loads((OUT / f"ladder_redvapor{tag}.json").read_text())
        rows = [r for r in rows if not r["level"].startswith("L4")]
        fields = dict(np.load(OUT / f"fields_redvapor{tag}.npz"))
        for c in cases:
            t = time.time()
            for level, m in run_l4(c, factor).items():
                fields[f"{c.name}_{level}"] = m.astype(np.float32)
                rows.append({"case": c.name, "level": level, **c.info,
                             **score(c.xyz, c.obs, m, c.source, c.flow.direction)})  # fmt: skip
            print(f"k{factor:g}", c.name, f"{time.time() - t:.0f}s", " ".join(
                f"{r['level']} auc={r['auc_hot']:.2f} prof={r['profile_r']:.2f} "
                f"w={r['width_ratio']:.2f}" for r in rows[-2:]), flush=True)  # fmt: skip
        (OUT / f"ladder_redvapor{tag}.json").write_text(json.dumps(rows, indent=1))
        np.savez_compressed(OUT / f"fields_redvapor{tag}.npz", **fields)


if __name__ == "__main__":
    raise SystemExit(main())
