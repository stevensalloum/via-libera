"""Result figures of the 3D study (``uv run python -m tabpfn_spatial.study3d.report``).

Reads ``outputs/study3d/*.json`` and writes ``docs/figures/3d-study/``:
- ``ladder.png``: each physics level against each real dataset (measured vs calibrated
  turbulence);
- ``profiles.png``: real vs simulated vertical profiles downwind, one panel per setup;
- ``transfer_hazard.png``: the two workflows for the hazard task (cloud TabPFN);
- ``transfer_source.png``: the two workflows for the source task.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tabpfn_spatial.study3d.overview import INK, MUTED, PALETTE, _style

RES = Path("outputs/study3d")
FIG = Path("docs/figures/3d-study")


def _load(name: str) -> list[dict]:
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else []


def _dots(ax: plt.Axes, rows: list[dict], key: str, groups: list[tuple[str, str]],
          color: str, offset: float, label: str) -> None:  # fmt: skip
    """Per-experiment dots (light) and the mean (bold) per group on the y axis."""
    for y, (field, value) in enumerate(groups):
        v = np.array([r[key] for r in rows if r.get(field) == value and np.isfinite(r[key])])
        if not v.size:
            continue
        ax.scatter(v, np.full(v.size, y + offset), s=14, color=color, alpha=0.35, lw=0)
        ax.scatter([v.mean()], [y + offset], s=60, color=color, edgecolor="white", lw=1,
                   label=label if y == 0 else None, zorder=3)  # fmt: skip


def fig_ladder(out: Path) -> Path:
    """Physics levels vs real data: AUC of the hot 10 %, vertical profile r, spread ratio."""
    sets = [("Red:Vapor", "redvapor", ["", "_kcal"]), ("UCI", "uci", [""]),
            ("Örebro", "orebro", [""])]  # fmt: skip
    metrics = [("auc_hot", "AUC: finds the real hottest 10 %"),
               ("profile_r", "vertical profile r (downwind)"),
               ("width_ratio", "crosswind spread, model / real")]  # fmt: skip
    levels = ["L1", "L2", "L3", "L4"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    labels = [f"{d} {lv}" for d, _, _ in sets for lv in levels]
    for ax, (m, title) in zip(axes, metrics, strict=True):
        variants = [
            ("", PALETTE[0], "measured turbulence"),
            ("_kcal", PALETTE[1], "calibrated (leave-one-setup-out)"),
        ]
        for k, (tag, color, name) in enumerate(variants):
            for j, (_, ds, tags) in enumerate(sets):
                if tag not in tags:
                    continue
                rows = _load(f"ladder_{ds}{tag}") if tag != "_kcal" else _kcal_rows(ds)
                for i, lv in enumerate(levels):
                    v = np.array([r[m] for r in rows if r["level"] == lv and np.isfinite(r[m])])
                    if not v.size:
                        continue
                    y = j * len(levels) + i + (0.18 if k else -0.18)
                    ax.scatter(v, np.full(v.size, y), s=12, color=color, alpha=0.35, lw=0)
                    ax.scatter([v.mean()], [y], s=55, color=color, edgecolor="white", lw=1,
                               zorder=3, label=name if (j == 0 and i == 0) else None)  # fmt: skip
        if m == "width_ratio":
            ax.set_xscale("log")
            ax.axvline(1, color=MUTED, lw=1, ls=":")
        else:
            ax.axvline(0.5 if m == "auc_hot" else 0, color=MUTED, lw=1, ls=":")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Physics ladder vs real data (dots: experiments, big dot: mean; dotted: "
                 "chance / perfect)", fontsize=11, color=INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _kcal_rows(ds: str) -> list[dict]:
    """Ladder rows of the calibrated factor chosen for each held-out setup and level."""
    cal = _load(f"calibration_{ds}")
    out = []
    for c in cal:
        tag = "" if c["factor"] == "_k1" else c["factor"]
        out += [r for r in _load(f"ladder_{ds}{tag}")
                if r["setup"] == c["setup"] and r["level"] == c["level"]]  # fmt: skip
    return out


def fig_transfer_hazard(out: Path, run: str = "transfer_cloud_L3_kcal") -> Path:
    """Hazard workflows: rank correlation, AUC of the hot 10 %, vertical profile."""
    rows = _load(run)
    order = ["real->real", "hybrid", "A real->sim", "B other-sites", "B same-site", "physics"]
    names = {"real->real": "real -> real (reference)", "hybrid": "hybrid: real + simulated",
             "A real->sim": "A: real context -> simulated site",
             "B other-sites": "B: simulated other sites -> real",
             "B same-site": "B: simulated target site -> real",
             "physics": "physics alone (calibrated L3)"}  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    panels = [
        ("rho", "rank correlation (Spearman)"),
        ("auc_hot", "AUC: finds the real hottest 10 %"),
        ("profile_r", "vertical profile r"),
    ]
    for ax, (m, title) in zip(axes, panels, strict=True):
        for y, w in enumerate(order):
            v = np.array([r[m] for r in rows if r["workflow"] == w])
            color = PALETTE[0] if w in ("real->real", "hybrid") else (
                PALETTE[1] if w.startswith("A") else PALETTE[2])  # fmt: skip
            ax.scatter(v, np.full(v.size, y), s=16, color=color, alpha=0.4, lw=0)
            ax.scatter([v.mean()], [y], s=70, color=color, edgecolor="white", lw=1, zorder=3)
            ax.text(v.mean(), y - 0.3, f"{v.mean():.2f}", ha="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_yticks(range(len(order)), [names[w] for w in order], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Hazard task on Red:Vapor, leave-one-setup-out, cloud TabPFN (dots: 8 "
                 "experiments; truth = real, except A: truth = simulator)", fontsize=11,
                 color=INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_source(out: Path, run: str) -> Path:
    """Source task: horizontal error per workflow (m, 1:50 model scale)."""
    rows = _load(run)
    order = ["real->real", "A real->sim", "B same-site", "B other-sites", "physics inversion",
             "strongest detector"]  # fmt: skip
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for y, w in enumerate(order):
        v = np.array([r["error_m"] for r in rows if r["workflow"] == w])
        if not v.size:
            continue
        color = (
            PALETTE[0]
            if w == "real->real"
            else (PALETTE[1] if w.startswith("A") else (PALETTE[2] if w.startswith("B") else MUTED))
        )
        ax.scatter(v, np.full(v.size, y), s=18, color=color, alpha=0.45, lw=0)
        ax.scatter([np.median(v)], [y], s=70, color=color, edgecolor="white", lw=1, zorder=3)
        ax.text(np.median(v), y - 0.3, f"{np.median(v):.2f} m", ha="center", fontsize=8)
    ax.set_yticks(range(len(order)), order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("horizontal error of the leak position (m, 1:50 model; x 50 for full scale)")
    ax.set_title("Source task on Red:Vapor (50 fixed detectors; big dot: median)",
                 fontsize=10, color=INK, loc="left")  # fmt: skip
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_profiles(out: Path) -> Path:
    """Real vs simulated vertical profiles downwind, one experiment per setup (calibrated)."""
    from tabpfn_spatial.study3d import cases as cs
    from tabpfn_spatial.study3d.evaluate import vertical_profile

    fields = np.load(RES / "fields_redvapor_kcal.npz")
    cases = {c.name: c for c in cs.redvapor_cases()}
    pick = [("exp1", "A: tank only"), ("exp2", "B: tank + tall cylinder"),
            ("exp4", "C: full site"), ("exp6", "D: full site, turned")]  # fmt: skip
    levels = [("L1", "L1 plume"), ("L3", "L3 mass-consistent wind"),
              ("L4", "L4a CFD wind + CFD K"), ("L4G", "L4b GADEN in CFD wind")]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.4), sharey=True)
    for ax, (name, title) in zip(axes, pick, strict=True):
        c = cases[name]
        z, p = vertical_profile(c.xyz, c.obs, c.source, c.flow.direction)
        ax.plot(p, z, color=INK, lw=2.5, marker="o", ms=4, label="real (PID excess)")
        for k, (lv, lab) in enumerate(levels):
            key = f"{name}_{lv}"
            if key not in fields:
                continue
            zm, pm = vertical_profile(c.xyz, fields[key].astype(float), c.source, c.flow.direction)
            ax.plot(pm, zm, color=PALETTE[k], lw=1.8, label=lab)
        ax.axhline(c.source[2], color=MUTED, lw=1, ls=":")
        ax.set_title(f"{name}, setup {title}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("normalised mean concentration")
        _style(ax)
    axes[0].set_ylabel("height (m, 1:50 model)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Does height come out right? Downwind vertical profiles, real vs each level "
                 "(calibrated turbulence; dotted: release height)", fontsize=11, color=INK,
                 x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    """Write every figure whose results exist."""
    FIG.mkdir(parents=True, exist_ok=True)
    print(fig_ladder(FIG / "ladder.png"))
    if (RES / "fields_redvapor_kcal.npz").exists():
        print(fig_profiles(FIG / "profiles.png"))
    if _load("transfer_cloud_L3_kcal"):
        print(fig_transfer_hazard(FIG / "transfer_hazard.png"))
    for b in ("cloud", "baseline"):
        if _load(f"source_redvapor_{b}"):
            print(fig_source(FIG / f"transfer_source_{b}.png", f"source_redvapor_{b}"))
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
