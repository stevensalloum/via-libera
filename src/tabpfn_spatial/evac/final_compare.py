"""Why TabPFN: the pitch figures (``uv run evac-compare``) in ``docs/figures/final``.

Reads the ``evac-final`` runs (``outputs/final/runs-<backend>[-stress].jsonl``) and the
real-data summary of the 3D study (``docs/figures/final/real_data_comparison.json``) and
draws, with the same colours everywhere (TabPFN = Via Libera orange):

1. ``1-exposure.png``: seconds in harmful gas per person, same leaks and people:
   today's plan (nearest exit, no model), CCRP with classic k-NN models, CCRP with
   TabPFN, CCRP with perfect knowledge; known and unseen plants;
2. ``2-accuracy-factory.png``: TabPFN vs k-NN in the 3D factory, per plant: leak
   position, leak height, hazard recall;
3. ``3-accuracy-real.png``: TabPFN vs k-NN on real gas data (Red:Vapor, SRGDM, Örebro,
   UCI), from the 3D study;
4. ``4-compute.png``: time to build the training data (CFD vs our calibrated physics) and
   the latency of one model update, measured in the 3D replays (``world --replay``);
5. ``5-stress.png``: the hard case (``evac-final --stress``); a placeholder until it exists.

A missing input draws a labelled placeholder, so the folder is always complete.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("outputs/final")
OUT = Path("docs/figures/final")
# fixed order: no two warm hues side by side; every bar is labelled as well
METHODS = (("nearest", "Today: nearest exit\n(no model)", "#e34948"),
           ("knn", "CCRP +\nk-NN models", "#8ba3b8"),
           ("tabpfn", "CCRP +\nTabPFN (ours)", "#f39200"),
           ("perfect", "CCRP +\nperfect knowledge", "#12355b"))  # fmt: skip
TABPFN, KNN = "#f39200", "#8ba3b8"
INK, MUTED, GRID = "#0b1622", "#5b6f82", "#dde4ea"
TRAIN_LEAKS = 200  # training leaks per model in evac-final (Rayan's and Ahmad's tables)
SIM_S = 2.5  # s per 3D training leak with our calibrated physics (gas3d, measured)


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [r for line in path.read_text().splitlines() for r in json.loads(line).get("rows", [])]


def _placeholder(path: Path, title: str, note: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", fontsize=14, color=INK, weight="bold")
    ax.text(0.5, 0.4, note, ha="center", fontsize=11, color=MUTED)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _bar_labels(ax, bars, fmt: str) -> None:
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h), ha="center", va="bottom",
                    fontsize=9, color=INK)  # fmt: skip


def exposure(cloud: list[dict], knn: list[dict], out: Path) -> Path:
    """Fig 1: seconds in gas per person by method, known vs unseen plants."""
    path = out / "1-exposure.png"
    if not cloud:
        return _placeholder(path, "Exposure by method", "pending: evac-final --backend cloud")

    def mean(rows, models, policy, split):
        v = [r["harm_s_mean"] for r in rows
             if (r["models"], r["policy"], r["split"]) == (models, policy, split)]  # fmt: skip
        return float(np.mean(v)) if v else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, split in zip(axes, ("known", "unseen"), strict=True):
        vals = [mean(cloud, "tabpfn", "nearest", split), mean(knn, "tabpfn", "ccrp", split),
                mean(cloud, "tabpfn", "ccrp", split),
                mean(cloud, "perfect", "ccrp", split)]  # fmt: skip
        bars = ax.bar(range(4), vals, 0.7, color=[c for *_, c in METHODS])
        _bar_labels(ax, bars, "{:.2f} s")
        ax.set_xticks(range(4), [lab for _, lab, _ in METHODS], fontsize=9)
        n = len({r["plant"] for r in cloud if r["split"] == split})
        ax.set_title(f"{split} plants ({n})", color=INK, fontsize=11)
        _style(ax)
    axes[0].set_ylabel("seconds in harmful gas per person", color=INK)
    fig.suptitle("Same leaks, same people: who breathes gas", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def accuracy_factory(cloud: list[dict], knn: list[dict], out: Path) -> Path:
    """Fig 2: TabPFN vs k-NN per plant: leak position, height, hazard recall."""
    path = out / "2-accuracy-factory.png"
    pick = lambda rows: {r["plant"]: r for r in rows  # noqa: E731
                         if r["models"] == "tabpfn" and r["policy"] == "nearest"}  # fmt: skip
    a, b = pick(cloud), pick(knn)
    plants = [p for p in a if p in b]
    if not plants:
        return _placeholder(path, "TabPFN vs k-NN in the 3D factory",
                            "pending: evac-final --backend baseline")  # fmt: skip
    measures = (("loc_err_median_m", "leak position error (m)", "lower is better"),
                ("height_err_m", "leak height error (m)", "lower is better"),
                ("hz_recall", "harmful cells caught (recall)", "higher is better"))  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, title, better) in zip(axes, measures, strict=True):
        for j, (rows, col) in enumerate(((b, KNN), (a, TABPFN))):
            v = [rows[p].get(key) for p in plants]
            v = [x for x in v if x is not None]
            ax.scatter(np.full(len(v), j) + np.linspace(-0.15, 0.15, len(v)), v, s=22, color=col,
                       alpha=0.85, edgecolor="white", lw=0.5)  # fmt: skip
            if v:
                m = float(np.median(v))
                ax.hlines(m, j - 0.3, j + 0.3, color=INK, lw=2)
                ax.text(j + 0.33, m, f"{m:.2f}", va="center", fontsize=9, color=INK)
        ax.set_xticks([0, 1], ["k-NN", "TabPFN"])
        ax.set_xlim(-0.6, 1.8)
        ax.set_title(f"{title}\n{better}", color=INK, fontsize=10)
        _style(ax)
    fig.suptitle(f"3D factory, {len(plants)} plants (dots), median (line)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def accuracy_real(summary: dict, out: Path) -> Path:
    """Fig 3: TabPFN vs k-NN on real gas data (3D study)."""
    path = out / "3-accuracy-real.png"
    if not summary:
        return _placeholder(path, "TabPFN vs k-NN on real data", "pending: real-data summary")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    src = (("Red:Vapor\n(metres)", summary["source_redvapor_m"]),
           ("SRGDM\n(grid cells)", summary["source_srgdm_cells"]))  # fmt: skip
    hz = [(n.replace("hazard_", ""), summary[n]["auc_hot"])
          for n in ("hazard_redvapor", "hazard_orebro", "hazard_uci") if n in summary]  # fmt: skip
    names = {"redvapor": "Red:Vapor", "orebro": "Örebro", "uci": "UCI"}
    for ax, items, title in ((axes[0], src, "leak position error (median), lower is better"),
                             (axes[1], [(names[n], v) for n, v in hz],
                              "hot-zone AUC, higher is better")):  # fmt: skip
        x = np.arange(len(items))
        for j, (key, col, lab) in enumerate((("knn", KNN, "k-NN"), ("tabpfn", TABPFN, "TabPFN"))):
            bars = ax.bar(x + (j - 0.5) * 0.36, [v[key] for _, v in items], 0.34, color=col,
                          label=lab)  # fmt: skip
            _bar_labels(ax, bars, "{:.2f}")
        ax.set_xticks(x, [n for n, _ in items])
        ax.set_title(title, color=INK, fontsize=10)
        _style(ax)
    axes[1].set_ylim(0.5, 1.0)
    axes[0].legend(frameon=False)
    fig.suptitle("Real gas measurements: TabPFN vs a classic k-NN on the same rows", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def latencies(world: Path) -> dict[str, list[dict]]:
    """Model updates of the 3D replays: ``tabpfn`` (cloud) and ``knn`` (baseline backend)."""
    out: dict[str, list[dict]] = {}
    for f in sorted(world.glob("replay-*-3d/replay.json")):
        rep = json.loads(f.read_text())
        key = (
            "tabpfn" if "cloud" in rep["models"] else "knn" if "baseline" in rep["models"] else None
        )
        if key:
            out.setdefault(key, []).extend(u for u in rep["updates"] if "locate_s" in u)
    return out


def compute(summary: dict, latency: dict, out: Path) -> Path:
    """Fig 4: building the training data, and seconds per leak end to end."""
    path = out / "4-compute.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    cfd = (summary.get("cfd_solve_s") or {}).get("median")
    ax = axes[0]
    if cfd:
        hours = [TRAIN_LEAKS * cfd / 3600, TRAIN_LEAKS * SIM_S / 3600]
        bars = ax.bar([0, 1], hours, 0.6, color=[KNN, TABPFN])
        for b, h in zip(bars, hours, strict=True):
            lab = f"{h:.1f} h" if h >= 1 else f"{h * 60:.0f} min"
            ax.text(b.get_x() + b.get_width() / 2, h, lab, ha="center", va="bottom", fontsize=10,
                    color=INK)  # fmt: skip
        ax.set_xticks([0, 1], [f"CFD (OpenFOAM)\n{cfd:.0f} s per leak", f"calibrated physics\n"
                               f"{SIM_S:.1f} s per leak"])  # fmt: skip
        ax.set_ylabel("hours", color=INK)
    ax.set_title(f"training data for {TRAIN_LEAKS} leaks, one plant\n(TabPFN needs no training: "
                 "it reads the table in context)", color=INK, fontsize=10)  # fmt: skip
    _style(ax)
    ax = axes[1]  # latency of one model update, measured in single replays (no rate limit)
    bottoms = np.zeros(2)
    for key, lab, col in (("locate_s", "find the leak", "#5b6f82"),
                          ("predict_s", "predict the hazard", "#12355b")):  # fmt: skip
        groups = (latency.get("knn", []), latency.get("tabpfn", []))
        vals = np.array([np.median([u[key] for u in g]) if g else np.nan for g in groups])
        ax.bar([0, 1], vals, 0.6, bottom=bottoms, color=col, label=lab)
        bottoms = bottoms + np.nan_to_num(vals)
    for x, v in enumerate(bottoms):
        ax.text(x, v, f"{v:.1f} s", ha="center", va="bottom", fontsize=10, color=INK)
    ax.axhline(15, color="#e34948", lw=1, ls="--")
    ax.text(1.32, 15, "update every 15 s", va="center", fontsize=9, color="#e34948")
    ax.set_xticks([0, 1], ["k-NN (laptop)", "TabPFN (cloud API)"])
    ax.set_ylabel("seconds per model update", color=INK)
    ax.set_title("one model update in an emergency (median)", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def stress(rows: list[dict], out: Path) -> Path:
    """Fig 5: the hard case, per policy (TabPFN vs perfect)."""
    path = out / "5-stress.png"
    if not rows:
        return _placeholder(path, "The hard case: low leaks near people, updates every 5 s",
                            "pending: evac-final --stress (Rayan's run)")  # fmt: skip
    pols = (("nearest", "nearest exit", "#e34948"), ("once", "CCRP planned once", "#8ba3b8"),
            ("ccrp", "CCRP replanned live", "#f39200"))  # fmt: skip
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, models in enumerate(("tabpfn", "perfect")):
        for k, (p, lab, col) in enumerate(pols):
            v = [r["harm_s_mean"] for r in rows if (r["models"], r["policy"]) == (models, p)]
            b = ax.bar(j * 4 + k, float(np.mean(v)) if v else np.nan, 0.85, color=col,
                       label=lab if j == 0 else None)  # fmt: skip
            _bar_labels(ax, b, "{:.2f}")
    ax.set_xticks([1, 5], ["TabPFN models", "perfect knowledge"])
    ax.set_ylabel("seconds in harmful gas per person", color=INK)
    ax.legend(frameon=False, fontsize=9)
    n = len({r["plant"] for r in rows})
    ax.set_title(f"Hard case ({n} plants): low leaks near people, 2x rate, updates every 5 s",
                 color=INK, fontsize=10)  # fmt: skip
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> int:
    """CLI: ``uv run evac-compare [--runs outputs/final] [--out docs/figures/final]``."""
    p = argparse.ArgumentParser(description="Why TabPFN: the pitch figures.")
    p.add_argument("--runs", type=Path, default=RUNS)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--world", type=Path, default=Path("outputs/world"), help="3D replays")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cloud, knn = _rows(a.runs / "runs-cloud.jsonl"), _rows(a.runs / "runs-baseline.jsonl")
    hard = _rows(a.runs / "runs-cloud-stress.jsonl")
    real = a.out / "real_data_comparison.json"
    summary = json.loads(real.read_text()) if real.exists() else {}
    for path in (exposure(cloud, knn, a.out), accuracy_factory(cloud, knn, a.out),
                 accuracy_real(summary, a.out), compute(summary, latencies(a.world), a.out),
                 stress(hard, a.out)):  # fmt: skip
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
