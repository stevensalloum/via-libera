"""Figures of the final 3D test (``evac-final``): policies, gains per plant, model skill."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# the viewer's colours: red nearest, orange planned once, green replanned, blue risk
COLOURS = {"nearest": "#e34948", "risk": "#2a78d6", "once": "#eda100", "ccrp": "#1baf7a"}
NAMES = {"nearest": "nearest exit", "risk": "risk-aware", "once": "CCRP planned once",
         "ccrp": "CCRP replanned live"}  # fmt: skip
INK, MUTED = "#1f2328", "#6e7781"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d8dee4", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def _mean(rows, **kw) -> tuple[float, float]:
    v = [r["harm_s_mean"] for r in rows if all(r[k] == x for k, x in kw.items())]
    return (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))) if v else (np.nan, 0.0)


def figures(rows: list[dict], out: Path, backend: str) -> list[Path]:
    """policies.png, plants.png, skill.png in ``out``."""
    if not rows:
        return []
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    pols = [p for p in COLOURS if any(r["policy"] == p for r in rows)]
    # 1. seconds in gas per policy: known vs unseen, team's models vs perfect
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, models in zip(axes, ("tabpfn", "perfect"), strict=True):
        for j, split in enumerate(("known", "unseen")):
            for k, p in enumerate(pols):
                m, se = _mean(rows, split=split, models=models, policy=p)
                x = j * (len(pols) + 1) + k
                ax.bar(x, m, 0.85, yerr=se, color=COLOURS[p], label=NAMES[p] if j == 0 else None,
                       error_kw={"ecolor": MUTED, "lw": 1})  # fmt: skip
        ax.set_xticks([(len(pols) - 1) / 2, len(pols) + 1 + (len(pols) - 1) / 2],
                      ["known plants", "unseen plants"])  # fmt: skip
        ax.set_title("TabPFN models" if models == "tabpfn" else "perfect models",
                     color=INK, fontsize=11)  # fmt: skip
        _style(ax)
    axes[0].set_ylabel("seconds in harmful gas per person", color=INK)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Final 3D loop: who breathes gas, per policy", color=INK, fontsize=12)
    fig.tight_layout()
    paths.append(out / f"policies-{backend}.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)
    # 2. per plant: nearest vs CCRP live (team's models)
    plants = list(dict.fromkeys(r["plant"] for r in rows))
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(plants)), 3.6))
    for k, p in enumerate(("nearest", "once", "ccrp")):
        ms = [_mean(rows, plant=n, models="tabpfn", policy=p)[0] for n in plants]
        ax.bar(np.arange(len(plants)) + (k - 1) * 0.27, ms, 0.25, color=COLOURS[p], label=NAMES[p])
    ax.set_xticks(range(len(plants)), plants, rotation=30, ha="right")
    ax.set_ylabel("seconds in harmful gas per person", color=INK)
    ax.set_title("Per plant, TabPFN models", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    paths.append(out / f"plants-{backend}.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)
    # 3. model skill per split (one row per measure, its own scale)
    team = [r for r in rows if r["models"] == "tabpfn" and r["policy"] == "nearest"]
    measures = (("loc_err_median_m", "leak position error (m)"),
                ("height_err_m", "leak height error (m)"),
                ("hz_recall", "hazard recall at P >= 0.5"),
                ("t_alarm", "alarm time (s)"))  # fmt: skip
    fig, axes = plt.subplots(1, len(measures), figsize=(12, 3.0))
    for ax, (key, title) in zip(axes, measures, strict=True):
        data = [[r[key] for r in team if r["split"] == s and r.get(key) is not None]
                for s in ("known", "unseen")]  # fmt: skip
        for j, d in enumerate(data):
            if d:
                ax.scatter(np.full(len(d), j) + np.linspace(-0.12, 0.12, len(d)), d, s=16,
                           color="#57606a", alpha=0.7)  # fmt: skip
                ax.hlines(np.median(d), j - 0.25, j + 0.25, color=INK, lw=2)
        ax.set_xticks([0, 1], ["known", "unseen"])
        ax.set_title(title, color=INK, fontsize=10)
        _style(ax)
    fig.suptitle("Source, height and hazard skill of the TabPFN models (line = median)", color=INK)
    fig.tight_layout()
    paths.append(out / f"skill-{backend}.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)
    return paths
