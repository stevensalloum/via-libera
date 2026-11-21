"""Heatmaps of a predicted map: truth (optional), mean, and 90 % interval width."""

from pathlib import Path

import numpy as np

from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.core.grid import Grid2D


def plot_map(
    grid: Grid2D,
    dist: QuantileDist,
    path: Path,
    truth: np.ndarray | None = None,
    readings: np.ndarray | None = None,
    title: str = "",
) -> Path:
    """Save side-by-side panels to ``path`` (PNG) and return it. O(grid.size)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("prediction (mean)", dist.mean, "viridis"),
        ("uncertainty (90 % width)", dist.width(0.9), "magma"),
    ]
    if truth is not None:
        panels.insert(0, ("truth", truth, "viridis"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4), squeeze=False)
    for ax, (name, values, cmap) in zip(axes[0], panels, strict=True):
        im = ax.imshow(grid.to_image(values), origin="lower", extent=grid.extent, cmap=cmap)
        if readings is not None:
            ax.scatter(
                readings[:, 0], readings[:, 1], s=6, c="white", edgecolors="black", linewidths=0.3
            )
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
