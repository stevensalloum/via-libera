"""Task S on SRGDM synthetic plumes: transfer of a locator to an unseen airflow.

SRGDM simulated 30 source positions under 4 airflows recorded in a real room ("seasons").
Rows: the 6 x 5 sparse sensors (the authors' mask) averaged over windows of 20 frames, each
divided by its strongest reading (the pattern, not the dose); 9 windows per run, 1,080 rows.
Target: the source cell (row, column) on the 30 x 25 grid. Leave-one-season-out: context
from 3 airflows, test on the 4th. Score: error in cells. Reference: the strongest
sensor; it scores ~1.4 cells by construction, since every source sits 1.4 cells from a
sensor (sources at cells 1 + 5 i, sensors at 2 + 5 i).

The real 6 x 5 recordings have no documented source position, so no real test exists here:
this is the sim -> sim part of the study (generalisation across airflow), which bounds what
workflow B can do when the target airflow differs from the simulated ones.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tabpfn_spatial.core.models import make_regressor
from tabpfn_spatial.realdata import srgdm
from tabpfn_spatial.study3d.ladder import OUT

WINDOW = 20


def rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features (n, 30), source cells (n, 2), season (n,)."""
    s = srgdm.load_tensor(Path("data/real/srgdm/30x25/valid.file"))
    r, c = srgdm.sensor_cells()
    x, y, season = [], [], []
    for i in range(s.shape[0]):
        frames = s[i][:, r][:, :, c]  # (180, 6, 5)
        for w in range(0, frames.shape[0] - WINDOW + 1, WINDOW):
            v = frames[w : w + WINDOW].mean(0).ravel()
            x.append(v / max(v.max(), 1e-9))
            y.append(srgdm.source_cell(i))
            season.append(i // srgdm.SOURCES)
    return np.array(x), np.array(y, float), np.array(season)


def run(backend: str) -> list[dict]:
    """Leave-one-season-out errors (cells) for TabPFN and the strongest sensor."""
    x, y, season = rows()
    r, c = srgdm.sensor_cells()
    grid = np.array([(rr, cc) for rr in r for cc in c], float)
    out = []
    for s in range(4):
        tr, te = season != s, season == s
        pred = np.c_[[make_regressor(backend).fit(x[tr], y[tr, k]).predict_dist(x[te]).mean
                      for k in (0, 1)]].T  # fmt: skip
        strongest = grid[np.argmax(x[te], 1)]
        for name, p in (("B sim->sim, unseen airflow", pred), ("strongest sensor", strongest)):
            err = np.hypot(*(p - y[te]).T)
            out.append({"season": s, "workflow": name, "backend": backend if "B" in name else "-",
                        "median_cells": float(np.median(err)),
                        "p90_cells": float(np.percentile(err, 90))})  # fmt: skip
        print(s, " ".join(f"{o['workflow']}: {o['median_cells']:.1f}" for o in out[-2:]))
    return out


def main() -> int:
    """CLI: ``--backend baseline|cloud``."""
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="baseline")
    a = p.parse_args()
    res = run(a.backend)
    path = OUT / f"source_srgdm_{a.backend}.json"
    path.write_text(json.dumps(res, indent=1))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
