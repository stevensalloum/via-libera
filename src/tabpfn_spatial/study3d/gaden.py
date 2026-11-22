"""L4b: GADEN filament dispersion (Ojeda et al., gaden_core) in the CFD wind.

GADEN is run natively through a small C++ driver (``~/dev/gaden/driver/gaden_run``) that
time-averages the concentration at the observation points. Its parameters, each mapped
to a physical quantity:

- **filamentGrowthGamma** (cm^2/s): a filament's width grows as sigma^2 = sigma0^2 + gamma t;
  a puff in eddy diffusivity K grows as sigma^2 = 2 K t, so gamma = 2 K (x 1e4 cm^2/m^2),
  with K = the CFD's median nu_t / Sc_t near the site;
- **filamentNoise_std** (m/s): random velocity per step for unresolved turbulence; set to
  sqrt(2 k / 3), the CFD's median velocity fluctuation;
- **filamentInitialSigma** = 3 cm (the same initial size as the L1 plume);
- **gas** = carbon monoxide (GADEN specific gravity 0.967, the nearest to the near-neutral fog);
- the tunnel: a closed box one cell outside the study grid, with an **outlet** face downwind
  (filaments leave there), so no near-floor layer is lost.
"""

import subprocess
from pathlib import Path

import numpy as np

from tabpfn_spatial.study3d import cfd
from tabpfn_spatial.study3d.cases import Case
from tabpfn_spatial.study3d.geometry import read_stl

DRIVER = Path.home() / "dev/gaden/driver/build/gaden_run"
CO = 12  # GADEN gas id (GasTypes.hpp)


def _box_faces(lo: np.ndarray, hi: np.ndarray, faces: list[str]) -> np.ndarray:
    """Triangles (n, 3, 3) of the named faces of an axis-aligned box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    quads = {
        "x0": [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        "x1": [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],
        "y0": [(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)],
        "y1": [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        "floor": [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        "top": [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)],
    }
    tri = []
    for f in faces:
        a, b, c, d = (np.array(v, float) for v in quads[f])
        tri += [(a, b, c), (a, c, d)]
    return np.array(tri)


def write_config(root: Path, c: Case, cfd_case: Path, speed_scale: float, t_end: float,
                 gamma_factor: float = 1.0) -> Path:  # fmt: skip
    """GADEN environment configuration + simulation ``sim1`` for one case."""
    g = c.grid
    lo = np.array(g.lo) - g.cell
    hi = np.array(g.lo) + g.cell * (np.array(g.shape) + 1)
    root.mkdir(parents=True, exist_ok=True)
    tri = read_stl(
        Path(f"data/real/redvapor/Setups/Setup{c.info['setup']}/setup{c.info['setup']}.stl")
    )
    cfd.write_stl_ascii(tri, root / "site.stl", "site")
    cfd.write_stl_ascii(
        _box_faces(lo, hi, ["x1", "y0", "y1", "floor", "top"]), root / "walls.stl", "walls"
    )
    cfd.write_stl_ascii(_box_faces(lo, hi, ["x0"]), root / "outlet.stl", "outlet")
    u = cfd.cell_velocity(cfd_case, g) * speed_scale
    xs, ys, zs = g.axis(0), g.axis(1), g.axis(2)
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    (root / "wind").mkdir(exist_ok=True)
    with open(root / "wind/wind_at_cell_centers_0.csv", "w") as fh:
        fh.write('"U:0","U:1","U:2","Points:0","Points:1","Points:2"\n')
        np.savetxt(fh, np.c_[u.reshape(-1, 3), pts], delimiter=",", fmt="%.5g")
    k_turb = cfd.scalar_on_grid(cfd_case, g, "k") * speed_scale**2
    k_diff = cfd.diffusivity_on_grid(cfd_case, g) * speed_scale
    near = (~c.solid) & (zs[None, None, :] < 0.6)
    gamma = 2 * float(np.median(k_diff[near])) * 1e4 * gamma_factor
    noise = float(np.sqrt(2 * np.median(k_turb[near]) / 3))
    (root / "config.yaml").write_text(f"""models:
  - {root.resolve() / "site.stl"}
  - {root.resolve() / "walls.stl"}
outlets_models:
  - {root.resolve() / "outlet.stl"}
unprocessed_wind_files: {root.resolve() / "wind/wind_at_cell_centers"}
empty_point: [1.0, 1.0, 1.0]
cell_size: {g.cell}
uniformWind: false
""")
    sim = root / "simulations/sim1"
    sim.mkdir(parents=True, exist_ok=True)
    src = c.source
    (sim / "sim.yaml").write_text(f"""source:
  sourceType: point
  position: [{src[0]:.4f}, {src[1]:.4f}, {src[2] + 0.02:.4f}]
  gasType: {CO}
deltaTime: 0.05
windIterationDeltaTime: 1
temperature: 298
pressure: 1
filamentPPMcenter: 20
filamentInitialSigma: 3
filamentGrowthGamma: {gamma:.4g}
filamentNoise_std: {noise:.4g}
numFilaments_sec: 100
expectedNumIterations: {int(t_end / 0.05) + 10}
saveResults: false
saveDeltaTime: 0.5
windLooping:
  loop: false
  from: 0
  to: 0
""")
    np.savetxt(root / "points.csv", c.xyz, delimiter=",", fmt="%.4f")
    return root


def run(root: Path, t_end: float = 60.0, every: float = 0.25, t_start: float = 10.0) -> np.ndarray:
    """Time-mean ppm at the observation points (runs the native driver)."""
    out = root / "mean.csv"
    args = [str(DRIVER), str(root), "sim1", str(t_end), str(every), str(t_start),
            str(root / "points.csv"), str(out)]  # fmt: skip
    with (root / "log.txt").open("w") as log:
        subprocess.run(args, check=True, stdout=log, stderr=log)
    return np.loadtxt(out, delimiter=",", skiprows=1)[:, 1]
