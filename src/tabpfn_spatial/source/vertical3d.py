"""Leak height on vertical pipes from the team's 3D simulator (``sim.plant3d`` + ``sim.gas3d``).

Replaces the stand-in ``source.vertical`` (2D simulator x assumed height factor) with
Steven's time-stepped 3D gas: real heights of pipes, machines and detectors, buoyancy near
the leak, 1 m voxels. The stand-in stays runnable for the before/after comparison.

- Leaks: the plant's own leak sites on vertical pipes (every 0.5 m), drawn uniformly;
  size and air as ``sim.sample_scenarios`` (rate 0.5-2, wind 0.2-1.5 m/s, any direction).
- Detector layouts (all read from the same simulated volume, so they compare fairly):
  ``mounted`` (the plant's detectors at the gas's mounting height, no masts),
  ``mast_rule`` (``Plant3D.detectors``: mounted + 3 per vertical pipe),
  ``three_heights`` (every mounted position at the three mounting heights).
- Rows: exactly the locator's row (``features.moment_row``), same noise model as
  ``library.noisy_series``.

Cost: one ``gas3d.simulate`` per leak, about 0.6 s for 70 s of gas on the default plant;
``workers`` > 1 runs leaks in parallel processes.
"""

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabpfn_spatial.sim import sample_scenarios
from tabpfn_spatial.sim.gas3d import Leak3D, Ventilation, simulate
from tabpfn_spatial.sim.plant3d import LEAK_DZ, MOUNT_Z, Plant3D
from tabpfn_spatial.source.features import alarm_time, moment_row
from tabpfn_spatial.source.library import ALARM_WINDOW, DEFAULT_WS
from tabpfn_spatial.source.vertical import HeightLibrary


@dataclass(frozen=True)
class PipeLeak:
    """One leak on a vertical pipe: position (m), rate (2D units), air (m/s), which pipe."""

    xyz: tuple[float, float, float]
    rate: float
    wind: tuple[float, float]
    pipe: int


def vertical_sites(plant: Plant3D) -> tuple[np.ndarray, np.ndarray]:
    """(n, 3) leak sites on vertical pipes (as in ``Plant3D.leak_sites``) and their pipe index."""
    pts, pipe = [], []
    for i, v in enumerate(plant.vertical_pipes):
        for z in np.arange(v.z0, v.z1 - 1e-9, LEAK_DZ):
            pts.append((*v.xy, float(z)))
            pipe.append(i)
    return np.array(pts, float), np.array(pipe, int)


def sample_pipe_leaks(plant: Plant3D, n: int, rng: np.random.Generator) -> list[PipeLeak]:
    """``n`` leaks, uniform over the vertical-pipe sites; size and air as ``sample_scenarios``."""
    scenarios = sample_scenarios(plant.f, n, rng)
    sites, pipe = vertical_sites(plant)
    pick = rng.integers(len(sites), size=n)
    return [
        PipeLeak(tuple(map(float, sites[i])), sc.rate, tuple(map(float, sc.wind)), int(pipe[i]))
        for sc, i in zip(scenarios, pick, strict=True)
    ]


def _beside(plant: Plant3D, p: np.ndarray, gap: float) -> np.ndarray:
    """``p`` moved ``gap`` m sideways into the first open voxel (+x, -x, +y, -y)."""
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        q = p + gap * np.array([dx, dy, 0.0])
        i, j, k = plant.grid.index(q[None])[0]
        inside = np.all(q >= 0) and q[0] < plant.f.width and q[1] < plant.f.height
        if inside and not plant.solid[i, j, k]:
            return q
    return p.copy()


def layouts3d(plant: Plant3D, mast_gap: float = 1.0) -> dict[str, np.ndarray]:
    """(k, 3) detector positions of the compared layouts.

    ``mast_rule`` puts the masts on the pipe itself (inside the leak's voxel column);
    ``mast_1m`` is the same rule with every mast ``mast_gap`` m beside the pipe, as built.
    """
    det, why = plant.detectors
    is_mast = np.array(why) == "mast"
    mounted = det[~is_mast]
    heights = sorted(MOUNT_Z.values())
    three = np.array([(x, y, z) for x, y, _ in mounted for z in heights])
    moved = det.copy()
    moved[is_mast] = [_beside(plant, p, mast_gap) for p in det[is_mast]]
    return {"mounted": mounted, "mast_rule": det, "mast_1m": moved, "three_heights": three}


def _one_run(args) -> dict[str, np.ndarray]:
    plant, lk, points, t_end = args
    speed = float(np.hypot(*lk.wind))
    vent = Ventilation(speed, float(np.arctan2(lk.wind[1], lk.wind[0])))
    run = simulate(plant, Leak3D(lk.xyz, lk.rate, plant.f.gas), vent, t_end=t_end)
    return {name: run.at_points(p).T for name, p in points.items()}


def clean_readings(
    plant: Plant3D,
    leaks: Sequence[PipeLeak],
    points: Mapping[str, np.ndarray],
    t_end: float = ALARM_WINDOW + max(DEFAULT_WS),
    workers: int = 1,
    cache: Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``times`` and noise-free readings (B, k, T) per layout; one 3D simulation per leak.

    ``cache`` (an ``.npz``) stores the result so a study reruns without re-simulating.
    """
    times = np.arange(0.0, t_end + 1e-9, 5.0)
    if cache is not None and cache.exists():
        d = np.load(cache)
        if set(points) <= set(d.files):  # a new layout means simulating again
            return d["times"], {name: d[name] for name in points}
    jobs = [(plant, lk, dict(points), t_end) for lk in leaks]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            runs = list(ex.map(_one_run, jobs, chunksize=8))
    else:
        runs = [_one_run(j) for j in jobs]
    out = {name: np.stack([r[name] for r in runs]) for name in points}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, times=times, **out)
    return times, out


def height_rows(
    times: np.ndarray,
    clean: np.ndarray,
    leaks: Sequence[PipeLeak],
    rng: np.random.Generator,
    ws: Sequence[float] = DEFAULT_WS,
    noise: float = 0.1,
    floor: float = 0.001,
    wind_noise: float = 0.1,
) -> HeightLibrary:
    """Rows at each moment ``ws`` after each leak's alarm (noisy, as ``library.noisy_series``).

    Leaks that do not alarm within ``ALARM_WINDOW`` s are skipped.
    """
    values = clean * (1 + noise * rng.standard_normal(clean.shape))
    values = np.maximum(values + floor * rng.standard_normal(clean.shape), 0.0)
    wind = np.array([lk.wind for lk in leaks]) + wind_noise * rng.standard_normal((len(leaks), 2))
    cols: dict[str, list] = {k: [] for k in ("X", "z", "w", "t_alarm", "leak", "riser")}
    for b, lk in enumerate(leaks):
        t_a = alarm_time(times, values[b])
        if t_a is None or t_a > ALARM_WINDOW:
            continue
        for w in ws:
            j = int(np.searchsorted(times, t_a + w + 1e-9))
            cols["X"].append(moment_row(times[:j], values[b, :, :j], wind[b], t_a))
            for key, val in (
                ("z", lk.xyz[2]),
                ("w", float(w)),
                ("t_alarm", t_a),
                ("leak", b),
                ("riser", lk.pipe),
            ):
                cols[key].append(val)
    return HeightLibrary(**{k: np.array(v) for k, v in cols.items()})
