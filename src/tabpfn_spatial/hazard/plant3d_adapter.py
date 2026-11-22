"""Experimental handover from the shared 3D plant and source estimate.

The source contract provides separate horizontal and height distributions, not
their joint distribution. Their product below assumes independence. Combinations
inside solids or outside the plant are discarded and their mass is reported.
"""

import numpy as np

from tabpfn_spatial.contracts import SourceEstimate
from tabpfn_spatial.hazard.volume3d import VolumeGrid, validate_times
from tabpfn_spatial.hazard.volume3d_predictor import features_3d
from tabpfn_spatial.sim.gas3d import Leak3D, Ventilation, simulate
from tabpfn_spatial.sim.plant3d import Plant3D


def plant_grid(plant: Plant3D, *, ignition_radius: float) -> VolumeGrid:
    """Use the simulator's exact solids and mark open voxels near hot boxes.

    The ignition radius is a modelling assumption, not a validated fire model.
    """
    if not np.isfinite(ignition_radius) or ignition_radius <= 0:
        raise ValueError("ignition radius must be finite and positive")
    solid = plant.to_zyx(plant.solid)
    cell = float(plant.grid.cell)
    blank = VolumeGrid(solid, np.zeros_like(solid), cell, cell)
    centres = blank.centres()
    ignition = np.zeros_like(solid)
    for machine in plant.machines:
        if machine["hot"]:
            box = np.asarray(machine["box"], dtype=float)
            delta = np.maximum(np.maximum(box[:3] - centres, centres - box[3:]), 0.0)
            ignition |= np.linalg.norm(delta, axis=-1) <= ignition_radius
    return VolumeGrid(solid, ignition & ~solid, cell, cell)


def source_hypotheses(
    plant: Plant3D, grid: VolumeGrid, estimate: SourceEstimate
) -> tuple[np.ndarray, np.ndarray, float]:
    """Combine horizontal probabilities and height quantiles into weighted 3D voxels.

    Quantiles represent bins whose edges are halfway between their probability
    levels, with the tails assigned to the first and last bins. Horizontal and
    height estimates are assumed independent; combinations are not restricted
    to the 3D pipe network. Return positions, normalised weights and retained
    mass before normalisation, after excluding solid and out-of-bounds voxels.
    """
    prob = np.asarray(estimate.prob, dtype=float)
    if (prob.shape != plant.f.walls.shape or np.any(~np.isfinite(prob))
            or np.any(prob < 0) or not np.isclose(prob.sum(), 1.0)):  # fmt: skip
        raise ValueError("source map must be a finite probability map on the factory grid")
    if estimate.height is None or estimate.height_levels is None:
        raise ValueError("source estimate requires height quantiles and levels")
    z = np.asarray(estimate.height, dtype=float)
    levels = np.asarray(estimate.height_levels, dtype=float)
    if (z.ndim != 1 or not len(z) or z.shape != levels.shape
            or not np.isfinite(z).all() or not np.isfinite(levels).all()
            or np.any(np.diff(z) < 0) or np.any(np.diff(levels) <= 0)
            or np.any((levels <= 0) | (levels >= 1))):  # fmt: skip
        raise ValueError("height quantiles and levels must be finite and ordered")
    if (grid.cell != plant.grid.cell or grid.dz != plant.grid.cell
            or not np.array_equal(grid.solid, plant.to_zyx(plant.solid))):  # fmt: skip
        raise ValueError("volume grid does not match plant")
    edges = np.r_[0.0, (levels[:-1] + levels[1:]) / 2.0, 1.0]
    z_weights = np.diff(edges)
    ys, xs = np.nonzero(prob > 0)
    xy = plant.f.centres()[ys, xs]
    points = np.column_stack((np.repeat(xy, len(z), axis=0), np.tile(z, len(xy))))
    weights = np.repeat(prob[ys, xs], len(z)) * np.tile(z_weights, len(xy))
    bounds = np.array(grid.solid.shape[::-1]) * [grid.cell, grid.cell, grid.dz]
    inside = np.all((points >= 0) & (points < bounds), axis=1)
    points, weights = points[inside], weights[inside]
    idx = grid.indices(points)
    valid = ~grid.solid[tuple(idx.T)]
    idx, weights = idx[valid], weights[valid]
    retained = float(weights.sum())
    if retained <= 0:
        raise ValueError("source estimate has no mass in open voxels")
    unique, inverse = np.unique(idx, axis=0, return_inverse=True)
    combined = np.bincount(inverse, weights=weights)
    positions = (unique[:, ::-1] + 0.5) * [grid.cell, grid.cell, grid.dz]
    return positions, combined / retained, retained


def plant_training_rows(
    plant: Plant3D,
    grid: VolumeGrid,
    *,
    n_scenarios: int,
    cells_per_scenario: int,
    times: np.ndarray,
    seed: int,
    diffusivity_range: tuple[float, float] = (0.4, 1.6),
    near: float = 0.0,
    near_m: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate labelled rows from independent simulated pipe leaks in this plant.

    ``near``: fraction of each leak's cells drawn within ``near_m`` (horizontal) of the
    leak, where the gas is (0 = uniform over the hall, the default). Times are positive
    multiples of five seconds. Simulator concentrations are
    relative model units. Generate evaluation leaks with a different seed.
    """
    times = validate_times(times)
    if (n_scenarios < 1 or cells_per_scenario < 1 or np.any(times <= 0)
            or not np.allclose(times / 5, np.round(times / 5))):  # fmt: skip
        raise ValueError("positive scenario counts and five-second times are required")
    if (grid.cell != plant.grid.cell or grid.dz != plant.grid.cell
            or not np.array_equal(grid.solid, plant.to_zyx(plant.solid))):  # fmt: skip
        raise ValueError("training grid must match the simulator solids")
    lo, hi = diffusivity_range
    if not np.isfinite([lo, hi]).all() or lo <= 0 or hi < lo:
        raise ValueError("invalid diffusivity range")
    sites, prior = plant.leak_sites
    ids = grid.indices(sites)
    valid = ~grid.solid[tuple(ids.T)]
    sites, prior = sites[valid], prior[valid]
    if not len(sites) or prior.sum() <= 0:
        raise ValueError("plant must have open leak sites")
    rng = np.random.default_rng(seed)
    positions = grid.centres()[~grid.solid]
    rows, targets = [], []
    for _ in range(n_scenarios):
        source = sites[rng.choice(len(sites), p=prior / prior.sum())]
        rate = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        speed, angle = rng.uniform(0.2, 1.5), rng.uniform(0.0, 2 * np.pi)
        diffusion = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        wind = speed * np.array([np.cos(angle), np.sin(angle)])
        run = simulate(
            plant,
            Leak3D(tuple(source), rate, plant.f.gas),
            Ventilation(speed, angle, diffusivity=diffusion),
            float(times[-1]),
        )
        n_near = round(near * cells_per_scenario)
        close = np.flatnonzero(np.hypot(*(positions[:, :2] - source[:2]).T) <= near_m)
        picks = rng.integers(len(positions), size=cells_per_scenario - n_near)
        if n_near and len(close):
            picks = np.r_[picks, close[rng.integers(len(close), size=n_near)]]
        cells = positions[picks]
        indices = grid.indices(cells)
        volumes = run.volumes_zyx()
        for t in times:
            rows.append(features_3d(grid, source, wind, rate, float(t), cells))
            targets.append(volumes[round(t / 5), indices[:, 0], indices[:, 1], indices[:, 2]])
    return np.concatenate(rows), np.concatenate(targets)
