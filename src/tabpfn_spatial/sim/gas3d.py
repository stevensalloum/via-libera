"""Time-stepped 3D gas in a ``Plant3D``: the inputs of the team's 3D models, all explicit.

**Inputs**
- ``Leak3D``: position (x, y, z) in metres, rate (model units/s, as in 2D), the plant's gas.
- ``Ventilation``: horizontal air ``speed`` (m/s) and ``direction`` (rad), direction ``drift``
  (rad / sqrt(s), a random walk; 0 = steady), mixing ``diffusivity`` (m^2/s, the same
  meaning and default as the 2D simulator), extraction through roof vents. Ahmad's stress
  conditions map onto it with ``Ventilation.from_condition``; a 2D ``Scenario`` with
  ``Ventilation.from_scenario``.

**Physics** (``study3d.physics``, the L3 level of the 3D study)
- wind: the horizontal air made mass-consistent around walls and machines (air leaves
  through the roof vents), re-solved every ``update_s`` seconds when it drifts;
- transport: the same finite-volume operator A as the steady L3 solve, stepped in time
  (implicit Euler, ``DT`` s): (V/dt + A) c_new = V/dt c_old + source;
- density near the leak: within the height-zone radius of ``sim.gases`` (Britter & McQuaid
  passive limit) light gas rises and heavy gas sinks with the buoyant-plume velocity scale
  w = (F / r)^(1/3), F = g |SG - 1| q0 / pi (Briggs), q0 the pure-gas release; none beyond it.

**Outputs** (``Run3D``): concentration volumes every ``every`` s; noisy detector readings
with detector heights; the sensor-triggered alarm; arrival time per detector; the
breathing-height map on the 2D grid for the planner; volumes in (time, z, y, x) for the
hazard model.

Units: concentrations are per m^3 of air in the same model units as 2D; ``RATE_SCALE`` turns
the 2D rate into a 3D source so the alarm and toxic thresholds keep their 2D meaning
(calibrated on a neutral gas, see the constant).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import splu

from tabpfn_spatial.contracts import SensorReadings
from tabpfn_spatial.sim.gases import GASES, LITRES_PER_RATE_UNIT, passive_distance
from tabpfn_spatial.sim.plant3d import Plant3D
from tabpfn_spatial.sim.plume import DIFFUSIVITY
from tabpfn_spatial.sim.scenarios import ALARM_LEVEL
from tabpfn_spatial.study3d import physics as ph

DT = 1.0  # s, time step
# 2D rate units -> 3D source. Calibrated: a neutral gas (CO) leaking at 1.6 m covers the same
# area above the alarm level at breathing height as the 2D simulator (default plant, 6 leaks,
# 60-90 s: median 19, range 12-33), so the 2D thresholds keep their meaning in 3D.
RATE_SCALE = 19.4
BREATHING = (0.5, 2.0)  # m: crouching to standing breathing heights
W_MAX = 1.0  # m/s cap on the buoyant velocity


@dataclass(frozen=True)
class Leak3D:
    """One leak: where, how strong (2D model units/s), which gas."""

    xyz: tuple[float, float, float]
    rate: float
    gas: str = "methane"


@dataclass(frozen=True)
class Ventilation:
    """How the air moves (all explicit inputs)."""

    speed: float = 0.5  # m/s
    direction: float = 0.0  # rad, the direction the air moves towards
    drift: float = 0.0  # rad / sqrt(s); 0 = steady
    diffusivity: float = DIFFUSIVITY  # m^2/s
    update_s: float = 30.0  # s between wind re-solves when drifting
    seed: int = 0

    @classmethod
    def from_scenario(cls, sc, diffusivity: float = DIFFUSIVITY) -> "Ventilation":
        """The air of a 2D ``Scenario`` (same speed, direction, drift and seed)."""
        u, v = (float(a) for a in sc.wind)
        return cls(float(np.hypot(u, v)), float(np.arctan2(v, u)), float(sc.drift),
                   diffusivity, seed=int(sc.seed))  # fmt: skip

    @classmethod
    def from_condition(cls, name: str, speed: float = 0.5, direction: float = 0.0,
                       seed: int = 0) -> "Ventilation":  # fmt: skip
        """One of Ahmad's stress conditions (``sim.conditions.CONDITIONS``)."""
        from tabpfn_spatial.sim.conditions import CONDITIONS

        c = CONDITIONS[name]
        return cls(speed, direction, float(c.drift), float(c.diffusivity), seed=seed)

    def directions(self, t_end: float) -> np.ndarray:
        """Wind direction for each ``update_s`` window up to ``t_end`` (random walk)."""
        n = int(np.ceil(t_end / self.update_s)) + 1
        rng = np.random.default_rng(self.seed)
        steps = self.drift * np.sqrt(self.update_s) * rng.standard_normal(n)
        steps[0] = 0.0
        return self.direction + np.cumsum(steps)


@dataclass
class Run3D:
    """A simulated leak: volumes every ``every`` s and everything derived from them."""

    plant: Plant3D
    leak: Leak3D
    vent: Ventilation
    times: np.ndarray  # (T,)
    volumes: np.ndarray  # (T, nx, ny, nz) concentration
    extras: dict = field(default_factory=dict)

    def at_points(self, xyz: np.ndarray) -> np.ndarray:
        """(T, k) clean concentration at points (voxel lookup)."""
        idx = self.plant.grid.index(np.asarray(xyz, float))
        return self.volumes[:, idx[:, 0], idx[:, 1], idx[:, 2]]

    def readings(self, rng: np.random.Generator, noise: float = 0.1, floor: float = 0.001,
                 dropout: float = 0.0, stuck: float = 0.0) -> SensorReadings:  # fmt: skip
        """Noisy detector readings every ``every`` s (same noise model as 2D), with heights."""
        det, _ = self.plant.detectors
        clean = self.at_points(det).T  # (k, T)
        noisy = clean * (1 + noise * rng.standard_normal(clean.shape))
        noisy += floor * rng.standard_normal(clean.shape)
        values = np.maximum(noisy, 0.0)
        k = len(values)
        is_stuck = rng.uniform(size=k) < stuck
        values[is_stuck] = values[is_stuck, :1]
        values[rng.uniform(size=k) < dropout] = np.nan
        wind = self.vent.speed * np.array(
            [np.cos(self.vent.direction), np.sin(self.vent.direction)]
        )
        return SensorReadings(det[:, :2].copy(), self.times.copy(), values, wind,
                              heights=det[:, 2].copy())  # fmt: skip

    def alarm(self, readings: SensorReadings, level: float = ALARM_LEVEL) -> float | None:
        """First sample time where any (noisy) detector reads above ``level``."""
        hit = np.nan_to_num(readings.values) > level
        cols = np.flatnonzero(hit.any(axis=0))
        return float(readings.times[cols[0]]) if cols.size else None

    def arrivals(self, level: float = ALARM_LEVEL) -> np.ndarray:
        """(k,) first sample time each detector's clean value exceeds ``level`` (NaN: never)."""
        det, _ = self.plant.detectors
        over = self.at_points(det) > level
        first = np.where(over.any(axis=0), over.argmax(axis=0), -1)
        return np.where(first >= 0, self.times[np.maximum(first, 0)], np.nan)

    def breathing_map(self) -> np.ndarray:
        """(T, ny, nx) on the factory grid: the worst concentration at breathing heights.

        Max over the voxels between ``BREATHING`` heights, averaged over each 2D cell's
        voxels (a person stands or crouches somewhere in that cell).
        """
        g, f = self.plant.grid, self.plant.f
        zs = g.axis(2)
        layer = self.volumes[..., (zs >= BREATHING[0]) & (zs <= BREATHING[1])].max(axis=-1)
        r = round(f.cell / g.cell)
        T = layer.shape[0]
        cells = layer[:, : f.nx * r, : f.ny * r].reshape(T, f.nx, r, f.ny, r).mean(axis=(2, 4))
        return np.transpose(cells, (0, 2, 1))

    def volumes_zyx(self) -> np.ndarray:
        """(T, nz, ny, nx) for the hazard model."""
        return np.moveaxis(self.volumes, (1, 2, 3), (3, 2, 1))


def _buoyancy(plant: Plant3D, leak: Leak3D, w: ph.Faces) -> ph.Faces:
    """Add the density-driven vertical velocity near the leak (zero beyond the zone)."""
    sg = GASES[leak.gas].sg
    if abs(sg - 1.0) < 0.05:
        return w
    q0 = leak.rate * LITRES_PER_RATE_UNIT / 1000.0  # m^3/s of pure gas
    radius = passive_distance(sg, q0, 0.5)
    flux = 9.81 * abs(sg - 1.0) * q0 / np.pi
    g = plant.grid
    xs, ys = g.axis(0), g.axis(1)
    r = np.hypot(xs[:, None] - leak.xyz[0], ys[None, :] - leak.xyz[1])
    wb = np.where(r < radius, np.minimum(W_MAX, (flux / np.maximum(r, 0.5)) ** (1 / 3)), 0.0)
    wb *= 1.0 if sg < 1.0 else -1.0
    uz = w.uz.copy()
    blocked = ph._blocked(plant.solid, 2)[:, :, 1:-1]
    uz[:, :, 1:-1] += np.where(blocked, 0.0, wb[:, :, None])
    return ph.Faces(w.ux, w.uy, uz)


def _wind(plant: Plant3D, vent: Ventilation, angle: float) -> tuple[ph.Flow, ph.Faces]:
    flow = ph.Flow(direction=(float(np.cos(angle)), float(np.sin(angle))),
                   z=np.array([0.0, 9.0]), speed=np.full(2, vent.speed),
                   sigma_v=np.full(2, 0.1), sigma_w=np.full(2, 0.1),
                   open_faces=("top",))  # fmt: skip
    w0 = ph.base_wind(plant.grid, plant.solid, flow)
    return flow, ph.mass_consistent(plant.grid, plant.solid, flow, w0)


def simulate(plant: Plant3D, leak: Leak3D, vent: Ventilation, t_end: float = 300.0,
             every: float = 5.0) -> Run3D:  # fmt: skip
    """Step the gas from t = 0 to ``t_end``; keep a volume every ``every`` s."""
    g, solid = plant.grid, plant.solid
    ids = np.full(solid.shape, -1)
    ids[~solid] = np.arange(int((~solid).sum()))
    n = int((~solid).sum())
    vol = g.cell**3
    src = ids[ph.source_cell(g, solid, np.array(leak.xyz, float))]
    k_field = np.full(solid.shape, vent.diffusivity)
    angles = vent.directions(t_end)
    c = np.zeros(n)
    times = np.arange(0.0, t_end + 1e-9, every)
    out = np.zeros((len(times), *solid.shape), dtype=np.float32)
    lu, window = None, -1
    k_out = 1
    steps = round(t_end / DT)
    for s in range(1, steps + 1):
        t = s * DT
        wi = int((t - DT) // vent.update_s) if vent.drift > 0 else 0
        if wi != window:
            flow, w = _wind(plant, vent, float(angles[wi]))
            w = _buoyancy(plant, leak, w)
            a, _ = ph.transport_matrix(g, solid, flow, w, k_field)
            lu = splu((identity(n, format="csc") * (vol / DT) + a).tocsc())
            window = wi
        rhs = c * (vol / DT)
        rhs[src] += leak.rate * RATE_SCALE
        c = np.maximum(lu.solve(rhs), 0.0)
        while k_out < len(times) and times[k_out] <= t + 1e-9:
            out[k_out][~solid] = c
            k_out += 1
    return Run3D(plant, leak, vent, times, out)
