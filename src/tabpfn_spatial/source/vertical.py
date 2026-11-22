"""Leak height on vertical pipes: a temporary 2.5D stand-in until the simulator has heights.

STAND-IN (owner: Rayan, prototype). ``sim`` is 2D; this module adds height on top of it
without changing it, so the height regressor can be built and tested now. It should move
to ``sim/`` (Steven) if the team adopts it.

Model. The 2D simulator gives the height-averaged concentration ``c2d``. At a sensor at
height ``z_s``, horizontal distance ``r`` from the leak, ``t`` seconds after the leak start:

    c(z_s) = c2d * H * profile(z_s; centre, sd)

- ``profile``: a normal curve in height, reflected at floor and ceiling (no gas is lost),
  so its height average is exactly ``c2d``.
- ``centre`` = leak height + direction * tilt * min(r, x_p): light gas rises, heavy gas
  sinks, but only within the passive distance ``x_p`` (``sim.gases``, Britter & McQuaid
  1988); beyond it the gas follows the air (the 3D study's finding).
- ``sd^2 = SIGMA0^2 + (INTENSITY r)^2 + 2 K t``: plume spread with distance (Taylor, the
  same 10 % turbulence intensity as ``sim.gases``) plus indoor vertical mixing ``K``.

``K`` (mixing) and ``tilt`` are ASSUMPTIONS; each simulated leak draws its own from a
range (``sample_riser_leaks``) so a model cannot overfit one exact rule.
Cost: one batched 2D simulation plus O(B * T * k) for the height factor.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from tabpfn_spatial.sim import Factory, Scenario, sample_scenarios, simulate, wind_at
from tabpfn_spatial.sim.gases import GASES, INTENSITY, LITRES_PER_RATE_UNIT, passive_distance
from tabpfn_spatial.source.features import alarm_time, moment_row
from tabpfn_spatial.source.library import _BATCH, ALARM_WINDOW, DEFAULT_WS, EVERY

HALL_HEIGHT = 8.0  # m, floor to ceiling
SIGMA0 = 0.3  # m, plume half-width at the leak
BREATHING = 1.5  # m, the one-height network's mounting height
MAST_Z = (1.0, 3.75, 6.5)  # m, low / middle / high sensor on a mast or column
RISER_RANGE = (0.5, 7.0)  # m, where on a vertical pipe a leak can be
MIXING_RANGE = (0.005, 0.05)  # m^2/s, indoor vertical mixing K (log-uniform)
TILT_RANGE = (0.5, 1.5)  # metres of rise (or fall) per metre travelled, near the leak


@dataclass(frozen=True)
class Physics:
    """Vertical behaviour of one leak: gas density ``sg`` (air = 1), ``mixing``, ``tilt``."""

    sg: float
    mixing: float
    tilt: float


@dataclass(frozen=True)
class Sensors3D:
    """Detector positions: ``xy`` (k, 2) metres on the floor plan, ``z`` (k,) heights."""

    xy: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class Leak3D:
    """A 2D scenario plus the leak height ``z`` on riser number ``riser``."""

    scenario: Scenario
    z: float
    phys: Physics
    riser: int


def profile(z: np.ndarray, centre: np.ndarray, sd: np.ndarray, h: float = HALL_HEIGHT):
    """Normal density in height, reflected at 0 and ``h``; integrates to 1 over [0, h]."""
    z, centre, sd = np.asarray(z), np.asarray(centre), np.asarray(sd)
    out = 0.0
    for n in range(-3, 4):
        for c in (centre + 2 * n * h, -centre + 2 * n * h):
            out = out + np.exp(-0.5 * ((z - c) / sd) ** 2)
    return out / (sd * np.sqrt(2 * np.pi))


def plume_centre(z_leak: float, r, phys: Physics, rate: float, wind: float):
    """Height of the plume's centre at horizontal distance ``r`` (m) from the leak."""
    x_p = passive_distance(phys.sg, rate * LITRES_PER_RATE_UNIT / 1000.0, wind)
    direction = np.sign(1.0 - phys.sg)  # +1 rises, -1 sinks, 0 neutral
    lift = direction * phys.tilt * np.minimum(r, x_p)
    return np.clip(z_leak + lift, 0.0, HALL_HEIGHT)


def vertical_factor(z, z_leak: float, r, t, phys: Physics, rate: float, wind: float):
    """``c(z) / c2d`` at heights ``z``, distance ``r`` and time ``t`` (broadcasts)."""
    sd = np.sqrt(SIGMA0**2 + (INTENSITY * np.asarray(r)) ** 2 + 2 * phys.mixing * np.asarray(t))
    return HALL_HEIGHT * profile(z, plume_centre(z_leak, r, phys, rate, wind), sd)


def pick_risers(factory: Factory, n: int) -> np.ndarray:
    """(n, 2) leak sites spread over the plant (farthest-point order from site 0)."""
    sites = factory.leak_sites
    chosen = [0]
    d = np.linalg.norm(sites - sites[0], axis=1)
    while len(chosen) < min(n, len(sites)):
        chosen.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(sites - sites[chosen[-1]], axis=1))
    return sites[chosen]


def _mast_spot(factory: Factory, xy: np.ndarray) -> np.ndarray:
    """An open spot 1 m from a riser (first free direction)."""
    for step in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        p = xy + np.array(step)
        if not factory.walls[factory.cell_of(p)]:
            return p
    return xy.copy()


def layouts(factory: Factory, risers: np.ndarray) -> dict[str, Sensors3D]:
    """The three detector layouts compared in the height study.

    ``one_height``: the plant's detectors at breathing height (today).
    ``masts``: plus a low, middle and high detector 1 m from every vertical pipe.
    ``three_heights``: every detector position at the three mast heights.
    """
    base = np.asarray(factory.sensors, dtype=float)
    k = len(base)
    masts = np.array([_mast_spot(factory, r) for r in risers for _ in MAST_Z])
    return {
        "one_height": Sensors3D(base, np.full(k, BREATHING)),
        "masts": Sensors3D(
            np.vstack([base, masts]),
            np.concatenate([np.full(k, BREATHING), np.tile(MAST_Z, len(risers))]),
        ),
        "three_heights": Sensors3D(np.repeat(base, 3, axis=0), np.tile(MAST_Z, k)),
    }


def sample_riser_leaks(
    factory: Factory,
    risers: np.ndarray,
    n: int,
    rng: np.random.Generator,
    gas: str,
    z_range: tuple[float, float] = RISER_RANGE,
    drift: float = 0.0,
) -> list[Leak3D]:
    """``n`` leaks on the risers: size and wind as ``sample_scenarios``, height uniform."""
    scenarios = sample_scenarios(factory, n, rng, drift=drift)
    which = rng.integers(len(risers), size=n)
    z = rng.uniform(*z_range, size=n)
    mixing = np.exp(rng.uniform(*np.log(MIXING_RANGE), size=n))
    tilt = rng.uniform(*TILT_RANGE, size=n)
    sg = GASES[gas].sg
    rows = zip(scenarios, which, z, mixing, tilt, strict=True)
    return [
        Leak3D(replace(sc, source=risers[i].copy()), float(zi), Physics(sg, m, t), int(i))
        for sc, i, zi, m, t in rows
    ]


def clean_series(
    factory: Factory, leaks: Sequence[Leak3D], sensors: Sensors3D, times: np.ndarray
) -> np.ndarray:
    """Noise-free readings (B, k, t) of detectors at their heights."""
    cells = np.array([factory.cell_of(p) for p in sensors.xy])
    out = []
    for i in range(0, len(leaks), _BATCH):
        batch = leaks[i : i + _BATCH]
        fields = simulate(factory, [lk.scenario for lk in batch], times)
        c2d = fields[:, :, cells[:, 0], cells[:, 1]]  # (b, t, k)
        for b, lk in enumerate(batch):
            sc = lk.scenario
            r = np.linalg.norm(sensors.xy - sc.source, axis=1)[None, :]
            v = vertical_factor(sensors.z[None, :], lk.z, r, times[:, None], lk.phys,
                                sc.rate, float(np.linalg.norm(sc.wind)))  # fmt: skip
            out.append((c2d[b] * v).T)
    return np.array(out)


def noisy_series_3d(
    factory: Factory,
    leaks: Sequence[Leak3D],
    sensors: Sensors3D,
    rng: np.random.Generator,
    t_max: float,
    noise: float = 0.1,
    floor: float = 0.001,
    wind_noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``times``, noisy ``values`` (B, k, t), running-mean ``wind`` (B, t, 2).

    Same noise model as ``library.noisy_series``.
    """
    times = np.arange(0.0, t_max + 1e-9, EVERY)
    clean = clean_series(factory, leaks, sensors, times)
    noisy = clean * (1 + noise * rng.standard_normal(clean.shape))
    noisy += floor * rng.standard_normal(clean.shape)
    n = np.arange(1, len(times) + 1)[None, :, None]
    wind = np.stack([np.cumsum(wind_at(lk.scenario, times), axis=0) for lk in leaks]) / n
    wind = wind + wind_noise * rng.standard_normal((len(leaks), 1, 2))
    return times, np.maximum(noisy, 0.0), wind


@dataclass(frozen=True)
class HeightLibrary:
    """Training rows for the height regressor, one per (leak, moment)."""

    X: np.ndarray  # (rows, 2k + 3), the same row as the locator's
    z: np.ndarray  # (rows,) true leak height, metres
    w: np.ndarray  # (rows,) seconds since the alarm
    t_alarm: np.ndarray  # (rows,)
    leak: np.ndarray  # (rows,) index into the input list
    riser: np.ndarray  # (rows,) which vertical pipe


def height_library(
    factory: Factory,
    leaks: Sequence[Leak3D],
    sensors: Sensors3D,
    rng: np.random.Generator,
    ws: Sequence[float] = DEFAULT_WS,
    **noise: float,
) -> HeightLibrary:
    """Rows at each moment ``ws`` after each leak's alarm; leaks that never alarm are skipped."""
    times, values, wind = noisy_series_3d(
        factory, leaks, sensors, rng, ALARM_WINDOW + max(ws), **noise
    )
    cols: dict[str, list] = {k: [] for k in ("X", "z", "w", "t_alarm", "leak", "riser")}
    for b, lk in enumerate(leaks):
        t_a = alarm_time(times, values[b])
        if t_a is None or t_a > ALARM_WINDOW:
            continue
        for w in ws:
            j = int(np.searchsorted(times, t_a + w + 1e-9))
            cols["X"].append(moment_row(times[:j], values[b, :, :j], wind[b, j - 1], t_a))
            for key, val in (("z", lk.z), ("w", float(w)), ("t_alarm", t_a), ("leak", b),
                             ("riser", lk.riser)):  # fmt: skip
                cols[key].append(val)
    return HeightLibrary(**{k: np.array(v) for k, v in cols.items()})
