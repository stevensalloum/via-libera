import numpy as np

from tabpfn_spatial.sim import Factory
from tabpfn_spatial.sim.gases import LITRES_PER_RATE_UNIT, passive_distance
from tabpfn_spatial.source.features import n_features
from tabpfn_spatial.source.vertical import (
    HALL_HEIGHT,
    Physics,
    height_library,
    layouts,
    pick_risers,
    plume_centre,
    profile,
    sample_riser_leaks,
    vertical_factor,
)

NEUTRAL = Physics(sg=1.0, mixing=0.02, tilt=1.0)
LIGHT = Physics(sg=0.55, mixing=0.02, tilt=1.0)
HEAVY = Physics(sg=1.52, mixing=0.02, tilt=1.0)


def test_profile_keeps_all_gas_inside_the_hall() -> None:
    z = np.linspace(0.0, HALL_HEIGHT, 4001)
    for zc, sd in ((0.3, 0.5), (4.0, 1.0), (7.8, 2.5), (1.0, 6.0)):
        g = profile(z, zc, sd)
        assert abs(np.trapezoid(g, z) - 1.0) < 1e-3  # floor and ceiling reflect


def test_vertical_factor_averages_to_the_2d_value() -> None:
    # the 2D simulator gives the height-averaged concentration; the 3D one must agree
    z = np.linspace(0.0, HALL_HEIGHT, 2001)
    f = vertical_factor(z, z_leak=2.0, r=3.0, t=20.0, phys=LIGHT, rate=1.0, wind=0.5)
    assert abs(np.trapezoid(f, z) / HALL_HEIGHT - 1.0) < 1e-3


def test_gas_is_strongest_at_the_leak_height_without_buoyancy() -> None:
    z = np.array([1.0, 3.0, 5.0])
    f = vertical_factor(z, z_leak=3.0, r=1.0, t=10.0, phys=NEUTRAL, rate=1.0, wind=0.5)
    assert f[1] > f[0] and abs(f[0] - f[2]) < 1e-4 * f[0]  # floor/ceiling: tiny asymmetry


def test_light_gas_rises_heavy_gas_sinks() -> None:
    z = np.array([2.0, 4.0])  # one metre below and above a leak at 3 m
    up = vertical_factor(z, z_leak=3.0, r=1.0, t=10.0, phys=LIGHT, rate=1.0, wind=0.5)
    down = vertical_factor(z, z_leak=3.0, r=1.0, t=10.0, phys=HEAVY, rate=1.0, wind=0.5)
    assert up[1] > up[0] and down[0] > down[1]


def test_density_only_matters_near_the_leak() -> None:
    # beyond the passive distance (sim.gases) the gas stops rising and follows the air
    x_p = passive_distance(0.55, 1.0 * LITRES_PER_RATE_UNIT / 1000.0, 0.5)
    near = plume_centre(1.0, r=0.5 * x_p, phys=LIGHT, rate=1.0, wind=0.5)
    far = [plume_centre(1.0, r, LIGHT, 1.0, 0.5) for r in (2 * x_p, 5 * x_p)]
    assert 1.0 < near < far[0] == far[1] == min(1.0 + x_p, HALL_HEIGHT)
    assert plume_centre(1.0, 3 * x_p, NEUTRAL, 1.0, 0.5) == 1.0


def test_gas_spreads_in_height_over_time() -> None:
    z = np.array([3.0, 6.0])
    early = vertical_factor(z, 3.0, 1.0, 5.0, NEUTRAL, 1.0, 0.5)
    late = vertical_factor(z, 3.0, 1.0, 60.0, NEUTRAL, 1.0, 0.5)
    assert late[1] / late[0] > early[1] / early[0]


def test_layouts_one_height_masts_and_three_heights() -> None:
    f = Factory.default()
    risers = pick_risers(f, 4)
    assert len(risers) == 4
    lay = layouts(f, risers)
    k = len(f.sensors)
    assert len(lay["one_height"].z) == k and np.all(lay["one_height"].z == 1.5)
    assert len(lay["masts"].z) == k + 3 * 4
    assert len(lay["three_heights"].z) == 3 * k
    mast = lay["masts"]
    for r in risers:  # three heights next to every riser (within 1.5 m)
        near = np.linalg.norm(mast.xy - r, axis=1) < 1.5
        assert near.sum() == 3 and len(np.unique(mast.z[near])) == 3
    for s in lay.values():
        assert all(not f.walls[f.cell_of(p)] for p in s.xy)


def test_height_library_rows_and_targets() -> None:
    f = Factory.default()
    risers = pick_risers(f, 4)
    rng = np.random.default_rng(0)
    leaks = sample_riser_leaks(f, risers, 30, rng, gas="methane")
    assert all(0.5 <= lk.z <= 7.0 for lk in leaks)
    sensors = layouts(f, risers)["masts"]
    lib = height_library(f, leaks, sensors, np.random.default_rng(1), ws=(0.0, 10.0))
    assert lib.X.shape[1] == n_features(len(sensors.z))
    assert len(lib.X) == len(lib.z) == len(lib.w) == len(lib.leak) and len(lib.X) % 2 == 0
    np.testing.assert_allclose(lib.z, [leaks[i].z for i in lib.leak])
    again = height_library(f, leaks, sensors, np.random.default_rng(1), ws=(0.0, 10.0))
    np.testing.assert_array_equal(lib.X, again.X)


def test_mast_sees_a_high_leak_higher_up() -> None:
    f = Factory.default()
    risers = pick_risers(f, 4)
    sensors = layouts(f, risers)["masts"]
    leaks = sample_riser_leaks(f, risers, 40, np.random.default_rng(3), gas="methane")
    quiet = {"noise": 0.0, "floor": 0.0, "wind_noise": 0.0}
    lib = height_library(f, leaks, sensors, np.random.default_rng(0), ws=(10.0,), **quiet)
    k = len(sensors.z)
    top, bottom = np.flatnonzero(sensors.z == sensors.z.max()), np.flatnonzero(sensors.z == 1.0)
    ratio = lib.X[:, top].sum(axis=1) - lib.X[:, bottom].sum(axis=1)  # log1p readings
    high, low = lib.z > 5.0, lib.z < 2.5
    assert ratio[high].mean() > ratio[low].mean()
    assert lib.X.shape[1] == 2 * k + 3
