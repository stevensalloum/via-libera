import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tabpfn_spatial.sim import (
    Factory,
    Scenario,
    readings,
    sample_people,
    sample_scenarios,
    simulate,
)


@pytest.fixture(scope="module")
def factory() -> Factory:
    return Factory.default()


def test_default_layout(factory: Factory) -> None:
    assert factory.walls.shape == (factory.ny, factory.nx)
    assert factory.walls.any() and not factory.walls.all()
    assert any(m.hot for m in factory.machines)
    assert len(factory.exits) >= 3 and len(factory.sensors) >= 8
    for xy in np.vstack([factory.sensors, factory.exits]):
        assert not factory.walls[factory.cell_of(xy)]
    assert factory.ignition.shape == factory.walls.shape and factory.ignition.any()


def test_cell_roundtrip(factory: Factory) -> None:
    iy, ix = factory.cell_of(np.array([10.3, 7.9]))
    centre = factory.centre(iy, ix)
    assert abs(centre[0] - 10.3) <= factory.cell / 2 and abs(centre[1] - 7.9) <= factory.cell / 2


def _one(factory: Factory, source=(34.0, 20.0), wind=(0.0, 0.0), rate=1.0) -> Scenario:
    return Scenario(np.array(source), rate, np.array(wind))


def test_mass_grows_with_release_and_solids_stay_empty(factory: Factory) -> None:
    sc = _one(factory)
    fields = simulate(factory, [sc], np.array([10.0, 20.0]))
    assert fields.shape == (1, 2, factory.ny, factory.nx)
    assert np.all(fields >= 0) and np.all(fields[..., factory.walls] == 0)
    total = fields.sum(axis=(-2, -1))[0]
    assert total[1] > total[0] > 0
    assert total[1] == pytest.approx(sc.rate * 20.0, rel=0.05)  # far from open boundary


def test_plume_moves_downwind(factory: Factory) -> None:
    fields = simulate(factory, [_one(factory, wind=(1.0, 0.0))], np.array([30.0]))[0, 0]
    xs = factory.centres()[..., 0]
    centroid_x = (fields * xs).sum() / fields.sum()
    assert centroid_x > 34.0 + 5.0


def test_source_inside_a_machine_is_rejected(factory: Factory) -> None:
    with pytest.raises(ValueError, match="solid"):
        simulate(factory, [_one(factory, source=(22.0, 20.0))], np.array([1.0]))


@settings(max_examples=10, deadline=None)
@given(st.integers(0, 10_000))
def test_sampled_scenarios_are_valid(seed: int) -> None:
    f = Factory.default()
    rng = np.random.default_rng(seed)
    for sc in sample_scenarios(f, 5, rng):
        assert not f.walls[f.cell_of(sc.source)]
        assert sc.rate > 0 and np.linalg.norm(sc.wind) <= 2.0
    people = sample_people(f, 7, rng)
    assert people.shape == (7, 2)
    assert not any(f.walls[f.cell_of(p)] for p in people)


def test_readings_shape_and_noise(factory: Factory) -> None:
    rng = np.random.default_rng(0)
    sc = _one(factory, wind=(0.5, 0.2))
    r = readings(factory, sc, t_obs=60.0, rng=rng, every=5.0)
    assert r.values.shape == (len(factory.sensors), len(r.times))
    assert r.times[0] == 0.0 and r.t_obs == 60.0
    assert np.all(r.values >= 0)
    np.testing.assert_allclose(r.wind, sc.wind, atol=0.3)
