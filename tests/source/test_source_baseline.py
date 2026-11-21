import numpy as np

from tabpfn_spatial.sim import Factory, Scenario, readings
from tabpfn_spatial.source import WeightedCentroidLocator, localisation_error


def test_baseline_returns_a_valid_map_near_the_leak() -> None:
    f = Factory.default()
    sc = Scenario(np.array([17.0, 21.0]), 1.0, np.array([0.0, 0.0]))  # right on a sensor
    est = WeightedCentroidLocator().locate(f, readings(f, sc, 60.0, np.random.default_rng(0)))
    assert est.prob.shape == (f.ny, f.nx)
    assert np.isclose(est.prob.sum(), 1.0) and np.all(est.prob[f.walls] == 0)
    assert localisation_error(f, est, sc.source) < 8.0


def test_error_of_a_perfect_map_is_zero() -> None:
    f = Factory.default()
    prob = np.zeros((f.ny, f.nx))
    prob[f.cell_of(np.array([35.0, 21.0]))] = 1.0
    from tabpfn_spatial.contracts import SourceEstimate

    centre = f.centre(*f.cell_of(np.array([35.0, 21.0])))
    assert localisation_error(f, SourceEstimate(prob), centre) == 0.0
