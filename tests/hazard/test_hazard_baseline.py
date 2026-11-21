import numpy as np

from tabpfn_spatial.contracts import SourceEstimate
from tabpfn_spatial.hazard import ForwardSimPredictor, brier
from tabpfn_spatial.sim import Factory, Scenario, readings


def _setup():
    f = Factory.default()
    sc = Scenario(np.array([35.0, 21.0]), 1.5, np.array([0.8, 0.0]))
    r = readings(f, sc, 60.0, np.random.default_rng(0))
    prob = np.zeros((f.ny, f.nx))
    prob[f.cell_of(sc.source)] = 1.0
    return f, sc, r, SourceEstimate(prob, rate=np.array([1.5]), rate_levels=np.array([0.5]))


def test_forward_sim_maps_are_probabilities_and_hit_the_source() -> None:
    f, sc, r, est = _setup()
    times = np.array([90.0, 120.0])
    hz = ForwardSimPredictor().predict(f, r, est, times)
    assert hz.p_toxic.shape == (2, f.ny, f.nx) and hz.p_fire.shape == (2, f.ny, f.nx)
    assert np.all((hz.p_toxic >= 0) & (hz.p_toxic <= 1))
    assert hz.p_toxic[0][f.cell_of(sc.source)] == 1.0
    assert np.all(hz.p_fire <= f.ignition)  # fire only next to ignition sources
    toxic_t, _ = hz.at(118.0)
    np.testing.assert_array_equal(toxic_t, hz.p_toxic[1])


def test_brier_is_zero_for_perfect_and_positive_otherwise() -> None:
    truth = np.array([[0, 1], [1, 0]], dtype=bool)
    assert brier(truth.astype(float), truth) == 0.0
    assert brier(np.full((2, 2), 0.5), truth) == 0.25
