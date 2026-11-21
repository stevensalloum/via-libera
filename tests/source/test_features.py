import numpy as np

from tabpfn_spatial.contracts import SensorReadings
from tabpfn_spatial.source.features import (
    DEFAULT_TIMES,
    feature_matrix,
    leak_features,
    n_features,
)


def _readings(values: np.ndarray, wind: tuple[float, float] = (0.5, -0.2)) -> SensorReadings:
    k, t = values.shape
    times = np.arange(t) * 5.0
    return SensorReadings(np.zeros((k, 2)), times, values, np.array(wind))


def test_feature_layout_logs_arrivals_and_wind() -> None:
    # 2 sensors, samples 0..60 s every 5 s; sensor 0 ramps up, sensor 1 never sees gas
    t = np.arange(13) * 5.0
    values = np.vstack([t / 60.0, np.zeros(13)])
    x = leak_features(_readings(values), level=0.1)
    assert x.shape == (n_features(2),) == (2 * len(DEFAULT_TIMES) + 2 + 2,)
    logs = x[: 2 * len(DEFAULT_TIMES)].reshape(2, len(DEFAULT_TIMES))
    np.testing.assert_allclose(logs[0], np.log1p(np.array(DEFAULT_TIMES) / 60.0))
    np.testing.assert_allclose(logs[1], 0.0)
    arrival = x[2 * len(DEFAULT_TIMES) : -2]
    assert arrival[0] == 10.0  # first sample above 0.1 is t = 10 s (value 1/6)
    assert arrival[1] == 65.0  # never: one step after the last sample
    np.testing.assert_allclose(x[-2:], [0.5, -0.2])


def test_batch_matrix_matches_single_rows() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 0.5, size=(4, 3, 13))
    winds = rng.normal(size=(4, 2))
    times = np.arange(13) * 5.0
    X = feature_matrix(times, values, winds)
    assert X.shape == (4, n_features(3))
    for i in range(4):
        np.testing.assert_allclose(X[i], leak_features(_readings(values[i], tuple(winds[i]))))


def test_feature_time_uses_latest_sample_not_after_it() -> None:
    times = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
    values = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    X = feature_matrix(times, values[None], np.zeros((1, 2)), at=(20.0, 60.0))
    np.testing.assert_allclose(X[0, :2], np.log1p([1.0, 4.0]))  # 20 s -> sample at 15 s
