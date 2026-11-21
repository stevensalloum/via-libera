import numpy as np
import pytest

from tabpfn_spatial.core.dist import QuantileDist
from tabpfn_spatial.hazard.batching import ChunkedRegressor


class EchoRegressor:
    def __init__(self):
        self.sizes = []

    def fit(self, x, y):
        return self

    def predict_dist(self, x):
        self.sizes.append(len(x))
        mean = x[:, 0]
        return QuantileDist(np.array([0.1, 0.9]), np.column_stack([mean - 1, mean + 1]), mean, mean)


def test_chunked_regressor_preserves_order_and_all_distribution_fields() -> None:
    model = EchoRegressor()
    wrapper = ChunkedRegressor(model, max_rows=2)
    x = np.arange(5.0).reshape(-1, 1)
    result = wrapper.fit(x, x[:, 0]).predict_dist(x)
    assert model.sizes == [2, 2, 1]
    np.testing.assert_array_equal(result.mean, x[:, 0])
    np.testing.assert_array_equal(result.mode, x[:, 0])
    np.testing.assert_array_equal(result.values, np.column_stack([x[:, 0] - 1, x[:, 0] + 1]))


def test_chunked_regressor_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        ChunkedRegressor(EchoRegressor(), max_rows=0)
    with pytest.raises(ValueError):
        ChunkedRegressor(EchoRegressor()).predict_dist(np.empty((0, 7)))
