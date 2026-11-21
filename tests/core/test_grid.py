import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tabpfn_spatial.core.grid import Grid2D


def test_points_and_image_roundtrip() -> None:
    g = Grid2D((0.0, 1.0), (0.0, 2.0), nx=3, ny=4)
    pts = g.points()
    assert pts.shape == (12, 2) and g.size == 12
    img = g.to_image(pts[:, 1])
    assert img.shape == (4, 3)
    np.testing.assert_allclose(img[:, 0], np.linspace(0, 2, 4))  # rows follow y
    np.testing.assert_allclose(img[0], 0.0)


def test_extent_for_imshow() -> None:
    assert Grid2D((0, 1), (2, 3), 2, 2).extent == (0, 1, 2, 3)


def test_with_features_appends_constant_columns() -> None:
    g = Grid2D((0, 1), (0, 1), 2, 2)
    rows = g.points(extra=[7.0, 8.0])
    assert rows.shape == (4, 4)
    assert np.all(rows[:, 2] == 7.0) and np.all(rows[:, 3] == 8.0)


def test_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError):
        Grid2D((0, 1), (0, 1), 0, 2)
    with pytest.raises(ValueError):
        Grid2D((0, 1), (0, 1), 2, 2).to_image(np.zeros(3))


@settings(max_examples=25, deadline=None)
@given(st.integers(1, 30), st.integers(1, 30))
def test_cell_index_inverts_points(nx: int, ny: int) -> None:
    g = Grid2D((-1.0, 1.0), (0.0, 5.0), nx, ny)
    pts = g.points()
    np.testing.assert_array_equal(g.cell_index(pts), np.arange(g.size))
