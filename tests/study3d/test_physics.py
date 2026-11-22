"""Physics ladder checks against conservation laws, not against the code's own output."""

import numpy as np
import pytest

from tabpfn_spatial.study3d import physics as ph
from tabpfn_spatial.study3d.geometry import Grid


def _flow(**kw: object) -> ph.Flow:
    z = np.array([0.0, 2.0])
    base = {"direction": (1.0, 0.0), "z": z, "speed": np.full(2, 1.0),
            "sigma_v": np.full(2, 0.1), "sigma_w": np.full(2, 0.08)}  # fmt: skip
    base.update(kw)
    return ph.Flow(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("x", [0.5, 2.0, 5.0])
def test_plume_mass_flux_equals_release_rate(x: float) -> None:
    f = _flow()
    y, z = np.meshgrid(np.linspace(-4, 4, 401), np.linspace(0, 6, 601), indexing="ij")
    pts = np.c_[np.full(y.size, x), y.ravel(), z.ravel()]
    c = ph.plume(pts, np.array([0.0, 0.0, 0.3]), f).reshape(y.shape)
    flux = 1.0 * c.sum() * (8 / 400) * (6 / 600)  # U * integral of c over the plane
    assert flux == pytest.approx(1.0, rel=0.02)


def test_plume_is_zero_upwind() -> None:
    c = ph.plume(np.array([[-1.0, 0.0, 0.3]]), np.array([0.0, 0.0, 0.3]), _flow())
    assert c[0] == 0.0


def test_transport_conserves_mass_in_a_duct() -> None:
    g = Grid.box((0.0, 0.0, 0.0), (2.0, 0.8, 0.4), 0.05)
    solid = np.zeros(g.shape, bool)
    f = _flow(open_faces=ph.CLOSED_DUCT)
    w = ph.base_wind(g, solid, f)
    c = ph.transport(g, solid, f, w, np.array([0.2, 0.4, 0.2]), rate=1.0)
    out = (w.ux[-1] * c[-1]).sum() * g.cell**2  # outflow at x1 carries the cell value
    assert out == pytest.approx(1.0, rel=1e-3)
    assert c.min() >= -1e-9


def test_mass_consistent_wind_goes_around_a_block() -> None:
    g = Grid.box((0.0, 0.0, 0.0), (2.0, 1.0, 0.8), 0.05)
    solid = np.zeros(g.shape, bool)
    solid[18:22, 8:12, :6] = True  # a 0.2 x 0.2 x 0.3 m block
    f = _flow()
    w0 = ph.base_wind(g, solid, f)
    assert np.abs(ph.divergence(g, solid, f, w0)).max() > 1.0  # L2 piles up in front
    w = ph.mass_consistent(g, solid, f, w0)
    assert np.abs(ph.divergence(g, solid, f, w)).max() < 1e-4
    beside = w.uy[17, 12, 3]  # face just beside the block, upwind corner: flow turns sideways
    above = w.uz[18, 10, 7]  # one face above the block's upwind half: flow goes up
    assert beside > 0.02 and above > 0.02


def test_source_on_top_of_a_tank_goes_to_the_air_above() -> None:
    g = Grid.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.1)
    solid = np.zeros(g.shape, bool)
    solid[4:6, 4:6, :2] = True  # tank 0.2 m tall
    assert ph.source_cell(g, solid, np.array([0.45, 0.45, 0.15])) == (4, 4, 2)


def test_adjoint_matches_forward_solves() -> None:
    g = Grid.box((0.0, 0.0, 0.0), (1.0, 0.5, 0.3), 0.05)
    solid = np.zeros(g.shape, bool)
    solid[8:10, 4:6, :3] = True
    f = _flow()
    w = ph.mass_consistent(g, solid, f, ph.base_wind(g, solid, f))
    sensors = np.array([[0.8, 0.25, 0.1], [0.6, 0.1, 0.2]])
    adj = ph.adjoint(g, solid, f, w, sensors)
    for leak in ([0.12, 0.27, 0.07], [0.3, 0.12, 0.17]):
        fwd = ph.transport(g, solid, f, w, np.array(leak))
        i = g.index(np.array([leak]))[0]
        for k, s in enumerate(sensors):
            assert adj[k][tuple(i)] == pytest.approx(ph.sample(g, fwd, s[None])[0], rel=1e-6)
