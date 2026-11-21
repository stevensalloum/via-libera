"""Gas pipe network: leaks happen only where a pipe runs, most often at its fittings."""

from collections import deque

import numpy as np
import pytest

from tabpfn_spatial.sim import FITTING_WEIGHT, Factory, sample_scenarios


def _connected(mask: np.ndarray) -> bool:
    """True if the True cells of ``mask`` form one 4-connected piece."""
    cells = {tuple(c) for c in np.argwhere(mask)}
    start = next(iter(cells))
    seen, todo = {start}, deque([start])
    while todo:
        y, x = todo.popleft()
        for n in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if n in cells and n not in seen:
                seen.add(n)
                todo.append(n)
    return seen == cells


def _touches(f: Factory, mask: np.ndarray, machine) -> bool:
    """True if a True cell of ``mask`` is next to (4-neighbour of) the machine's footprint."""
    c = f.centres()
    pad = f.cell
    near = (
        (c[..., 0] > machine.x0 - pad)
        & (c[..., 0] < machine.x1 + pad)
        & (c[..., 1] > machine.y0 - pad)
        & (c[..., 1] < machine.y1 + pad)
    )
    return bool((near & mask & ~f.walls).any())


FACTORIES = {
    "default": Factory.default(),
    **{f"random{s}": Factory.random(np.random.default_rng(s)) for s in (1, 2, 3)},
}


@pytest.mark.parametrize("name", FACTORIES)
def test_pipe_network_is_one_piece_with_fittings_on_it(name: str) -> None:
    f = FACTORIES[name]
    assert f.pipes is not None and f.fittings is not None
    assert f.pipes.shape == f.walls.shape
    assert _connected(f.pipes)
    assert not (f.fittings & ~f.pipes).any()
    assert f.fittings.sum() >= 3


@pytest.mark.parametrize("name", FACTORIES)
def test_every_hot_machine_has_a_gas_connection(name: str) -> None:
    f = FACTORIES[name]
    for m in f.machines:
        if m.hot:
            assert _touches(f, f.fittings, m), m.name


@pytest.mark.parametrize("name", FACTORIES)
def test_leak_sites_are_the_open_pipe_cells(name: str) -> None:
    f = FACTORIES[name]
    sites = {f.cell_of(s) for s in f.leak_sites}
    assert sites == {tuple(c) for c in np.argwhere(f.pipes & ~f.walls)}
    assert len(f.leak_weights) == len(f.leak_sites)
    on_fitting = np.array([f.fittings[f.cell_of(s)] for s in f.leak_sites])
    np.testing.assert_array_equal(f.leak_weights, np.where(on_fitting, FITTING_WEIGHT, 1.0))


def test_leak_prior_is_a_probability_map_on_the_pipes() -> None:
    f = Factory.default()
    p = f.leak_prior
    assert p.shape == f.walls.shape
    assert p.sum() == pytest.approx(1.0)
    assert (p[~(f.pipes & ~f.walls)] == 0).all()
    assert p[f.fittings & ~f.walls].min() > p[f.pipes & ~f.fittings & ~f.walls].max()


def test_sampled_leaks_sit_on_pipes_and_favour_fittings() -> None:
    f = Factory.default()
    scs = sample_scenarios(f, 2000, np.random.default_rng(0))
    cells = [f.cell_of(s.source) for s in scs]
    assert all(f.pipes[c] and not f.walls[c] for c in cells)
    share = np.mean([f.fittings[c] for c in cells])
    expected = f.leak_prior[f.fittings].sum()
    assert share == pytest.approx(expected, abs=0.04)
    assert share > f.fittings[f.pipes & ~f.walls].mean()  # more often than by area


def test_plant_without_pipes_keeps_the_old_machine_rule() -> None:
    old = Factory.default(pipes=False)
    assert old.pipes is None and old.fittings is None
    assert len(old.leak_sites) == 110  # the v0 baseline plant
    np.testing.assert_array_equal(old.leak_weights, np.ones(110))
    picks = np.random.default_rng(5).integers(110, size=4)  # the v0 draw
    got = [s.source for s in sample_scenarios(old, 4, np.random.default_rng(5))]
    np.testing.assert_array_equal(got, old.leak_sites[picks])


def test_random_pipes_are_reproducible() -> None:
    a = Factory.random(np.random.default_rng(7))
    b = Factory.random(np.random.default_rng(7))
    np.testing.assert_array_equal(a.pipes, b.pipes)
    np.testing.assert_array_equal(a.fittings, b.fittings)
