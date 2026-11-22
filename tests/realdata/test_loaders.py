"""Real dataset loaders: geometry facts and, when the data is on disk, the loaded arrays."""

from pathlib import Path

import numpy as np
import pytest

from tabpfn_spatial.realdata import orebro, redvapor

DATA = Path("data/real")


def test_redvapor_sources_follow_the_turntable() -> None:
    a = redvapor.source_xyz("A")
    np.testing.assert_allclose(a, [0.783, 0.017, 0.18])
    np.testing.assert_allclose(redvapor.source_xyz("B"), a)
    c, d = redvapor.source_xyz("C"), redvapor.source_xyz("D")
    assert np.linalg.norm(c[:2]) == pytest.approx(np.linalg.norm(a[:2]), abs=1e-9)
    assert c[1] < 0 < d[1] and c[2] == d[2] == 0.18  # C turned -22.5 deg, D +22.5 deg


def test_orebro_grid_has_three_heights_and_nine_poles() -> None:
    xyz = orebro.sensor_xyz()
    assert xyz.shape == (27, 3)
    np.testing.assert_allclose(sorted(set(xyz[:, 2])), [0.45, 1.20, 2.23])
    assert len({(x, y) for x, y, _ in xyz}) == 9
    assert (xyz[:, 0] > 0).all() and (xyz[:, 0] < orebro.ROOM[0]).all()
    assert orebro.EXPERIMENTS[7].source == (2.70, 0.10, 2.20)


@pytest.mark.skipif(not (DATA / "redvapor/voxel_maps.pkl").exists(), reason="data not downloaded")
def test_redvapor_voxel_maps_load() -> None:
    runs = redvapor.load_voxel_maps(DATA / "redvapor/voxel_maps.pkl")
    assert len(runs) == 8 and {r.setup for r in runs} == set("ABCD")
    r = runs[0]
    assert r.xyz.shape[1] == 3 and len(r.pid) == len(r.xyz) and r.wind.shape == (len(r.xyz), 3)
    assert r.wind[:, 0].mean() < 0  # wind blows towards -x


@pytest.mark.skipif(not (DATA / "orebro/Exp01.csv").exists(), reason="data not downloaded")
def test_orebro_log_loads() -> None:
    run = orebro.load(DATA / "orebro", 7)
    assert run.c.shape[1] == 27 and len(run.t) == len(run.c)
    assert run.t[1] - run.t[0] == pytest.approx(0.5)  # 2 Hz
    assert run.source == (2.70, 0.10, 2.20)
    assert run.wind_speed.max() < 5 and run.wind_dir.max() > 5  # m/s vs degrees
