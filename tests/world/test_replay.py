"""3D replay export: a whole closed-loop run, compact enough for the browser."""

import base64
import json
from dataclasses import replace

import numpy as np

from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.crowd_study import RUNGS
from tabpfn_spatial.world.replay import GAS_LOG_MAX, GAS_LOG_MIN, decode_u8, make_replay

RUNG = replace(RUNGS[0], people=25, t_max=80.0)


def _rep():
    return make_replay(RUNG, seed=500, models="perfect")


def test_replay_has_the_plant_the_timeline_and_every_planner() -> None:
    r = json.loads(json.dumps(_rep()))
    assert r["scene"]["width"] == 60
    assert r["t_alarm"] is not None and r["t_alarm"] <= r["t_end"]
    assert set(r["planners"]) == {*PLANNERS, "once"}  # once: CCRP planned at the alarm only
    for p in PLANNERS:
        assert set(r["planners"][p]["metrics"]) >= {"harm_s_mean", "exposed_pct", "out_95_s"}


def test_gas_frames_decode_to_the_grid_and_match_the_simulation() -> None:
    r = _rep()
    ny, nx = len(r["scene"]["solid"]), len(r["scene"]["solid"][0])
    frames = [decode_u8(g, (ny, nx)) for g in r["gas"]["frames"]]
    assert len(frames) == len(r["gas"]["times"])
    last = frames[-1].astype(float) / 255 * (GAS_LOG_MAX - GAS_LOG_MIN) + GAS_LOG_MIN
    assert last.max() > -1  # somewhere above the toxic limit (log10 0.1 = -1)
    assert frames[0].max() == 0  # no gas before the leak


def test_people_tracks_cover_every_second_and_stay_on_open_cells() -> None:
    r = _rep()
    solid = np.array(r["scene"]["solid"], dtype=bool)
    n_t = len(r["times"])
    for p in PLANNERS:
        cells = np.frombuffer(base64.b64decode(r["planners"][p]["cells"]), dtype=np.int16)
        cells = cells.reshape(n_t, -1, 2)
        assert cells.shape[1] == 25
        assert not solid[cells[..., 0], cells[..., 1]].any()
        state = decode_u8(r["planners"][p]["state"], (n_t, 25))
        assert set(np.unique(state)) <= {0, 1, 2, 3}  # walking, queueing, in gas, out


def test_model_updates_carry_danger_maps_and_source_guesses() -> None:
    r = _rep()
    assert r["updates"] and r["updates"][0]["t"] == r["t_alarm"]
    u = r["updates"][0]
    ny, nx = len(r["scene"]["solid"]), len(r["scene"]["solid"][0])
    assert decode_u8(u["danger"], (ny, nx)).shape == (ny, nx)
    assert len(u["source_top"]) >= 1 and 0 <= u["source_top"][0]["p"] <= 1
