from dataclasses import replace
from pathlib import Path

from tabpfn_spatial.evac.crowd_plan import PLANNERS
from tabpfn_spatial.evac.crowd_study import RUNGS
from tabpfn_spatial.evac.crowd_viewer import record, save_gif, save_still


def test_record_still_and_gif(tmp_path: Path) -> None:
    rec = record(replace(RUNGS[0], people=20, t_max=60.0), seed=1)
    assert set(rec.frames) == set(PLANNERS)
    assert len(rec.frames["ccrp"]) == 61  # one snapshot per second
    assert save_still(rec, tmp_path / "s.png").stat().st_size > 20_000
    assert save_gif(rec, tmp_path / "g.gif", every=20).stat().st_size > 5_000
