import json
from pathlib import Path

from tabpfn_spatial.milestone import _readme


def test_readme_indexes_files_and_tabulates_crowd_results(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    rows = [
        {"rung": "B", "models": m, "planner": p, "seed": s, "harm_s_mean": 1.0, "harm_s_p95": 2.0,
         "exposed_pct": 5.0, "out_95_s": 90.0, "queued_s_mean": 3.0, "still_inside": 0.0,
         "t_alarm": 10.0}
        for m in ("perfect", "tabpfn") for p in ("nearest", "risk", "ccrp") for s in (0, 1)
    ]  # fmt: skip
    (tmp_path / "data" / "crowd_ladder.json").write_text(json.dumps({"rows": rows}))
    fig = tmp_path / "figures" / "a.png"
    fig.parent.mkdir()
    fig.write_bytes(b"png")
    text = _readme(tmp_path, "M1", [fig]).read_text()
    assert "# M1" in text and "`figures/a.png`" in text
    assert "| B | tabpfn | ccrp | 1.00 | 5.0 | 90 |" in text
    assert "uv run world --replay" in text and "--keep --out" in text
    assert "offline" in text  # the stand-in caveat travels with the numbers
