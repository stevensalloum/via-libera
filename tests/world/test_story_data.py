import json

from tabpfn_spatial.world.serve import WEB


def test_story_data_has_every_chart_input_and_its_source():
    d = json.loads((WEB / "story-data.json").read_text())
    for key in ("source_2d", "hazard_2d", "planner_2d", "factory_3d", "real", "results"):
        assert key in d and key in d["sources"]
    for m in ("baseline", "tabpfn", "perfect"):
        mean, se = d["planner_2d"][f"{m}/ccrp"]
        assert 0 <= se < mean
