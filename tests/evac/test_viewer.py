from pathlib import Path

import numpy as np

from tabpfn_spatial.evac.viewer import Config, save_gif, save_story
from tabpfn_spatial.sim import Factory


def test_story_and_gif_render(tmp_path: Path) -> None:
    f = Factory.default()
    cfg = Config(seed=1, start=np.array([5.0, 15.0]), t_alarm=10.0, update_s=10.0, t_max=30.0)
    story = save_story(f, cfg, [10.0, 20.0, 25.0, 30.0], tmp_path / "story.png")
    assert story.stat().st_size > 20_000
    gif = save_gif(f, Config(seed=1, t_alarm=5.0, t_max=8.0), tmp_path / "run.gif", every=4)
    assert gif.stat().st_size > 5_000
