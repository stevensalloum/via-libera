import numpy as np

from tabpfn_spatial.sim import Factory
from tabpfn_spatial.source.height_study import PRIOR_ERROR, WS, run_study


def test_masts_find_the_height_better_than_guessing(tmp_path) -> None:
    res = run_study(
        Factory.default(), "propane", "baseline", n_train=150, n_test=40, cache_dir=tmp_path
    )
    assert set(res) == {"one_height", "masts", "three_heights"}
    masts = res["masts"]
    assert masts.error().shape == (len(WS),)
    assert np.all(masts.error() < PRIOR_ERROR)
    assert np.all((masts.lo <= masts.med) & (masts.med <= masts.hi))
    assert not list(tmp_path.iterdir())  # k-NN is never cached
