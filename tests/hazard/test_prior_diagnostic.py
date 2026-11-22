import runpy
from pathlib import Path

import numpy as np

from tabpfn_spatial.sim import Factory, sample_scenarios

DIAGNOSTIC = runpy.run_path(str(Path(__file__).with_name("prior_diagnostic.py")))


def test_continuation_preserves_the_original_observations():
    previous = DIAGNOSTIC["PREVIOUS"]
    factory = Factory.default()
    case = previous["make_cases"](factory, 1, previous["TEST_SEED"])[0]
    scenario = sample_scenarios(factory, 1, np.random.default_rng(previous["TEST_SEED"]))[0]
    extended = DIAGNOSTIC["continue_readings"](factory, scenario, case, 120.0, seed=7)
    n = len(case["obs"].times)
    np.testing.assert_array_equal(extended.times[:n], case["obs"].times)
    np.testing.assert_array_equal(extended.values[:, :n], case["obs"].values)
    np.testing.assert_array_equal(extended.wind, case["obs"].wind)
    assert extended.t_obs == 120.0
