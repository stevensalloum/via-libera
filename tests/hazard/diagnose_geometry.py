"""Offline counterexample: the same features for doors in different positions.

Run with ``uv run python tests/hazard/diagnose_geometry.py``.
"""

from dataclasses import replace

import numpy as np

from tabpfn_spatial.hazard.features import cell_features
from tabpfn_spatial.sim import Factory, Scenario, simulate

walls_near = np.zeros((5, 7), dtype=bool)
walls_near[:, 3] = True
walls_near[0, 3] = False
walls_far = walls_near.copy()
walls_far[0, 3] = True
walls_far[4, 3] = False
f_near = Factory(14.0, 10.0, 2.0, walls_near, (), np.empty((0, 2)), np.empty((0, 2)))
f_far = replace(f_near, walls=walls_far)
source = np.array([3.0, 3.0])
target = np.array([[11.0, 3.0]])
wind = np.array([0.8, 0.0])
scenario = Scenario(source, 1.5, wind)
times = np.array([30.0, 60.0, 120.0, 180.0])
print("features for target at each time (near door / far door):")
for time in times:
    near = cell_features(f_near, source, wind, 1.5, float(time), target)
    far = cell_features(f_far, source, wind, 1.5, float(time), target)
    print(f"t={time:.0f}", "identical=", np.array_equal(near, far), "features=", near[0].tolist())
near = simulate(f_near, [scenario], times)[0, :, 1, 5]
far = simulate(f_far, [scenario], times)[0, :, 1, 5]
print("simulated concentration at target, near door:", np.round(near, 8).tolist())
print("simulated concentration at target, far door: ", np.round(far, 8).tolist())
print("cloud credits: 0")
