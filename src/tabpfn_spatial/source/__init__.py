"""Leak source detection (owner: Rayan).

Input: ``SensorReadings`` up to ``t_obs``. Output: ``SourceEstimate`` (probability map over
the grid, optionally release-rate quantiles and, on 3D plants, leak-height quantiles).
Baseline: ``WeightedCentroidLocator``. TabPFN version: ``TabPFNLocator`` (x, y, size),
trained on simulated leaks (see docs/brief.md); ``HeightLocator`` adds the height on a 3D
plant (``sim.plant3d``).
"""

from tabpfn_spatial.source.baseline import WeightedCentroidLocator, localisation_error
from tabpfn_spatial.source.height_locator import HeightLocator
from tabpfn_spatial.source.tabpfn_locator import TabPFNLocator

__all__ = ["HeightLocator", "TabPFNLocator", "WeightedCentroidLocator", "localisation_error"]
