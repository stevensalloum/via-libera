"""Leak source detection (owner: Rayan).

Input: ``SensorReadings`` up to ``t_obs``. Output: ``SourceEstimate`` (probability map over
the grid, optionally release-rate quantiles). Baseline: ``WeightedCentroidLocator``.
TabPFN version: ``TabPFNLocator``, trained on simulated leaks (see docs/brief.md).
"""

from tabpfn_spatial.source.baseline import WeightedCentroidLocator, localisation_error
from tabpfn_spatial.source.tabpfn_locator import TabPFNLocator

__all__ = ["TabPFNLocator", "WeightedCentroidLocator", "localisation_error"]
