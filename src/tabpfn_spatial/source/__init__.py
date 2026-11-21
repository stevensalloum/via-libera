"""Leak source detection (owner: Rayan).

Input: ``SensorReadings`` up to ``t_obs``. Output: ``SourceEstimate`` (probability map over
the grid, optionally release-rate quantiles). Baseline: ``WeightedCentroidLocator``.
Goal: a TabPFN locator trained on simulated leaks that beats it (see docs/brief.md).
"""

from tabpfn_spatial.source.baseline import WeightedCentroidLocator, localisation_error

__all__ = ["WeightedCentroidLocator", "localisation_error"]
