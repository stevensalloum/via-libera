"""Hazard prediction (owner: Ahmad).

Input: readings, the ``SourceEstimate`` and forecast times. Output: ``HazardMap`` with
P(toxic) and P(fire) per cell and time. Baseline: ``ForwardSimPredictor``.
Goal: a TabPFN predictor with calibrated probabilities that beats it (see docs/brief.md).
"""

from tabpfn_spatial.hazard.baseline import ForwardSimPredictor, brier, hazard_from_fields

__all__ = ["ForwardSimPredictor", "brier", "hazard_from_fields"]
