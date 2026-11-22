"""Hazard prediction (owner: Ahmad).

Input: readings, the ``SourceEstimate`` and forecast times. Output: ``HazardMap`` with
P(toxic) and P(fire) per cell and time. Baseline: ``ForwardSimPredictor``.
Goal: a TabPFN predictor with calibrated probabilities that beats it (see docs/brief.md).
"""

from tabpfn_spatial.hazard.baseline import ForwardSimPredictor, brier, hazard_from_fields
from tabpfn_spatial.hazard.tabpfn_predictor import TabPFNHazardPredictor
from tabpfn_spatial.hazard.volume3d import HazardVolume, MachineBox3D, VolumeGrid
from tabpfn_spatial.hazard.volume3d_predictor import ForwardVolumePredictor, TabPFNVolumePredictor

__all__ = [
    "ForwardSimPredictor", "ForwardVolumePredictor", "HazardVolume", "MachineBox3D",
    "TabPFNHazardPredictor", "TabPFNVolumePredictor", "VolumeGrid", "brier", "hazard_from_fields",
]  # fmt: skip
