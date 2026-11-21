"""Baseline leak locator and the shared localisation score.

Baseline: the leak is near the reading-weighted centroid of the sensors (a Gaussian blob
of 6 m around it). No learning; this is what TabPFN must beat. O(k + ny * nx).
"""

import numpy as np

from tabpfn_spatial.contracts import SensorReadings, SourceEstimate
from tabpfn_spatial.sim import Factory


class WeightedCentroidLocator:
    """Centroid of sensor positions weighted by their latest reading."""

    def __init__(self, sigma: float = 6.0) -> None:
        """``sigma``: blob radius in metres."""
        self.sigma = sigma

    def locate(self, factory: Factory, readings: SensorReadings) -> SourceEstimate:
        """Probability map: Gaussian around the weighted centroid, zero on solid cells."""
        w = readings.values[:, -1] + 1e-12
        centre = (readings.positions * w[:, None]).sum(axis=0) / w.sum()
        d2 = ((factory.centres() - centre) ** 2).sum(axis=-1)
        prob = np.exp(-d2 / (2 * self.sigma**2)) * ~factory.walls
        return SourceEstimate(prob / prob.sum())


def localisation_error(factory: Factory, est: SourceEstimate, true_source: np.ndarray) -> float:
    """Distance in metres from the most likely cell's centre to the true leak."""
    iy, ix = est.top_cells(1)[0]
    return float(np.linalg.norm(factory.centre(int(iy), int(ix)) - np.asarray(true_source)))
