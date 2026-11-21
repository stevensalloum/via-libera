import numpy as np
import pytest

from tabpfn_spatial.hazard.benchmark import binary_metrics


def test_binary_metrics_handle_ties_and_calibration() -> None:
    result = binary_metrics(np.array([0.2, 0.2, 0.8, 0.8]), np.array([0, 1, 0, 1]))
    assert result["brier"] == pytest.approx(0.34)
    assert result["ece"] == pytest.approx(0.3)
    assert result["auc"] == pytest.approx(0.5)
