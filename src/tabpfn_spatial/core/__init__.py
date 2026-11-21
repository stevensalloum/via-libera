"""Shared starter for every application (integrator-owned).

- ``dist``: ``QuantileDist``, the predictive distribution every regressor returns.
- ``models``: TabPFN adapters (cloud / local) and numpy baselines behind one interface;
  ``make_regressor`` / ``make_classifier`` pick the backend.
- ``budget``: ``CostGuard`` refuses cloud calls above a token limit.
- ``grid``: ``Grid2D`` turns a map into one batched query and back into an image.
- ``metrics``: CRPS, interval coverage, calibration error.
- ``viz``: mean / uncertainty heatmaps.
- ``example``: end to end, a full map from sparse readings (``uv run spatial-example``).
"""
