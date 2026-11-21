"""TabPFN for spatial awareness: gas-leak emergencies in a factory.

Areas (one owner each, see AGENTS.md):

- ``tabpfn_spatial.sim``: factory layout and gas-dispersion simulator.
- ``tabpfn_spatial.source``: leak source detection.
- ``tabpfn_spatial.hazard``: toxic / fire hazard prediction.
- ``tabpfn_spatial.evac``: evacuation planning.

Shared: ``contracts`` (interfaces), ``pipeline`` (end to end), ``core`` (TabPFN backends,
distributions, grids, metrics, budget guard).
"""

__version__ = "0.1.0"
