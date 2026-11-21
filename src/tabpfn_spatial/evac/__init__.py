"""Emergency evacuation planning (owner: Steven).

Input: people positions, the ``HazardMap`` and the start time. Output: ``EvacPlan`` with
one route per person. Baseline: ``RiskAwarePlanner`` (with a zero hazard map it is the
conventional nearest-exit plan). Goal: a risk-budget planner that uses the hazard
probabilities and lowers true exposure (see docs/brief.md).
"""

from tabpfn_spatial.evac.planner import WALK_SPEED, RiskAwarePlanner, route_exposed

__all__ = ["WALK_SPEED", "RiskAwarePlanner", "route_exposed"]
