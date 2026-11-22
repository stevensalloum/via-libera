"""Stress conditions for the simulator, from easy to hard (from Ahmad's hazard study).

Each condition changes how the gas mixes (``diffusivity``), how much the wind direction
wanders (``drift``) and how many sensors are dead (``dropout``) or stuck (``stuck``).
Use them for the truth and the readings; the models keep their own training assumptions,
so this tests how they cope with a plant that behaves differently.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    """One stress setting."""

    name: str
    diffusivity: float  # m^2/s turbulent mixing (default plant: 0.8)
    drift: float  # rad / sqrt(s) random walk of the wind direction
    dropout: float  # probability a sensor is dead (NaN row)
    stuck: float  # probability a sensor repeats its first value


CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in (
        Condition("standard", 0.8, 0.0, 0.0, 0.0),
        Condition("wind_drift", 0.8, 0.05, 0.0, 0.0),
        Condition("sensor_faults", 0.8, 0.0, 0.15, 0.05),
        Condition("slower_mixing", 0.4, 0.0, 0.0, 0.0),
        Condition("faster_mixing", 1.6, 0.0, 0.0, 0.0),
        Condition("combined_stress", 0.4, 0.05, 0.15, 0.05),
    )
}
