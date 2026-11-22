"""Gas lookup table and the height zone around a leak.

The gas in a plant is known (it is what the pipes carry), so it is not a feature to learn:
it selects the physics. ``SG`` is the density relative to air.

Height zone (from the 3D study, ``docs/research/3d-two-workflows.md`` section 6): gas density
decides where the gas goes only while the mixture's density excess is above ~1 % of air
(Britter & McQuaid 1988, passive limit), i.e. up to the volume fraction
c_p = 0.01 / |SG - 1|. With near-field plume dilution, centreline c = q0 / (pi U sigma^2) and
sigma = i x (turbulence intensity i), that happens at x_p = sqrt(q0 / (pi U c_p)) / i.
Inside x_p people get a height instruction; beyond it the gas follows the air.
"""

from dataclasses import dataclass

import numpy as np

INTENSITY = 0.10  # turbulence intensity sigma / U (Red:Vapor measured 0.09)
# Our simulator's leak rate is in model units (0.5-2). For the height zone, one model unit
# is read as this many litres of pure gas per second. ASSUMPTION, kept in one place.
LITRES_PER_RATE_UNIT = 5.0


@dataclass(frozen=True)
class Gas:
    """One gas of the lookup table."""

    name: str
    sg: float  # density relative to air

    @property
    def advice(self) -> str:
        """Instruction inside the height zone."""
        if abs(self.sg - 1.0) < 0.1:
            return "none"  # about as dense as air: height does not help
        return "stay low" if self.sg < 1.0 else "stay high"


GASES = {g.name: g for g in (
    Gas("hydrogen", 0.07), Gas("methane", 0.55), Gas("ammonia", 0.59), Gas("CO", 0.97),
    Gas("H2S", 1.19), Gas("propane", 1.52), Gas("CO2", 1.52), Gas("butane", 2.0),
    Gas("chlorine", 2.45),
)}  # fmt: skip


def passive_distance(sg: float, q0: float, u: float, i: float = INTENSITY) -> float:
    """Distance (m) from the leak after which density no longer decides height.

    ``q0``: pure-gas release (m^3/s); ``u``: air speed (m/s).
    """
    if abs(sg - 1) < 1e-9 or q0 <= 0:
        return 0.0
    cp = min(0.01 / abs(sg - 1), 1.0)
    return float(np.sqrt(q0 / (np.pi * max(u, 0.05) * cp)) / i)


def height_zone(gas: str, rate_units: float, wind_speed: float) -> dict:
    """Radius (m) and instruction of the height zone for a leak of ``rate_units``."""
    g = GASES[gas]
    q0 = rate_units * LITRES_PER_RATE_UNIT / 1000.0
    r = passive_distance(g.sg, q0, wind_speed)
    return {"gas": g.name, "sg": g.sg, "radius_m": r, "advice": g.advice,
            "litres_per_s": q0 * 1000.0}  # fmt: skip
