"""Factory and gas-leak simulator (owner: Steven).

``Factory.default()`` is the plant; ``sample_scenarios`` draws leaks; ``simulate`` runs
many leaks at once and returns concentration fields; ``readings`` gives what the
plant's sensors measure. Everything other areas train on comes from here.
"""

from tabpfn_spatial.sim.factory import FITTING_WEIGHT, Factory, Machine
from tabpfn_spatial.sim.plume import Scenario, simulate, wind_at
from tabpfn_spatial.sim.scenarios import readings, sample_people, sample_scenarios, sensor_values

__all__ = [
    "FITTING_WEIGHT",
    "Factory",
    "Machine",
    "Scenario",
    "readings",
    "sample_people",
    "sample_scenarios",
    "sensor_values",
    "simulate",
    "wind_at",
]
