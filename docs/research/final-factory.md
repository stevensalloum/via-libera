# The final factory: 3D inputs for the source and hazard models

Steven, 2026-11-22. One dataset for the whole team: `sim/plant3d.py` (geometry) and
`sim/gas3d.py` (time-stepped gas). Every input is explicit; nothing is inferred from 2D.

## Geometry: `Plant3D(factory)`

Works for every plant: `Factory.default()`, `Factory.random(...)`, the shaped templates
(`sim.layouts.template("L-plant" | "U-plant" | "T-plant" | "campus")`) and unseen random shapes
(`sim.layouts.shaped(rng, kind)`).

| attribute | what | shape / unit |
|---|---|---|
| `grid` | 1 m voxels, floor to the 9 m eaves | `study3d.geometry.Grid` |
| `solid` | walls and outside to the roof, machines to their height | (nx, ny, nz) bool; `to_zyx()` for (z, y, x) |
| `machines` | name, kind, hot, `box` = (x0, y0, z0, x1, y1, z1) | list of dicts, metres |
| `pipe_segments` | every pipe as 3D segments: rack at 4.2 m + vertical sections | list of ((x, y, z), (x, y, z)) |
| `vertical_pipes` | drops (rack -> machine inlet) and supply risers (floor -> rack) | `VerticalPipe(xy, z0, z1, kind)` |
| `leak_sites` | rack cells at 4.2 m + every 0.5 m along each vertical section | ((n, 3) m, (n,) weights; fittings x5) |
| `detectors` | the mast rule below | ((k, 3) m, (k,) "mounted" / "mast") |

**Mast rule:**
1. Every 2D detector is mounted at the height for the plant's gas: light gases (methane,
   hydrogen, ammonia) at 4.0 m, near the rack; heavy gases (propane, CO2, H2S, butane,
   chlorine) at 0.45 m; gases about as dense as air at 1.6 m.
2. Every vertical pipe section of at least 1.5 m gets a **mast of 3 detectors** beside it, at
   10 %, 50 % and 90 % of its height (Rayan's study: masts at vertical pipes locate the
   leak height best).

## Air: `Ventilation`

| field | meaning | default |
|---|---|---|
| `speed`, `direction` | horizontal air, m/s and rad (towards) | 0.5, 0 |
| `drift` | random walk of the direction, rad / sqrt(s) (0 = steady) | 0 |
| `diffusivity` | turbulent mixing, m^2/s (same meaning as 2D) | 0.8 |
| `update_s`, `seed` | wind re-solve interval when drifting; reproducibility | 30, 0 |

- `Ventilation.from_scenario(sc)` takes a 2D `Scenario`'s air.
- `Ventilation.from_condition("wind_drift")` takes one of Ahmad's stress conditions
  (`standard`, `wind_drift`, `sensor_faults`, `slower_mixing`, `faster_mixing`,
  `combined_stress`). The sensor-fault fields go to `Run3D.readings(dropout=, stuck=)`.
- The wind is made mass-consistent around walls and machines, and the air leaves through the
  roof vents (the 3D study's L3).

## Gas: `simulate(plant, Leak3D((x, y, z), rate, gas), vent, t_end=300, every=5)`

- Time-stepped L3: the same transport operator as the steady 3D solve, advanced in 1 s steps
  (implicit Euler). Checked: it converges to the steady solve.
- Density near the leak: within the height-zone radius of `sim.gases`, light gas rises and
  heavy gas sinks with the buoyant-plume velocity scale (Briggs); none beyond it.
  Checked: hydrogen rises, chlorine sinks.
- Units: the 2D rate units and thresholds (alarm 0.03, toxic 0.1) keep their meaning.
  `RATE_SCALE` = 19.4 is calibrated on a neutral gas against the 2D simulator.
- About 1 s per 120 s of gas on the default plant.

`Run3D` gives:

| method | for | returns |
|---|---|---|
| `readings(rng, noise, floor, dropout, stuck)` | Rayan | `SensorReadings` with `heights` (k,), every 5 s, the 2D noise model |
| `alarm(readings)` | everyone | first time any **noisy** reading > 0.03 (sensor-triggered) |
| `arrivals()` | Rayan | (k,) first time each detector's clean value > 0.03 |
| `volumes_zyx()` | Ahmad | (T, nz, ny, nx) concentration |
| `breathing_map()` | planner | (T, ny, nx) worst value between 0.5 and 2.0 m on the 2D grid |

Contracts (`contracts.py`, optional fields, nothing breaks):
- `SensorReadings.heights`: (k,) detector heights;
- `SourceEstimate.height`, `height_levels`: quantiles of the leak height.

## Example

```python
import numpy as np
from tabpfn_spatial.sim.layouts import template
from tabpfn_spatial.sim.plant3d import Plant3D
from tabpfn_spatial.sim.gas3d import Leak3D, Ventilation, simulate

plant = Plant3D(template("T-plant"))  # hydrogen
sites, weights = plant.leak_sites  # (n, 3), (n,)
leak = Leak3D(tuple(sites[0]), rate=1.2, gas=plant.f.gas)
run = simulate(plant, leak, Ventilation.from_condition("wind_drift", speed=0.6))
r = run.readings(np.random.default_rng(0))  # positions (k, 2), heights (k,), values (k, T)
t_alarm = run.alarm(r)
volume = run.volumes_zyx()  # (T, z, y, x) for the hazard model
```

## What stays open

- The planner still uses the 2D breathing-height map. The height-zone instruction
  (crouch / stay high) is the next step.
- Indoor HVAC layouts (inlets and outlets per room) are not modelled; the roof-vent
  extraction is the stand-in.
