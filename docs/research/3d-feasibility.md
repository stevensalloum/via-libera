# Feasibility study: going 3D with real data and GADEN

Steven, 2026-11-22. Status: **for discussion, nothing built yet.**

The question: can we use real gas datasets and the GADEN 3D simulator so that the three
parts work in 3D?
- **source:** where is the leak, now in 3D;
- **hazard:** the hazard now, at +30 s and at +60 s, **with height**;
- **evacuation:** which routes are safe at the height of a person, and when to crouch or
  close doors.

The goal is physical realism, so the bar is "does it hold up", not "does it run".

## 1. The candidate real datasets

| | UCI 251 (Vergara 2013) | **Red:Vapor** (Hinsen et al., Sci Data 2026) | Örebro 3DSEN (Burgués et al.) | VGR (Ojeda et al. 2023) |
|---|---|---|---|---|
| Real or simulated | Real | Real | Real | **Simulated** (GADEN + OpenFOAM) |
| Space | Wind tunnel 2.5 x 1.2 x 0.4 m, empty | Wind tunnel, 3 x 2.25 m section, **1:50 scale model of an industrial site** (setups with no buildings, one tall cylinder, many buildings) | 30 m^2 office, HVAC in at floor, out at ceiling | 30 real house models |
| Leak | Gas valve at one end | Fog machine through a hole in a **cylinder that plays a leaking tank**, 0.18 m above ground | "Evaporating chemical source", several positions **and heights** | Source points per house (oven, toilet, sink ...) |
| Gas | 10 gases (methane, ammonia, CO, acetone ...) at 100-1000 ppm | Glycol/water fog with a little ethanol: **an aerosol, not a pure gas** | Not named in the README (likely ethanol, unconfirmed) | Numeric GADEN gas IDs |
| Sensors | 72 MOX, one line of 9 boards at one of 6 distances at a time | Probe with 4 layers (MiCS-5524, MiCS-6814, **PID-AH2 reference**, BME680, **3D ultrasonic anemometer**) moved by a gantry | 27 MOX (Figaro TGS) in a **3 x 3 x 3 grid**, recorded together | Any point (10 cm grid) |
| Height (z) | **None** | **12 heights, 0-88 cm** (8 cm steps), 2D raster cells ~12 cm | **3 heights** (exact values in the CAD image) | 3D grid, 10 cm |
| Wind | 3 fan settings | Profile set, pitot tubes, **3D wind per sample** | Wind speed and direction at the room centre, 1.3 m high | CFD field |
| Time | 100 Hz, 260 s, 18,000 runs | 10 Hz; 8 rasterised runs (~2 h each), 22 "fly-through" runs, 9 purge runs | 2 Hz, 10 experiment logs | ~1000 s per case |
| Size, licence | 7.8 GB, CC BY 4.0 (research use) | **voxel_maps.pkl 9 MB**, experiments.pkl 1.3 GB, CSV 0.8 GB; CC BY 4.0 | 10 CSV logs, ~33 MB total; GPL-3.0 | 48.6 GB; CC BY 4.0 |

"SRGDN": I could not find a dataset with this name. Did you mean VGR (GADEN-simulated)
or Örebro 3DSEN? Tell me where you saw it.

### What each one is good for

- **UCI 251:** sensor physics (MOX response, drift, gas identity, distance along the wind).
  It has no height and no obstacles, so it **cannot teach anything about 3D**. Use: an
  optional check of sensor models only.
- **Red:Vapor:** the closest match to our problem. It is a real industrial-site model with
  buildings and a leaking tank, measured in 3D with a PID reference and 3D wind, and it
  ships the **STL models of the scale site**. Limits:
  - **The plume is fog**, roughly neutrally buoyant: it says nothing about heavy or light
    gases.
  - **It is 1:50 scale:** 88 cm of measured height is ~44 m in reality, and dispersion
    does not scale exactly (Reynolds number).
  - **There are only a few leak positions** (one per setup).
  - **The raster is sequential.** Each cell was measured at a different time, so the
    voxel maps are time-averaged, not snapshots.
- **Örebro 3DSEN:** the only **simultaneous** 3D real sensing (27 sensors at once) and real
  indoor air with HVAC, with the source moved in position and height. Limits: a small
  office, 10 runs, the chemical not stated in the README, only 3 heights.
- **VGR:** large and 3D, with a ground truth everywhere. But it is **simulated**, and it is
  houses, not a factory.

## 2. Training direction: real -> simulated, or simulated -> real?

The proposal was: train on a real dataset, then generalise to unseen GADEN 3D data. My
critical take:

- **Source (Rayan): real -> simulated does not work.** A locator learns from many leaks at
  many positions. Red:Vapor has one leak position per setup (about 4 in total), Örebro a
  handful in one room, and UCI none in 3D. TabPFN would memorise a few leaks, not learn to
  locate. There are also no real datasets with our sensor layout.
- **Hazard (Ahmad): partly feasible.** With **dimensionless features** (downwind and
  crosswind distance over a length scale, height above the source, wind speed,
  obstacles upwind), the Red:Vapor voxel maps give thousands of real (feature ->
  concentration) rows. A model trained on them can be tested on GADEN plumes. That is a
  fair test of "does real plume shape transfer", but it inherits the fog/scale caveats.
- **Recommended: the reverse, plus a calibration step.**
  1. **Calibrate GADEN against Red:Vapor:** GADEN takes STL geometry, and Red:Vapor ships
     STL models of its site. We run GADEN on the same site and wind and compare the
     simulated 3D plume with the measured voxel maps (vertical profiles, plume width,
     arrival times). If they agree, we trust GADEN for our factory; if not, we know by
     how much.
  2. **Train on GADEN** (our factory, many leaks, gases, winds), where every leak has a
     3D ground truth.
  3. **Validate on real data:** Red:Vapor (outdoor-ish, obstacles, 3D wind) and Örebro
     (indoor, HVAC, simultaneous 3D sensors).

  This is the sim-to-real workflow the team agreed on before ("train on simulation,
  validate on real data"), now with a measured realism check for GADEN itself.

## 3. GADEN: what it really does

- **Pipeline:**
  1. CAD geometry as **STL**, in two parts: the walls and objects, and the free-air volume
     for the CFD.
  2. Wind from **OpenFOAM CFD**, steady or transient (tutorial: incompressible,
     k-epsilon, PIMPLE solver).
  3. Gas as **filaments** carried by that wind.
- **Physics:** the paper's abstract (as summarised in the search results, not read
  verbatim) says gravity and buoyancy come from the **molecular weight** of the gas. The
  wind field is computed first and the gas does not change it. So **dense-gas slumping
  that pushes air aside is not modelled**; heavy gas sinks inside a given wind. That is
  fine for small leaks, too weak for large heavy-gas releases.
- **Simulated sensors:** MOX, PID, T-DLAS, anemometer.
- **Practical side:**
  - It is a **ROS 2** package (default branch `humble`, so Ubuntu). On your Mac it would
    need **Docker or a Linux VM**.
  - There is also a standalone `gaden_core` library and a `gaden_gui` (not checked yet).
  - **OpenFOAM is the slow part:** one CFD solve per geometry and wind setting. Our factory
    is one geometry, so a handful of wind settings x many leak positions is affordable.
    The filament runs are cheap.
- **Ready-made data:** VGR (48.6 GB) is GADEN output we could study without installing
  anything, but it is houses.

## 4. The physics we must respect (and test)

- **Gas lookup table** (we know what is in our pipes). Density relative to air:

  | Gas | Relative density |
  |---|---|
  | hydrogen | 0.07 |
  | methane | 0.55 |
  | ammonia | 0.59 |
  | CO | 0.97 |
  | H2S | 1.19 |
  | propane, CO2 | 1.52 |
  | ethanol vapour | 1.59 |
  | butane | 2.0 |
  | chlorine | 2.45 |

  The table also holds LEL/UEL, toxic limits (for example IDLH) and detector alarm
  levels. So the gas is **not a feature to learn**: it selects the physics, and the
  thresholds.
- **Height matters mostly near the source and in still air.** A small leak at room
  temperature mixes with air within metres and then behaves almost like air (passive),
  whatever its density. Strong stratification needs a big release, cold gas, or no
  ventilation. So the first test is to measure, in GADEN, **how far from the source the
  vertical profile still matters** for each gas. If it is only a few metres, 3D hazard
  adds little and we simplify.
- **People:**
  - **Heights:** breathing zone ~1.5-1.7 m standing, ~0.3-0.6 m crouching or crawling.
  - **"Get down" is only right for gases lighter than air** (or hot smoke). For heavy
    gases (propane, CO2, H2S, chlorine) the advice is the opposite: stay high, avoid pits
    and stairs down.
  - **Crawling is slow** (~0.5 m/s against 1.3 m/s walking), so the planner must weigh it.
- **Wind and doors:** room ventilation decides most of the vertical structure. Closing
  doors changes the flow, so in a CFD-based simulator it needs a new solve. That is
  feasible for a few door states, not for every combination.

## 5. How each part scales to 3D

- **Rayan:**
  - **The leak height comes from the pipes**, not from the model: pipes run at known
    heights (overhead rack, drops to machines). The candidate set stays "fittings on the
    pipe network", now with (x, y, z).
  - What changes is the input: sensors at several heights, so the features gain the
    sensor height. That is low risk.
- **Ahmad:**
  - Today he predicts per cell and time. In 3D, predicting every voxel (60 x 40 x 9 at
    1 m = 21,600 cells x 13 times) would be **280k rows per call**, too slow and costly.
  - **Proposal:** predict only where people can be, at **two heights** (standing breathing
    zone 1.6 m, crouched 0.5 m) on the walkable cells. That is ~2 x 450 x 13 = 12k rows,
    close to today's size.
  - Ahmad's idea of relating component height to the predicted gas height then becomes:
    hazard at the heights that matter.
- **Steven:**
  - The planner moves to a **2.5D graph**: each walkable cell gets two layers (standing,
    crouched) with different speed and different danger, plus "change posture" edges with
    a time cost.
  - Instructions become "go to exit 2", "go to exit 2, crouch between here and the door",
    or "stay high".
  - **Door commands:** a room whose forecast danger crosses a threshold within +30/+60 s
    gets a "close" command, applied in the simulation as a geometry change, which needs a
    CFD case per door state.

## 6. Risks and how we would know

| Risk | Test | If it fails |
|---|---|---|
| GADEN plumes do not match reality | GADEN on the Red:Vapor STL site vs the measured voxel maps | Report the gap; keep 2D + vertical correction |
| Height barely matters for small leaks | GADEN: vertical profile vs distance per gas | Drop crouch routing; keep 3D only near the source |
| Too slow to simulate (OpenFOAM) | Time one CFD solve of our factory in Docker | Fewer wind and door cases; precompute |
| Real data too far from our factory | Hazard model trained on GADEN, scored on Red:Vapor and Örebro | Report it honestly as the domain gap |
| Mac cannot run GADEN | Docker image with ROS 2 Humble + OpenFOAM | Linux machine or cloud VM |

## 7. Proposed first steps (each one is a stop-and-check)

1. **Look at the data:** download Red:Vapor's `voxel_maps.pkl` (9 MB) and Örebro's 10 logs
   (~33 MB), and plot the real vertical profiles and the 3D plume.
2. **Get GADEN running** in Docker on its own test scenario. Record the CFD and
   simulation times.
3. **Calibration:** simulate the Red:Vapor site in GADEN and compare with the measured
   maps.
4. **Height study** in GADEN for our gases: does the vertical profile matter at the
   distances where people are?
5. Only then: our factory in 3D, the gas lookup table, two-height hazard, and the 2.5D
   planner.
