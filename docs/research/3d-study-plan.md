# 3D study: plan, data facts and decisions (working notes)

Steven, started 2026-11-22. This is the lab notebook for the two-workflow study. It records
every fact we rely on, where it comes from, and every decision, so results can be traced.

## Goal (from Steven)

Try both workflows, on every dataset, for both model tasks, from simple to near-real
physics. Then report where each works, with figures, explanations and how to proceed.
It may turn out not to work; we must know and show we tried everything.

- **Workflow A (real -> sim):** fit TabPFN on real measurements, test on simulated 3D data
  (our physics ladder and GADEN).
- **Workflow B (sim -> real):** fit TabPFN on simulated data, test on real measurements.
- **Task H (hazard):** concentration (or above-threshold probability) at a point
  (x, y, z, t), given the source, wind and geometry.
- **Task S (source):** leak position from sensor readings.

TabPFN is used in context, not trained, so small real datasets can be enough as context.
That is the reason workflow A is worth testing at all.

## Datasets (facts, with sources)

### Red:Vapor (Hinsen, Wiedemann, Shutin, Lilienthal; Sci Data 2026 13:586; Zenodo 16414473, CC BY 4.0)

- **Facility:** DNW low-speed wind tunnel, 3.0 x 2.25 m section. A 1:50 industrial site on a
  turntable.
- **Frame:** origin at the turntable centre on the floor. +x points upwind, so the **wind
  blows towards -x**. z points up, units metres (paper text).
- **Setups:**
  - A: the source tank only;
  - B: the source plus one tall cylinder;
  - C and D: a full site, the same layout rotated by +22.5 and -22.5 degrees.

  The source is a fog release through a hole in a tank model, **z = 0.18 m**. Its x, y per
  setup are in Fig. 8 and the setup 3D files (supplementary_material.zip, 262 MB), **not
  in the text**.
- **Fog:** 2 L propylene glycol, 1 L triethylene glycol, 1 L water, 20-40 mL ethanol. It is
  an **aerosol**, near neutral buoyancy, not a pure gas.
- **voxel_maps.pkl (9 MB):** 8 sampling experiments. Each is a DataFrame with:
  - the position x, y, z (z from 0.025 to 1.145);
  - the MiCS-6814 (NH3, NO2) and MiCS-5524 readings (voltage, resistance, Rs/R0, ppm);
  - the PID reading (ppm, ethanol- and PG-equivalent, relative);
  - BME680 data and 3D wind (u, v, w).

  Grid: 12 x 12 cm horizontal, 8 cm vertical, up to 20 x 20 x 12 cells. Dwell time 5 s
  per cell (20-30 s for setup B).
- **PID ppm is relative and uncalibrated** (an assumed 55 mV/ppm, response factors
  RF_ethanol = 3, RF_PG = 11). Use it as a **concentration-proportional** signal, not
  absolute ppm. Closed-circuit tunnel: the background builds up (12-42 ppm max per run).
- **Experiments** (overview_table.csv):

  | Setup | Wind 1.3-1.5 m/s | Wind 3.6-3.8 m/s |
  |---|---|---|
  | A | Exp 1 | - |
  | B | Exp 2 | Exp 3 |
  | C | Exp 4, 8 | Exp 5 |
  | D | Exp 6 | Exp 7 |

  Plus 22 fly-through runs and 9 purging runs.

### Örebro 3DSEN (Burgués, Hernández, Lilienthal, Marco 2018/2020; GitHub jburgues/Orebro3DSEN, GPL-3.0)

- **Room:** office, 5.94 m (x) x 4.56 m (y). HVAC supplies air at floor level and extracts it
  at the ceiling; one door, one window that opens.
- **Sensors:** 27 MOX (Figaro TGS), calibrated to ppm, on **9 poles x 3 heights:
  z = 0.45, 1.20, 2.23 m** (wsn_lite.py). The x and y of the poles are given only as grid
  indices 0-2 ("evenly distributed").
  - **Assumption:** pole centres at x = 5.94 (i + 0.5) / 3 and y = 4.56 (j + 0.5) / 3.
- **Sampling:** 2 Hz, about 1.9 h per experiment. The logs also have wind speed and
  direction at 1.3 m.
- **The 10 experiments** (README Table 1):

  | Exp | Source (x, y, z) m | Beaker | Airflow | Window |
  |---|---|---|---|---|
  | 1 | (2.70, 0.50, 0.90) | small | off | closed |
  | 2 | (2.70, 0.50, 0.90) | large | off | closed |
  | 3 | (5.95, 1.60, 0.00) | large | off | closed |
  | 4 | (0.10, 4.50, 2.00) | large | off | open |
  | 5 | (2.70, 2.70, 0.00) | large + bubbler | off | closed |
  | 6 | (2.70, 0.50, 0.90) | small | DC fan | closed |
  | 7 | (2.70, 0.10, 2.20) | small | DC fan | closed |
  | 8 | (2.70, 0.50, 0.90) | small | tower fan I | closed |
  | 9 | (2.70, 0.50, 0.90) | large | tower fan I | closed |
  | 10 | (2.70, 3.15, 1.75) | large | tower fan III | closed |

  So there are 6 distinct source positions at **heights 0 to 2.2 m**.
- **Chemical:** an "evaporating chemical source"; the README does not name it (ethanol
  likely, unconfirmed).

### UCI 251 (Vergara et al. 2013, CC BY 4.0, 7.8 GB)

- **Tunnel:** 2.5 x 1.2 x 0.4 m. 72 MOX on 9 boards, at one of 6 line positions L1-L6 at a
  time.
- **Gases:** 10. Wind: 3 fan settings. Heater: 4-6 V. Sampling: 100 Hz, 260 s, about
  18,000 runs.
- **No height information.** Use: distance-to-source (1D) and concentration vs distance.
- **Processing plan:** keep the gases and settings needed, downsample to 1 Hz, keep a
  compact npz, then delete the zip (25 GB free disk).

### SRGDM (Winkler et al., Sens. Actuators B 419 (2024) 136267; GitHub BAMresearch/SRGDM, MIT)

- **Synthetic part:** 2D gas maps (30 x 25) generated from **measured airflow**. We keep
  the valid split (65 MB) out of 410 MB.
- **Real part:** a 6 x 5 sensor grid, 4 recording days (4 x 13.7 MB), stored as tensors
  [n_sources, n_images, 6, 5].
- **2D only.** Use: horizontal plume shape and the sensor-grid source task.

### VGR (Ojeda et al. 2023, GADEN houses, 48.6 GB)

Not downloaded (disk space). Not needed if GADEN runs locally.

## Physics ladder (simple -> near-real), each level checked against real data

| Level | Model | Parameters from |
|---|---|---|
| L0 | Today's 2D advection-diffusion | Our simulator |
| L1 | 3D Gaussian plume with ground reflection + vertical drift by density | Plume theory; dispersion coefficients; Britter-McQuaid for dense gas |
| L2 | 3D finite-volume advection-diffusion with obstacles, buoyant drift, ventilation | Indoor turbulent diffusivity, air changes per hour from literature |
| L3 | 3D flow around obstacles + L2 dispersion | Potential flow or a small Navier-Stokes solver |
| L4 | GADEN: OpenFOAM CFD wind + filament dispersion with buoyancy (molecular weight) | GADEN papers |

## Decisions so far

- **Red:Vapor is the primary real 3D dataset** for task H: real obstacles, 3D wind,
  12 heights, PID reference.
- **Örebro is the primary real dataset for height and indoor physics**: sources at
  heights 0-2.2 m, HVAC, simultaneous 3D sensing.
- **UCI and SRGDM are 1D and 2D checks.** They cannot test height.
- **Concentrations differ in units across datasets** (relative PID, calibrated MOX ppm,
  raw MOX resistance). All comparisons use **normalised concentration** (per experiment:
  c / c_reference) and **rank-based metrics** (Spearman, AUC of top-decile cells), plus
  log-scale R^2.
- **Features shared across domains** (needed for transfer): downwind / crosswind
  distance and height above the source scaled by a length scale, the height of the point,
  wind speed, and whether obstacles lie between source and point.
- **Disk:** 25 GB free at the start. Big raw files are deleted after conversion.

## Progress log

### 2026-11-22, session 1

- **Red:Vapor leak positions** (from the setup STL models): hole at (0.783, 0.017, 0.18) m
  for A and B; C turned -22.5 degrees gives (0.727, -0.29, 0.18); D turned +22.5 degrees
  gives (0.72, 0.309, 0.18). The tank radius is 0.157 m.
- **First real-data finding, Red:Vapor:** the plume is densest near the release height
  (bands 0.12-0.42 m) and 3-10x weaker above 0.7 m. At 3.6 m/s it stays lower.
- **First real-data finding, Örebro:** with HVAC supplying at the floor and extracting at
  the ceiling, the **middle and top sensors read higher than the bottom ones**, even for
  floor-level sources (exp 3: 0.57 / 1.40 / 1.63 ppm at 0.45 / 1.20 / 2.23 m). Room airflow
  dominates over gas density, so 3D needs real wind fields.
- **Örebro data quality:** some MOX channels go negative (exp 4, -1.7 ppm), a calibration
  drift to handle.
- **Loaders:** `src/tabpfn_spatial/realdata/{redvapor,orebro}.py`, with tests.
- **GADEN core native macOS build** (`~/dev/gaden/gaden_core`):
  - fixed so far: submodules fetched over HTTPS, `third_party/libbsc/version` renamed to
    `version.txt` (it clashed with the C++ `<version>` header), formatter `format()` made
    const for fmt 12, a stray `{}` removed in PlaybackSimulation.cpp, `brew install libomp
    fmt eigen`;
  - **next blocker:** `boost/compute.hpp` not found (try `brew install boost`, or point at
    `third_party/compute/include`).
- **Installed:** colima and docker (not started; images are pulled only after UCI is
  converted and its zip deleted).
- **Downloads:**
  - done: Red:Vapor voxel maps, experiments.pkl, data_records_csv.zip, supplementary (STL);
    Örebro logs and images; SRGDM real 6x5 and the valid split;
  - running: UCI zip (7.8 GB) in the background.
  - **Still to fetch:** SRGDM `30x25_multipleSources/test.pt` (came back empty); torch is
    needed to convert the SRGDM tensors.
