# 3D study: real -> simulated or simulated -> real? Every dataset, both tasks

Steven, 2026-11-22. Lab notebook with every fact and decision:
[`3d-study-plan.md`](3d-study-plan.md). Figures: `docs/figures/3d-study/`. Every number
below is printed by `uv run python -m tabpfn_spatial.study3d.summary` from
`outputs/study3d/*.json`.

## 0. The answer in five lines

1. **Workflow B (simulation -> real) is the way to build our models; workflow A (real ->
   simulation) is the way to check them.** A cannot build them: real datasets have too few
   leaks and are too unlike a factory. A real-trained model and the calibrated simulator
   agree on new sites (AUC 0.89-0.92), so A is a good validation tool.
2. **Simple calibrated physics is enough**: a plume or a 3D solve with a mass-consistent
   wind, with the plume spread calibrated once on real data (measured local turbulence
   explains only 1/8 to 1/4 of it). CFD (OpenFOAM) and GADEN did not improve accuracy.
3. **Hazard:** B reproduces the physics on real data (AUC 0.85-0.89 on the 1:50 site, 0.98
   in the empty tunnel). In an office whose ventilation is unknown, nothing works (0.56).
4. **Source:** B is the best locator (0.22 m on a 0.45 m detector grid, better than
   physics inversion) and survives an unseen airflow when the candidate positions are
   known (our pipe fittings).
5. **Height matters only near large leaks:** within 0.1-7 m of a 1 L/s leak, up to 10-70 m
   for 10-100 L/s in still air. A 2D map plus a height zone around the leak is the
   physically right design; full 3D everywhere is not needed.

## 1. What we tested

- **Workflow A (real -> sim):** TabPFN's context is real measurements; it predicts a new,
  simulated site. Truth there = the simulator. It asks: do a model built on real data and
  the physics agree on a site neither has seen?
- **Workflow B (sim -> real):** the context is simulated; it predicts real measurements.
  It asks: can we build the factory model from simulation and trust it on real gas?
- **Task H (hazard, Ahmad):** concentration at a point (x, y, z).
- **Task S (source, Rayan):** where the leak is, from fixed detectors.
- **Always held out:** a whole setup (site) or experiment the context never saw.
- **References:** "real -> real" (what real data alone gives on an unseen site), "physics
  alone" (the simulator, no TabPFN), "hybrid" (real + simulated context with a flag).
- **Backends:** the final numbers use cloud TabPFN-3.5; the offline k-NN stand-in ran first
  to debug (its numbers are in the notebook only).

## 2. The data (what the sensors are, what a row is)

Details and figures: notebook section "What each dataset measures";
`data_redvapor.png`, `data_orebro.png`, `data_uci.png`, `data_srgdm.png`.

| dataset | what it is | sensors | used for | kept on disk |
|---|---|---|---|---|
| **Red:Vapor** | 1:50 industrial site in a wind tunnel (4 setups, 8 runs) | PID reference + MOX + **3D anemometer** on a gantry, 12 heights | H and S, main 3D test | 16 MB (from 2.2 GB) |
| **Örebro 3DSEN** | real office with HVAC, 10 runs, sources at 0-2.2 m | 27 calibrated MOX on a 3 x 3 x 3 grid, simultaneous | H, indoor | 37 MB |
| **UCI 251** | empty 2.5 m wind tunnel, 10 gases, 3 fans | 72 MOX (9 boards x 8) at 6 distances | H, 1D check | 14 MB (from 8.4 GB) |
| **SRGDM** | 2D plumes simulated with airflow measured in a real room (30 sources x 4 airflows) + a real 6 x 5 grid | gas maps 0..1 | S, unseen airflow | 114 MB |

Data decisions worth knowing:
- **Red:Vapor background.** The closed tunnel fills with fog (as large as the plume
  signal) and the raster is measured over 2 h, so each voxel gets the background at its own
  time (10th percentile of readings within +-7.5 min). Checked: correlation 0.88-1.00 with
  the independent upstream sensor.
- **Turbulence measured, not guessed:** per-voxel wind fluctuations from the 10 Hz series
  (sigma_u ~ 0.13 m/s, 9 % intensity at 1.4 m/s).
- **UCI:** only the centre board sees the plume (narrower than the 13 cm board spacing). The
  stored value *rises* in gas although documented as a resistance; we use relative change.
- **SRGDM real part:** no documented source positions, so no real test for task S.

## 3. The physics ladder (simple -> near-real)

| level | what it adds | parameters from |
|---|---|---|
| L1 | Gaussian plume, Taylor spread, ground reflection | measured wind and turbulence at the release; Taylor (1921); time scale of Hanna (1982) |
| L2 | 3D finite volumes, buildings block wind and gas | STL geometry voxelised at 4 cm; measured profiles |
| L3 | mass-consistent wind around buildings (no wakes) | Sherman (1978), as in QUIC-URB |
| L4a | **CFD wind** (OpenFOAM RANS k-epsilon, 150-160k cells per site) + K = nu_t / Sc_t | inlet fitted to the measured profile (Richards & Hoxey 1993); Sc_t = 0.7 (Tominaga & Stathopoulos 2007) |
| L4b | **GADEN** filaments in the CFD wind | gamma = 2K and noise = sqrt(2k/3) from the CFD |

All levels pass conservation tests (plume flux = release; duct outflow = release; L3 wind
divergence-free and deflected around a block; adjoint = forward solves).

### Results against real data (figures `ladder.png`, `profiles.png`)

Means over experiments. "Calibrated" = one turbulence factor (1, 4 or 8) chosen per held-out
setup on the *other* setups. Scores are scale-free: AUC of finding the real hottest 10 % of
points (0.5 = chance), correlation of the downwind vertical profile, and model / real
crosswind spread (1 = right).

| dataset | turbulence | level | AUC hot 10 % | profile r | spread ratio |
|---|---|---|---|---|---|
| Red:Vapor | measured | L1 | 0.87 | 0.72 | 0.18 |
| Red:Vapor | measured | L2 | 0.84 | 0.78 | 0.38 |
| Red:Vapor | measured | L3 | 0.85 | 0.81 | 0.41 |
| Red:Vapor | measured | L4 | 0.87 | 0.71 | 0.49 |
| Red:Vapor | measured | L4G | 0.74 | 0.77 | 0.77 |
| Red:Vapor | calibrated | L1 | 0.88 | 0.87 | 0.89 |
| Red:Vapor | calibrated | L2 | 0.85 | 0.83 | 1.09 |
| Red:Vapor | calibrated | L3 | 0.85 | 0.85 | 1.18 |
| Red:Vapor | calibrated | L4 | 0.84 | 0.81 | 0.59 |
| Red:Vapor | calibrated | L4G | 0.69 | 0.70 | 1.09 |
| UCI | assumed 10 % | L1 | 0.99 | - | 0.41 |
| UCI | assumed 10 % | L2 | 0.98 | - | 1.46 |
| UCI | assumed 10 % | L3 | 0.98 | - | 1.46 |
| Örebro | measured | L1 | 0.55 | 0.38 | 0.50 |
| Örebro | measured | L2 | 0.58 | 0.58 | 0.93 |
| Örebro | measured | L3 | 0.56 | 0.29 | 0.91 |

What this says:

1. **Every level finds where the gas is on the wind-tunnel sites** (AUC 0.84-0.99), and
   none does in the office (Örebro 0.55-0.58: the HVAC inlet and outlet positions are not
   published, so the room's flow cannot be simulated).
2. **The measured turbulence is far too small.** With it, every plume is 2-6x too narrow
   (spread 0.18-0.49). Each voxel's turbulence comes from a 5 s record, which cannot see
   the slow meandering of the plume. One factor of 8 (found leave-one-setup-out, the same
   for every setup) fixes the spread for L1-L3. **Lesson: a simulator must be calibrated
   on plume spread against real data; local turbulence measurements are not enough.**
3. **More physics did not buy accuracy.** Calibrated, the simple plume (L1) is as good as
   CFD (L4) and better than GADEN (L4G). Buildings help with height on the full sites
   (L2/L3 profile 0.81-0.98 on setups C, D), but the averaged scores are close.
4. **CFD (L4) alone** finds the hot spots well (AUC 0.87 uncalibrated) but under-spreads
   even calibrated (0.59): steady RANS has no meandering, a known limit (Tominaga &
   Stathopoulos 2007).
5. **GADEN (L4G) in a steady CFD wind is a deterministic plume**: 10x more filaments and
   3x longer averaging change nothing (checked on exp 4). Its realism (meander,
   intermittency) needs a *sequence* of wind fields from unsteady CFD, which is how its
   authors use it. Its vertical buoyancy is a dimensionless heuristic (Stokes velocity
   without a particle size) and it lists ethanol vapour at specific gravity 1.04 instead of
   1.59: do not use it to decide crouching.

## 4. Hazard task (H)

Figures: `verdict.png` (where each workflow works), `transfer_hazard.png` (per experiment).
Cloud TabPFN, context up to 8,000 rows, each site held out in turn. "Spread ratio" > 1
means TabPFN's prediction is smoother than the real plume.

**Red:Vapor, simulated side = calibrated L3:**

| workflow | rho | AUC hot 10 % | profile r | spread ratio | n |
|---|---|---|---|---|---|
| real->real | 0.75 | 0.91 | 0.92 | 2.74 | 8 |
| A real->sim | 0.46 | 0.89 | 0.91 | 2.25 | 8 |
| B other-sites | 0.50 | 0.88 | 0.83 | 1.41 | 8 |
| B same-site | 0.42 | 0.85 | 0.85 | 1.09 | 8 |
| hybrid | 0.53 | 0.88 | 0.91 | 1.55 | 8 |
| physics | 0.42 | 0.85 | 0.85 | 1.18 | 8 |

**Red:Vapor, simulated side = calibrated L1 (the simplest level):**

| workflow | rho | AUC hot 10 % | profile r | spread ratio | n |
|---|---|---|---|---|---|
| real->real | 0.75 | 0.91 | 0.92 | 2.74 | 8 |
| A real->sim | 0.42 | 0.92 | 0.94 | 2.91 | 8 |
| B other-sites | 0.49 | 0.89 | 0.87 | 0.90 | 8 |
| B same-site | 0.45 | 0.87 | 0.87 | 0.94 | 8 |
| hybrid | 0.52 | 0.89 | 0.90 | 1.35 | 8 |
| physics | 0.45 | 0.88 | 0.87 | 0.89 | 8 |

**UCI (empty tunnel; held out: one fan speed):**

| workflow | rho | AUC hot 10 % | profile r | spread ratio | n |
|---|---|---|---|---|---|
| real->real | 0.77 | 0.98 | - | 0.88 | 3 |
| A real->sim | 0.93 | 0.98 | - | 0.60 | 3 |
| B same-site | 0.73 | 0.98 | - | 1.36 | 3 |
| B other-sites | 0.73 | 0.98 | - | 1.34 | 3 |
| hybrid | 0.78 | 0.99 | - | 0.79 | 3 |
| physics | 0.72 | 0.98 | - | 1.46 | 3 |

**Örebro (office; held out: one experiment):**

| workflow | rho | AUC hot 10 % | profile r | spread ratio | n |
|---|---|---|---|---|---|
| real->real | 0.35 | 0.71 | 0.66 | 1.00 | 10 |
| A real->sim | 0.40 | 0.59 | -0.01 | 1.14 | 10 |
| B same-site | 0.10 | 0.57 | 0.35 | 0.80 | 10 |
| B other-sites | 0.07 | 0.56 | 0.57 | 0.79 | 10 |
| hybrid | 0.32 | 0.68 | 0.62 | 1.06 | 10 |
| physics | 0.09 | 0.56 | 0.29 | 0.91 | 10 |

**Which physics makes the best simulated context?** Offline (k-NN stand-in), Red:Vapor,
mean over the 8 held-out experiments, AUC / profile r:

| context level | B other sites | B same site | physics alone |
|---|---|---|---|
| L1 plume | 0.88 / 0.87 | 0.88 / 0.87 | 0.88 / 0.87 |
| L2 finite volumes | 0.86 / 0.83 | 0.85 / 0.85 | 0.85 / 0.83 |
| L3 mass-consistent | 0.86 / 0.84 | 0.85 / 0.86 | 0.85 / 0.85 |
| L4a CFD | 0.84 / 0.81 | 0.85 / 0.80 | 0.84 / 0.81 |
| L4b GADEN | 0.67 / 0.48 | 0.71 / 0.62 | 0.69 / 0.70 |

What this says:

1. **B works, and is as good as its physics.** TabPFN fitted on simulated rows reproduces
   the simulator's skill on real data (B same-site = physics alone, within 0.02). Context
   from *other* simulated sites is as good: the model learns plume structure that
   transfers across sites.
2. **The best B context is the simplest calibrated physics**; CFD and GADEN do not help
   (previous section: they need unsteady winds to add meander).
3. **Real -> real is best on Red:Vapor** (rank correlation 0.75 against 0.42-0.52), but only
   because all four setups share the tunnel, the tank and the wind range: an "unseen site"
   here is a close cousin. Our factory will not be that close to any public dataset, so
   this number is an upper bound that no workflow could reach on the factory.
4. **A agrees with the physics** (AUC 0.89-0.92, profile 0.91-0.94 on Red:Vapor; rank 0.93
   on UCI): a model built only on real measurements and the calibrated simulator predict the
   same hot spots and heights on a site neither saw. That makes A a **validation tool**, not
   a way to build the factory model.
5. **Hybrid does not beat real alone** here: mixing in simulated rows halves the real
   context and pulls predictions towards the simulator.
6. **In the office nothing transfers** (B and physics at chance, 0.56-0.57); real data of
   the same room gives some signal (0.71). Indoors, the ventilation layout must be known.

## 5. Source task (S)

Figure: `transfer_source.png`. 50 fixed detectors (5 x 5 at 0.10 and 0.42 m height,
0.45 m apart). Simulated leaks: 600 per experiment at equipment heights, read from the
**adjoint** (source-receptor sensitivity, Keats et al. 2007) of calibrated L3 with each
experiment's measured wind: one backward solve per detector gives every possible leak.
Error = horizontal distance to the true leak (1:50 model metres; x 50 for full scale).

**Red:Vapor, cloud TabPFN (targets: leak x and y):**

| workflow | median error_m | max | n |
|---|---|---|---|
| real->real | 0.47 | 0.52 | 8 |
| real->real new position | 0.47 | 0.52 | 8 |
| B same-site | 0.22 | 0.51 | 8 |
| B other-sites | 0.22 | 0.34 | 8 |
| strongest detector | 0.56 | 0.78 | 8 |
| physics inversion | 0.34 | 0.63 | 8 |
| A real->sim | 0.99 | 1.07 | 8 |

- **B is the best locator: 0.22 m median**, better than the detector spacing (0.45 m),
  better than inverting the physics directly (0.34 m) and than "strongest detector"
  (0.56 m). With the offline k-NN stand-in B was 0.75-0.85 m: TabPFN makes the difference.
- **A fails (0.99 m, about chance on a 2.4 m site)**: real data hold 4 leak positions; a
  locator cannot be learned from 4 examples. "Real -> real" for setup B looks perfect
  (0.04 m) only because setups A and B share the tank position (a lookup, not a
  localisation); for new positions it is 0.47 m.

**SRGDM, does a locator survive an unseen airflow?** (simulated plumes in 4 airflows
measured in a real room; held out: one airflow; error in grid cells)

| workflow | median median_cells | max | n |
|---|---|---|---|
| B sim->sim, unseen airflow | 0.02 | 0.25 | 4 |
| strongest sensor | 1.41 | 1.41 | 4 |

- **Yes, for a known set of candidate positions** (the same 30 cells under every
  airflow): TabPFN places the source within 0.02 cells (median) under an airflow it never
  saw. The strongest sensor scores 1.4 cells by construction (each source sits 1.4 cells
  from a sensor). This is Rayan's situation: the candidates are the pipe fittings.

## 6. Height: when does it matter?

Figure: `heights.png`. Method: Britter & McQuaid (1988). Gas density decides height only
while the mixture's density excess is above ~1 % of air; after that the gas follows the
air. With plume dilution (10 % turbulence intensity, close to the 9 % measured in
Red:Vapor), the distance from the leak where that happens:

| leak | still air (0.1-0.3 m/s) | ventilated (1-3 m/s) |
|---|---|---|
| 0.1 L/s (a small fitting) | 0.1-2 m | 0.1-0.7 m |
| 1 L/s | 0.6-7 m | 0.2-2 m |
| 10 L/s | 2-21 m | 0.6-7 m |
| 100 L/s (a broken line) | 6-68 m | 2-21 m |

(range over the gases of our lookup table; CO, nearly as dense as air, is lowest; chlorine,
butane and hydrogen highest.)

The real data agree: Red:Vapor's near-neutral fog stays at release height; in Örebro
only the strongest source (exp 5, with a bubbler) was highest at the floor, all others
followed the HVAC air upwards.

**Consequence for the planner:** a crouch / stay-high instruction is only meaningful near
a large leak in calm air. Everywhere else a height-mixed hazard is physically right. So:
keep the 2D hazard map + a "height zone" around the leak whose radius comes from this
table (gas, leak size from Rayan, air speed); inside it, the gas's density gives the
instruction: lighter than air (hydrogen, methane, ammonia) rises, so stay low; heavier
(propane, CO2, H2S, butane, chlorine) sinks, so stay high and avoid pits and stairs down.

## 7. Which workflow, and how to proceed

**Recommendation: workflow B, with simple calibrated physics, validated with real data.**

For each part of our system:

- **Ahmad (hazard):**
  - Build the TabPFN context from simulation of *our* factory geometry. L3 is enough:
    seconds per leak and wind; L1 even for open areas.
  - **Calibrate the spread** with real data first: one turbulence factor, as done here
    (x8 on turbulence measured over 5 s records).
  - Use workflow A on a public dataset as the check that the simulator and real data
    agree, not as the training source.
- **Rayan (source):**
  - Train on simulated leaks at the **known candidate fittings**, generated with the
    adjoint: one solve per detector gives all leaks, for each wind / ventilation state.
  - This beat physics inversion and detectors here, and SRGDM shows it survives a change of
    airflow when the candidates are fixed.
  - Do not try to learn localisation from real datasets (A fails: too few leaks).
- **Steven (evacuation):**
  - Keep the **2D hazard map** and add a **height zone** around the leak (radius from the
    Britter & McQuaid table: gas, Rayan's leak size, air speed).
  - Only inside it give height instructions (stay low for light gases, stay high for heavy
    ones).
  - Indoors, the ventilation decides where gas goes (Örebro): rooms need their HVAC
    layout. Door states need a solve each, which is feasible for a few states with L3 (CFD
    only where it matters).
- **What not to do:**
  - Expect a model trained on Red:Vapor, UCI or Örebro to transfer to our factory.
  - Use GADEN with a steady wind (it adds nothing; it needs unsteady CFD).
  - Trust any indoor simulation without the ventilation geometry.
  - Use GADEN's buoyancy for crouch decisions.

## 8. Feasibility: cost and time

Measured on Steven's laptop (M-series, 4 cores used):

| step | size | time |
|---|---|---|
| L1 plume | any number of points | < 0.1 s |
| L2 / L3 steady solve | 124k cells (4 cm, 2.4 x 2.4 x 1.4 m) | 10-15 s |
| L3 adjoint (source task) | 50 detectors, every leak cell at once | ~1 min per wind |
| L4a CFD (OpenFOAM, Docker) | 150-160k cells, 1,200 iterations | 20-25 min per geometry (wind speeds rescale) |
| L4b GADEN | 60 s of release, 100 filaments/s | < 1 s |
| TabPFN cloud | up to 10k context x 14k test rows | 10,000 tokens per call, 2-30 s |

Scaled to the 60 x 40 x 9 m factory at 0.5 m cells (~86k cells): L3 is seconds per leak
and wind; the adjoint gives every candidate leak for a fixed detector layout in one go;
one CFD per ventilation / door state is minutes to an hour. Tokens used for this study:
see the progress log in the notebook (well under the 2M agreed).

## 9. Room for improvement

1. **Measure turbulence on the slow scales.** The biggest error in every simulator is the
   plume spread: 5 s dwells miss the meandering. With longer records (or the 22 Red:Vapor
   fly-through runs, not used yet) the calibration factor could be derived, not fitted.
2. **CFD inlet from the measured profile** (OpenFOAM `fixedProfile` with the measured U(z)
   and k from the measured sigmas) instead of a fitted log law: the tunnel profile is not a
   developed log law (setup B: near uniform below 0.6 m).
3. **Unsteady CFD (LES or URANS)** to get the meander that RANS lacks; costly (hours).
4. **Rooms:** Örebro cannot be simulated without the HVAC inlet/outlet geometry. A room
   dataset with documented ventilation (or our own measurement) is needed before trusting
   any simulator indoors.
5. **More real leak positions:** task S on real data has 4 positions (Red:Vapor) and 6
   (Örebro). Any real-only locator is a lookup. The VGR dataset (GADEN in 30 houses,
   48.6 GB, not downloaded) or our own tests would add positions.
6. **Gas physics:** all real data here are near-neutral (fog, ethanol, low ppm). Dense and
   buoyant behaviour is covered only by the Britter & McQuaid criterion, not by a measured
   dataset; GADEN's buoyancy is a non-dimensional heuristic and should not be used to
   decide crouching.
7. **Time:** everything here is time-averaged (steady). The factory hazard at +30 / +60 s
   needs the transient version (same solvers with a time step; UCI and Örebro have
   time series to check arrival times).

## 10. Reproduce

```bash
cd ~/dev/tabpfn-spatial-live
uv run python -m tabpfn_spatial.study3d.overview            # dataset figures
uv run python -m tabpfn_spatial.study3d.ladder              # L1-L3, measured turbulence
uv run python -m tabpfn_spatial.study3d.ladder --k-factor 4  # and 8: calibration candidates
uv run python -m tabpfn_spatial.study3d.ladder --l4         # CFD + GADEN (needs Docker, gaden driver)
uv run python -m tabpfn_spatial.study3d.transfer --backend cloud --level L3 --k cal --max-context 8000
uv run python -m tabpfn_spatial.study3d.transfer_small --dataset uci --backend cloud
uv run python -m tabpfn_spatial.study3d.source --backend cloud
uv run python -m tabpfn_spatial.study3d.source_srgdm --backend cloud
uv run python -m tabpfn_spatial.study3d.heights
uv run python -m tabpfn_spatial.study3d.report && uv run python -m tabpfn_spatial.study3d.summary
```
