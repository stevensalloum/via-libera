# 3D study: real -> simulated or simulated -> real? Every dataset, both tasks

Steven, 2026-11-22. Lab notebook with every fact and decision:
[`3d-study-plan.md`](3d-study-plan.md). Figures: `docs/figures/3d-study/`. Every number
below is printed by `uv run python -m tabpfn_spatial.study3d.summary` from
`outputs/study3d/*.json`.

## 0. The answer in five lines

<!-- VERDICT -->

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

<!-- LADDER -->

## 4. Hazard task (H)

<!-- HAZARD -->

## 5. Source task (S)

<!-- SOURCE -->

## 6. Height: when does it matter?

<!-- HEIGHT -->

## 7. Which workflow, and how to proceed

<!-- PROCEED -->

## 8. Feasibility: cost and time

<!-- COST -->

## 9. Room for improvement

<!-- IMPROVE -->

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
