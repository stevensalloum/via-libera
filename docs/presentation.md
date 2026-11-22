# Via Libera: presentation guide

## Before you present (5 minutes)

```bash
git pull && uv sync
uv run via-libera            # opens the story page; nothing computed, no tokens
```

- The 3D factory loads Three.js from a CDN: have internet (a phone hotspot works).
- Browser full screen (Ctrl+Cmd+F on a Mac), zoom 100 %.
- Backup if the laptop fails: the same command on Rayan's or Ahmad's laptop (the saved run
  ships in `demo/`), or the gallery: `uv run gallery Final`.

## The video, scene by scene

`uv run via-libera` opens the scenes in this order; **→ / space** next, **←** back, **C** shows
the face-cam frame (bottom right is kept free on every scene), **H** hides the navigation.

| Scene | Say | On screen |
|---|---|---|
| (0) | generated clip, 10 s (edited in) | — |
| 1 · Via Libera | the name: Italian "the way is clear" | logo |
| 2 · Vision | smart factories first; then robots and drones for logistics and maintenance | live 3D factory behind the text |
| 3 · System | detectors → source → hazard → routing → people and robots, every 15 s | flow chart (hover a block) |
| 4 · Source (2D) | sensor history in, source map out; TabPFN vs k-NN vs centroid | error over time, release-rate error |
| 5 · Hazard (2D) | P(toxic) per cell; TabPFN vs forward physics | Brier across mixing, operating points |
| 6 · Routing (2D) | CCRP on length + w·P(harm); TabPFN gives P(harm), not the search | seconds in gas: baseline / TabPFN / perfect vs nearest exit |
| 7 · 3D | same metrics in 3D; leak height too | source error in 3D, exposure in 3D |
| 8 · Real data | only four public datasets; TabPFN still transfers | source error and AUC on real data |
| 9 · Demo | one worker, three plans | live map (Play); "Person view in 3D" for the first-person shot |
| 10 · Results | −83 % time in gas, −90 % unseen, −64 % exposed, no slower | counters |
| 11 · Future | models, guidance, real sites, robots and drones, interfaces | cards |

## Recordings (screen recording: Cmd+Shift+5 on a Mac, "Record selected portion")

Record with the browser in front (a hidden window slows the 3D down).

1. **Story scroll** (20 s): top to the live map, slow scroll.
2. **Live map** (20 s): Play at the live map; click a person near the purple cloud.
3. **Factory overview** (20 s): Enter the factory, tick *cinematic camera*, Play.
4. **Person view** (20 s): click a person near the leak, **Person view**, Play: the green route
   ahead on the floor.
5. **Shortest path vs Via Libera** (15 s): same person, switch the crowd selector between
   "nearest exit" and "CCRP replanned live".
6. **Walk inside** (10 s): Walk inside, W and Q/E to turn.

The viewer's own **Record video** button saves the 3D canvas only (no panels) into the run's
`videos/` folder; a clip saved as `videos/1-overview-via-libera.webm` becomes the story page's
hero background.

## Commands

| Command | What |
|---|---|
| `uv run via-libera` | the presentation (story page + factory, saved cloud run) |
| `uv run via-libera --list` | saved runs in `demo/` |
| `uv run world --replay --rung T --leak 500 --backend cloud` | a new 3D run (about 1M tokens) |
| `uv run world --keep` | reopen the last run |
| `uv run evac-final --backend cloud --leaks 1 --unseen 1 --n-train 100` | the final study (about 1M tokens per plant) |
| `uv run evac-compare` | redraw `docs/figures/final` from the saved runs (free) |
| `uv run gallery Final` | every figure and video in pitch order (`milestones/`) |

## Where things are

- Pitch figures: `docs/figures/final/` (1 exposure, 2 accuracy in the factory, 3 accuracy on
  real data, 4 compute, 5 hard case).
- Numbers: `docs/results/final-table.md`.
- Per person: `docs/journey/{steven,rayan,ahmad}.md`, figures in `docs/figures/<name>/`.
- Logo: `docs/brand/via-libera-logo.webp`.
