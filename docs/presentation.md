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

## The pitch, screen by screen (1 min product + 2 min technical)

| Time | Say | Show |
|---|---|---|
| 0:00 | A gas leak, 300 people, one minute. | Story page, top (hero). |
| 0:10 | Today: green signs to the nearest exit, even into the cloud. | Scroll to **2 · Today**: the shortest-path table (0.50 s → 0.09 s, −83 %). |
| 0:25 | Via Libera: every person their own way out, live. | **3 · Live map**: Play; person 204 (selected first): red 24 s in gas, green 2 s. |
| 0:45 | Where it goes: robots as detectors, phones and signs as displays. | **8 · Next**. |
| 1:00 | How: three TabPFN models, no training. | **4 · How**: the three steps vs the usual method. |
| 1:30 | In 3D: light gas rises, heavy gas sinks; detectors climb the pipes. | **Enter the factory**: Play, click a person, **Person view**. |
| 2:15 | Why TabPFN: 3.5x closer to the leak than k-NN, no training, physics in minutes not hours. | Back to the story: **6 · Why TabPFN**. |
| 2:40 | Results on 10 plants it never saw: −90 % time in gas. Next: wider hazard map, faster updates. | **7 · Results**, **8 · Next**. |

Be honest if asked: the hazard model is precise (no false alarms) but too narrow (26 % of
harmful cells at P >= 0.5); a model update takes about 33 s on the largest hall (asked every
15 s, the planner uses the latest finished one); one leak per plant.

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
