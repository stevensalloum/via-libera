# Ahmad's hazard-prediction journey

## 2026-11-21: Optional sensor-history features

We tested whether gas readings add information beyond the source estimate,
measured wind and existing spatial features. The original seven-feature model
remains the default. Two optional configurations add observation/horizon time
alone, or time plus nearby sensor values, trends and availability flags. The
simulated event assumes the alarm coincides with leak start at T=0.

![Five-case offline sensor-history comparison](../figures/ahmad/01-sensor-history-pilot.png)

Recreate the figure without model calls:

```bash
uv run python tests/hazard/plot_sensor_history.py
```

The five-leak offline pilot used the default pipe factory, centroid source,
evaluation seed 0, 640 identical context rows seed 1001 and first-30-second
forecasts. Forward simulation vs offline k-NN spatial A / time B / sensor C:
toxic Brier 0.05806 / 0.07700 / 0.07679 / 0.06681; recall at a 0.5 threshold
0.52153 / 0.72847 / 0.47608 / 0.35646; false-positive rate 0.02092 / 0.10207 /
0.10366 / 0.03339. All used zero cloud credits. C improved Brier relative to A
but missed more hazardous cell/time observations at this threshold. None of the
four routes exposed a person in five cases; this is inconclusive. The sample is
too small to select a production configuration.

A proposed separate five-scenario cloud pilot using 8,000 context rows was quoted
at 150,000 credits for 15 calls (forward and centroid baselines zero). It has
not been run and awaits approval. Exact settings, source assumptions, feature
definitions and limitations are in
[`SENSOR_HISTORY.md`](../../src/tabpfn_spatial/hazard/SENSOR_HISTORY.md).
