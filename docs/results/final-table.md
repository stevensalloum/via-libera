# Final 3D test: the numbers behind the pitch

Made with `uv run evac-final` (9 plants, 1 leak each, cloud TabPFN, 2026-11-22).
Policies: nearest exit (today), risk-aware, CCRP replanned live, CCRP planned once.
Models: the team's TabPFN models, or perfect knowledge of the gas (upper bound).

```
split   models   policy    s in gas  % exposed  95% out s
known   tabpfn   nearest       0.23        3.7         41
known   tabpfn   risk          0.14        2.3         48
known   tabpfn   ccrp          0.14        2.3         42
known   tabpfn   once          0.14        2.3         42
known   perfect  nearest       0.23        3.7         41
known   perfect  risk          0.15        2.4         49
known   perfect  ccrp          0.14        2.3         42
known   perfect  once          0.14        2.3         40
unseen  tabpfn   nearest       0.84        6.2         43
unseen  tabpfn   risk          0.24        3.0         47
unseen  tabpfn   ccrp          0.09        2.4         40
unseen  tabpfn   once          0.09        2.4         40
unseen  perfect  nearest       0.84        6.2         43
unseen  perfect  risk          0.01        0.1         46
unseen  perfect  ccrp          0.01        0.1         45
unseen  perfect  once          0.01        0.1         42
known: locator 1.1 m (first 0.6 m), height 0.95 m, hazard Brier 0.100, recall 0.22, false alarms 0.000, alarm 5 s
unseen: locator 0.6 m (first 0.1 m), height 0.58 m, hazard Brier 0.053, recall 0.26, false alarms 0.000, alarm 5 s
```

## k-NN comparison (same plants and leaks, `--backend baseline`)

_pending_

## Hard case (`--stress`: low leaks near people, 2x rate, updates every 5 s)

_pending: Rayan's run_
