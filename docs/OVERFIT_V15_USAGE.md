# Stage 15 small-cache control

Read [the frozen protocol](OVERFIT_V15_PROTOCOL.md). Original results/review:
[overfit_v15_run1](results/overfit_v15_run1/REVIEW.md). This is task-adaptation
fitting from original language endpoints, not another PPT/language run.

With retained private stage 13/14/v4 evidence and the offline GPT-2 cache:

```powershell
python scripts/run_logged.py --label NEW-freeze -- python -m overfit_v15.run freeze --out runs/NEW_RUN
python scripts/run_logged.py --label NEW-fit -- python -m overfit_v15.run fit --out runs/NEW_RUN
python scripts/run_logged.py --label NEW-replay -- python -m overfit_v15.run verify --out runs/NEW_RUN
```

Use a NEW directory. Freeze refuses existing folders and fit refuses a second
attempt. Preserve failures/checkpoints before separately declaring any recovery.
No resumable or extra-training command is provided. Every arm starts from its
verified old step-0 checkpoint with a fresh optimizer and finishes 600 updates.
Progress appears in models/ARM/progress.json after cache evaluations at
0,100,300,600. The logger captures output until completion. Full-study completion
needs root results/receipt plus separate verification passed, 20 checkpoints and
960 new predictions replayed. One completed arm is not whole-study completion.

Original-run export/audit helpers are bound to run1 and its logged chronology:

```powershell
python scripts/run_logged.py --label overfit-v15-export-verified-evidence -- python scripts/report_overfit_v15.py
python scripts/run_logged.py --label overfit-v15-evidence-accounting-audit -- python scripts/check_overfit_v15_review.py
```

A standalone publication can regenerate data/order, reaggregate old references
and reproduce all generated-answer scores without private weights or network:

```powershell
python scripts/run_logged.py --label public-overfit-reproduction -- python scripts/reproduce_overfit_v15.py
```

This verifies public source/data identities, old reference subsets, 960 new free
answer scores and fixed decisions. It does not rerun training or new neural/
teacher-forced measurements; these need private checkpoints and environment.
Cached raw arrays, weights and optimizer states remain excluded. Recorded length
counts/budgets are frozen; the private audit also recomputes them with the exact
GPT-2 codec. Provenance checks may rehash prior assessment bytes, while only the
selected cache and ID development enter the new fit/evaluation.

The cache-fit criterion has four examples per cell, so 80% requires 4/4. It is a
memorization/fitting control. Even passing it leaves original synthetic readiness,
the categorical bridge and OOD gated. Changing cache size couples diversity,
which graphs are included and repetition. Fixed 600 updates/4800 presentations
do not imply equal tokens, FLOPs, time or a graph-learning speedup.
