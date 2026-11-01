# Stage 17: intermediate-cache experiment

This is a new local CPU development fit under the
[separately frozen protocol](CACHE_V17_PROTOCOL.md). Only cache size changes;
all five complete original stage 13 step0 language-trained starts and the exact
600-update numerical procedure are retained. It is not a full language/PPT job,
fresh-cohort graph confirmation or a repair of synthetic readiness.

Authoring commands, each through the append-only logger:

```powershell
python scripts/run_logged.py --label cache-v17-freeze-before-first-update -- python -m cache_v17.run freeze
python scripts/run_logged.py --label cache-v17-five-fixed-budget-fits -- python -m cache_v17.run fit
python scripts/run_logged.py --label cache-v17-separate-checkpoint-replay -- python -m cache_v17.run verify
```

These require retained private original checkpoints/caches and the offline
GPT-2 tokenizer. Do not rerun `fit` on the completed first attempt: it refuses
to overwrite the start receipt. Preserve failed runs and all endpoints. No
remote/paid job, GPU launch or automatic OOD/language stage is implemented.

The new64 model is measured on its full64 cache at0/100/300/600 and on full
ID-development128 at600 only. Common16 is derived from the final full-cache
outputs/counters. Imported16 and512 references share exactly common16/dev128;
only imported512 covers full64. Never report the unmeasured16 model on the
additional48 cache cases as zero or compare different evaluation cases silently.

After evidence export, a standalone/public-only score replay is:

```powershell
python scripts/run_logged.py --label cache-v17-public-score-reproduction -- python scripts/reproduce_cache_v17.py
```

It reconstructs data/order, both reference subsets and 1920 new generated
scores/decisions without weights, private data, tokenizer cache or network.
The inherited deterministic generator also creates unused synthetic assessment/
OOD rows in memory; none are scored, used for selection or passed to a model.
Teacher counters are reaggregated only; new neural/teacher logits and optimizer
states require the private checkpoint replay. Replay is not retraining or an
independent neural implementation. Regression fixtures are execution tests.

Private weights/order arrays/raw caches remain excluded from publication.
Keep source/config/protocol/result receipts and all imports unchanged. In the
canonical workspace regenerate notebooks, then directly sync/check Prod after
logged checks. In a standalone publication refresh its inventory after edits.
