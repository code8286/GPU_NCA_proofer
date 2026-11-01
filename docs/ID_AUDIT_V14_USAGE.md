# Stage 14 matched ID error audit

Read [the protocol](ID_AUDIT_V14_PROTOCOL.md). The historical run is
`runs/id_audit_v14_run1`; [its review](results/id_audit_v14_run1/REVIEW.md)
retains all five fixed endpoints and zero new optimizer updates.

With private stage 13/12/v4 checkpoint folders and the local offline GPT-2 cache
present, freeze a NEW output directory before scoring:

```powershell
python scripts/run_logged.py --label NEW-freeze -- python -m id_audit_v14.run freeze --out runs/NEW_RUN
python scripts/run_logged.py --label NEW-audit -- python -m id_audit_v14.run evaluate --out runs/NEW_RUN
python scripts/run_logged.py --label NEW-replay -- python -m id_audit_v14.run verify --out runs/NEW_RUN
```

No fitting command exists. The audit refuses an existing freeze folder or a
second evaluation attempt and verifies source/input/checkpoint identities.
Only ID probes enter new scores; inherited provenance checks may rehash old
assessment bytes. Each completed arm writes metrics plus both prediction files.
Full completion needs root results/receipt AND separate verification with passed
true, five models and 1280 predictions replayed. The logger prints captured
output after the command exits. Preserve any failed attempt and its outputs.

The original-run export/check helpers are bound to run1 and its logged chronology:

```powershell
python scripts/run_logged.py --label id-audit-v14-export-verified-evidence -- python scripts/report_id_audit_v14.py
python scripts/run_logged.py --label id-audit-v14-evidence-accounting-audit -- python scripts/check_id_audit_v14_review.py
```

Publications exclude weights, raw probe caches and private correspondence. From a
standalone publication, verify public data and generated errors with:

```powershell
python scripts/run_logged.py --label public-v14-reproduction -- python scripts/reproduce_id_audit_v14.py
```

This regenerates exact probe bytes, checks labels independently and rescores all
1280 generated answers without private weights/data, tokenizer cache or network.
It verifies free-generation scoring, not neural inference or teacher-forced
metrics. Those require retained checkpoints. Training exposures and prior receipts
are preserved in plan/provenance; the full private evidence audit needs them.

The matched task/size cells have different graph instances and may differ in
degree, token lengths and distances. Report covariates and original exposure
counts. Invalid outputs count wrong in unconditional node/source/hop metrics;
valid-only signed errors have a separate denominator. Teacher-forced digit tokens
can include punctuation and gold earlier answer tokens; their score is not
semantic integer prediction or unconstrained rollout. No outcome clears original
random-rule readiness, the categorical bridge or stage 13's failed criterion.
