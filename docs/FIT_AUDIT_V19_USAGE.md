# Stage19 usage

The [protocol](FIT_AUDIT_V19_PROTOCOL.md) freezes a checkpoint-only diagnostic
before new prediction. All five stage18 endpoints are unchanged; no additional
training, earlier checkpoint selection, OOD run or gate relaxation occurs.

Private execution sequence, always through the append-only logger:

```powershell
python scripts/run_logged.py --label fit-audit-v19-freeze-before-prediction -- python -m fit_audit_v19.run freeze
python scripts/run_logged.py --label fit-audit-v19-five-seen-probe-evaluations -- python -m fit_audit_v19.run evaluate
python scripts/run_logged.py --label fit-audit-v19-separate-checkpoint-replay -- python -m fit_audit_v19.run verify
python scripts/run_logged.py --label fit-audit-v19-export-verified-evidence -- python scripts/report_fit_audit_v19.py
```

These require private source checkpoints/arrays and the offline GPT-2 tokenizer.
Do not delete a failed attempt or silently rerun evaluation; retain all evidence.
The five new seen-probe generations/teacher scores are separately replayed.
The same640 held-out outputs/counters are imported exactly from stage18, not
new inference. Probe selection is balanced/outcome-blind and actual exposure
is checked. The marginal predictor remains fitted on full training512 and is
scored separately on each probe's own cases. Original dev targets stay intact.

After export, public-only reproduction from root or standalone is:

```powershell
python scripts/run_logged.py --label fit-audit-v19-public-reproduction -- python scripts/reproduce_fit_audit_v19.py
```

This reconstructs ID data/probe/order/baselines, independently parses all1280
new/imported predictions and checks all600-step numeric trace records, blocks,
metrics and decisions without weights/private data/tokenizer cache/network.
It reaggregates teacher counters, not neural logits. No assessment/OOD rows
are generated. Imported synthetic-ID optimization traces are explicitly selected
for distribution; weights, arrays, raw caches and corpora remain excluded.

Canonical root files are maintained before notebook regeneration and final direct
Prod sync/check after all logger writes. Standalone edits require inventory
refresh. No paid/remote/GPU job, email, push or external publication is launched.
