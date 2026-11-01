# Stage 16 usage and scope

This is a retained-output CPU audit, not a training job or new neural evaluation.
The [protocol](ERROR_AUDIT_V16_PROTOCOL.md) was frozen before the new tables.
Read the [review](results/error_audit_v16_run1/REVIEW.md) and full result/CSV
counters. Historical data, checkpoints and prior evidence remain unchanged.

From either the research root or a standalone publication:

```powershell
python scripts/run_logged.py --label error-audit-v16-public-replay -- python -m error_audit_v16.run verify
```

This regenerates the selected graph identities and rescores all 1440 IMPORTED
endpoint outputs, with independent generated-vector parsing and hop distances.
It checks input/source/receipt hashes, exact CSVs and null empty-bin handling.
It uses no network, private weights or tokenizer cache. It reaggregates imported
teacher counters; it cannot reproduce their neural logits/NLLs from raw weights.
The imported stage 15 checkpoint replay is not newly executed by this command.
The inherited generator also recreates unused assessment/OOD rows in memory;
see the explicit [protocol disclosure](ERROR_AUDIT_V16_REGENERATION_NOTE.md).
Verification refreshes only the stage 16 verification receipt timestamp; if
running in a standalone publication, refresh its inventory after testing edits.

The original authoring sequence was `freeze`, then `analyze`, then `verify`
using `python -m error_audit_v16.run`, each through the logger. Freeze and analyze
refuse to overwrite the existing plan/results. Preserve all completed evidence;
do not remove it to rerun this development study. The optional derived review
renderer is `python scripts/report_error_audit_v16.py` (also through the logger).

Digit accuracy is token-micro, source/hop correctness uses node denominators,
and node-macro retains per-example weighting. Conditional valid-vector metrics
do not replace unconditional metrics or control which cases a model parses.
The two tasks per graph and one inherited cohort do not provide independent
replicates. No new pass threshold, language suite or OOD eligibility is added.
