# Stage 19: seen-example fitting versus held-out prediction

Prospective checkpoint-only development diagnostic, 1 November 2026. Freeze
configuration, sources, selected IDs, baseline/reference/input/checkpoint hashes
and trace summaries before computing new model outputs. No fitting or selected
earlier checkpoint. Preserve all stage18 targets, costs, censoring and failures.

## Fixed question and probes

Stage18 never reaches its held-out semantic target. Determine whether its fixed
final models meet that criterion on a balanced SEEN training probe, and describe
actual optimization. This distinguishes a seen-probe fitting failure from a
held-out gap after seen-probe competence; neither identifies a unique cause.

Retain ALL five arms at update600 only, from the unchanged verified stage18
checkpoints with full model/optimizer/count receipts. Select128 seen examples:
32 base graphs per size8/10, both BFS/Dijkstra tasks,32 examples per cell. Rank
training base IDs by SHA256('fit-audit-v19-20261101:'+base_id), tie by ID. Select
no case by outcome, loss, difficulty or label. This is a subset of stage18's512
training examples and each selected case must have9 or10 actual presentations.
Both probes have identical task/size composition, not identical graphs or all
covariates. Report edge/source-degree/target-distance distributions and graph
counts. The two tasks per graph are correlated; no independent-sample CI/test.

Use the SAME full development128 from stage18. Import its fixed final generation
and teacher counters exactly, plus its verification receipt. Do not rerun or
label those as new held-out predictions. No assessment/OOD data or official
language test enters this study. No alternative checkpoint or arm is selected.

## Baselines, metrics and predeclared interpretation

Keep the original source-aware marginal predictor fitted to ALL stage18 training
labels. Do not refit it on the selected probe. Score that same frozen predictor
on BOTH probes; freeze its predictions/counters before any new neural scoring.
Held-out baseline/targets must reproduce stage18 exactly. For each probe, apply
the SAME relative criterion: non-source accuracy >= baseline+10pp in EVERY
task/size cell, pooled validity>=95% and coverage1. Separate same-case baseline
rates can differ between probes; do not silently reuse a different distribution's
absolute threshold or treat that as a neural generalization effect.

Report all raw per-cell and pooled semantic/exact/format/whole-node/source/hop/
signed/first-error counters, gold-prefix teacher digit/other NLL/accuracy and
source-aware marginal advantages. Invalid outputs count wrong on every node.
Teacher tokens can span punctuation; token and semantic accuracy are different.
Also retain the original80% per-cell exact-vector criterion as a separately
labeled calibration measure (26/32 per cell here), not a replacement target.

Per-arm descriptive labels, using fixed endpoint scores only:

- Seen-probe semantic target fails: fitting on this seen probe is not established;
  held-out generalization alone cannot explain every failure. This does not prove
  that training longer solves it, or diagnose capacity/representation uniquely.
- Seen passes and development fails: held-out gap after seen-probe competence.
- Both pass: both sampled probes meet this DEVELOPMENT criterion, without
  retrospective stage18 crossing changes, confirmation or automatic OOD access.
- Seen fails but development passes is retained, never dropped or relabeled.

Do not construct new speedup ratios or alter the stage18 primary target. This
diagnostic measures fixed predictions and actual traces, not learning acceleration.

## Actual optimization and exposure

Import all600 update records per arm byte-for-byte. Check sampled IDs, LR, input/
supervised/padded tokens and cumulative counts against the frozen stage18 order,
per-example encoding and budget. Report fixed non-overlapping blocks1–100,
101–200,...,501–600: mean loss, mean/min/max gradient norm, fraction of PRE-CLIP
norms>1, mean LR and token counts; additionally first50/last50 loss means and
unique-example presentations. Never infer a unique optimization mechanism from
these descriptive summaries. Loss mixes digit/format/EOS targets and changing
example/order/exposure composition; it is not an independent held-out metric.
Do not use gradients, losses or a best checkpoint to choose the new seen probe.

## Execution, replay and boundaries

New work: evaluate five fixed checkpoints on128 seen examples each,640 new
prompt-only generations plus teacher scores. Use the original decoder,128-token
cap, batch8 teacher scorer, tokenizer/context/format, CPU2 threads and deterministic
PyTorch. No hints, logit masks, truncation, head reset or model modification.
Copy the640 earlier development outputs as IMPORTED evidence, not new inference.

Separate replay reloads all five endpoints and reproduces all640 new outputs/
teacher scores, optimizer600/count receipts, fixed decisions and trace summaries.
Same-code replay is not independent neural implementation or retraining. Public
reconstruction regenerates only stage18 ID data (assessment count0), derives
probe/baseline/order, independently parses generation semantics and recomputes
all trace/cell/decision counters without weights, private caches or network.
It cannot reproduce teacher logits from counters. Preserve failures/resolution.

New optimizer updates, PPT/language/GPU/remote/paid runs, OOD/official-test
predictions, emails, push and external publication:ZERO. Preserve original
checkpoints/datasets/ZIPs. Distribute compact scores, exact prediction/trace
copies, receipts and source identities; exclude private weights/raw arrays/cache.
Original costs remain inherited, not newly measured or zero. Any later treatment
needs a separately frozen budget/data/target, all controls and explicit cost
accounting. This stage never promotes eligibility or changes the original
categorical/random-rule bridge and separate confirmation/OOD requirements.
