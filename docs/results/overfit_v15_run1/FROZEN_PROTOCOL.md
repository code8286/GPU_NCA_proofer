# Stage 15: small-cache overfit positive control

Prospective development study, 31 October 2026. Freeze this protocol, sources,
configs, selected IDs, full batch order, starting states, reference identities
and expected exposure before new fitting. Stage 14 showed weak prediction even
on seen examples; it did not identify a unique cause. This tests reliable fitting
of a small declared set under the SAME numerical adaptation procedure.

## One declared training change

Reduce the training cache from 512 to 16 examples. Select four base graphs per
size (8 and 10) using EXACTLY stage 14's salted base-ID hash ranking, retaining
both BFS/Dijkstra tasks for each. This is a nested subset of its 128-example
training probe and of the original stage 13 training cache. Four examples in
each task/size cell; eight graphs total. No label, loss, graph difficulty or
model output informs selection. These inspected development data are not fresh
confirmation, and this subset is not an exhaustive sample of possible caches.

Use all five arms from the SAME complete pre-adaptation language-trained states
as stage 13: its verified step-0 checkpoints, not continued step-600 weights.
Verify full initial tensor hashes, empty optimizer state and original saved
language model/head identities. Keep 6 layers/192 width/4 heads, GPT-2 vocabulary,
tied learned head, context 1024, output format and answer+EOS encoding unchanged.
Use the exact stage 13 answer-only loss, masks, full-model fp32 CPU AdamW,
betas (.9,.999), eps 1e-8, weight decay .01 on all parameters, gradient clipping
1.0, batch 8 and fresh optimizer. Keep its 600 updates, 30-update warmup to .0003,
cosine endpoint .00003, two threads, deterministic PyTorch, order seed 20261031
and repeated-permutation sampler. Reuse the frozen original numerical helpers.
Every arm gets the SAME actual order over the 16 selected examples.

Thus every example appears exactly 300 times, versus 9–10 in the historical
512-cache procedure. Cache diversity, which examples are included and repetition
are coupled consequences of this one cache-size treatment. Fix 4800 example
presentations/600 updates; DO NOT claim identical token count, FLOPs or time:
graph lengths/target tokenizations can change these. Freeze and report actual
unpadded inputs, supervised answer/EOS targets and padded positions. This is
a fitting positive control, not economical-PPT or language-learning acceleration.

## Reference and measurements

Retain the already verified five 512-cache stage 13 runs as IMPORTED historical
reference, all at their fixed 600-update endpoints. Stage 14 saved reference
generations and teacher-forced per-case counters for its training probe and
all 128 development examples. Extract the EXACT selected 16 examples from those
records and reaggregate by task/size; retain the original complete development
records. Verify hashes and identities. No reference fit or GPU timing is rerun,
and imported reference optimizer updates/costs are not counted as new or zero.

For each new small-cache model, save model/optimizer checkpoints and generate
answers on the entire cache at fixed updates 0,100,300,600, using the SAME
unconstrained prompt-only decoder and 128-token cap. Log exact vectors, validity,
node/source/hop errors, coverage and teacher-forced answer/EOS plus digit/other
token scores. At update 600 ONLY, also generate all original 128 development
answers and teacher-forced scores. Development is diagnostic and never selects
an endpoint, cache or hyperparameter. This endpoint-only development measurement
saves inference cost; it changes no optimizer update or training decision.

A positive control PASSES only if EVERY arm's fixed final cache score has >=80%
exact match in EVERY task/size cell, >=95% pooled valid vectors and 100% coverage.
With four examples per cell, 80% requires 4/4: every selected vector must be
correct. Preserve this discretization. Report first observed pooled 80% cache
crossing at recorded checkpoints, WITHOUT interpolation; unreached is censored.
A transient crossing cannot override a failed final cell. Do not select a best
checkpoint, drop an arm, extend training, loosen a target or replace the cache.

Report new versus reference scores on the SAME 16 examples and full 128-example
development set. Show both cache-fit and held-out competence separately; do not
replace stage 13's failed full-ID criterion with a memorization result. Gold
earlier answer tokens condition teacher-forced scores only. Digit-bearing tokens
can span punctuation; their accuracy is not semantic per-node correctness.
Invalid free outputs remain wrong in unconditional node/source/hop denominators.

## Execution, replay and boundaries

New scientific training: 5 x600 =3000 CPU task-adaptation updates, from fixed
language endpoints. No new synthetic PPT, language stage, official language test,
OOD prediction, graph rewiring, serializer change, remote/paid job or email.
Original random-rule readiness and categorical bridge remain failed/required.
Inherited provenance checks may rehash prior assessment bytes; only the selected
training cache and ID development enter new training/prediction/metrics.

Retain all actual sampled IDs, update/loss/LR/gradient records, token counters,
checkpoint identities, initial states and evaluation outputs. Failures must be
retained; no silent restart or recovery. Tiny execution/regression fixtures are
separate from scientific fits. Model weights, cached arrays and correspondence
remain private. Export compact receipts, predictions, scores, costs and source
identities; public code can regenerate data and rescore without weights.

In a SEPARATE invocation, reopen all 20 new checkpoints and exactly reproduce
every cache and development generation, teacher-forced score, fixed decision,
initial identity, optimizer step and actual exposure. Also reproduce extracted
historical reference metrics from unchanged stage 14 evidence. Same-code replay
is not independently implemented neural inference or a second fit. Log new CPU
update/evaluation timers and full wall time, keeping imported costs separate.

If the control passes, conclude only that this pipeline can fit this small cache
at the fixed budget. Development failure would separate that fitting capability
from reliable unseen-graph prediction. If it fails, inspect retained conditional
and generated fitting errors before another separately frozen treatment. Neither
outcome clears synthetic readiness or justifies a fresh full language/OOD suite.
Any later GraphNCA confirmation still needs eligible initialization, categorical
readiness, fresh paired cohorts, all controls, identical full language/adaptation
stages, ID-only selection, frozen OOD targets/censoring/multiplicity and total costs.
