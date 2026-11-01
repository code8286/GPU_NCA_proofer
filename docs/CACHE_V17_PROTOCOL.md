# Stage 17: intermediate 64-example cache, fixed 600-update development study

Freeze this protocol, sources/configuration, dataset/order identities, initial
states, references and expected token counts BEFORE fitting. Stage 16 diagnosed
poor held-out prediction even among valid vectors and under gold-prefix digit
prediction. It did not identify a unique cause. This is a new cache-size ablation
on inspected development evidence, not confirmation of GraphNCA acceleration.

## Treatment, starts and controls

Change the cache to 64 examples: 16 base graphs per size (8 and 10), both BFS and
Dijkstra tasks, 16 examples per task/size cell, 32 graphs total. Use the SAME
salted base-ID hash ranking as stages 14/15; common16 is nested in cache64,
which is nested in stage 14's train128 and original train512. No output, loss,
difficulty or label is used for selection. These are inspected development
data, not fresh independent confirmation cohorts.

Retain scratch/grid_nca/graph_nca/prefix_grid/prefix_graph from the SAME complete
stage 13 step-0 states and fresh optimizers, not continued adapted models.
Preserve all learned heads, GPT-2 vocabulary, 6L/192D/4H/context1024 model,
prompt/answer/EOS format, answer-only objective/masks, full-model fp32 CPU AdamW
(.9,.999), eps1e-8, weight decay .01, clipping1, batch8, two threads, deterministic
operations, order seed20261031 and repeated-permutation sampler. Use the same
600 updates, warmup30 to .0003 and cosine to .00003 numerical helpers.
Every arm gets the same cache64 batch order and schedule.

Each selected example appears exactly 75 times, versus 300 in small16 and 9–10
in historical512; all have 4800 presentations. Freeze actual input, answer/EOS
target and padding counters. Cache content/diversity/repetition and lengths are
coupled consequences, not independently identified treatments. Fixed updates
do not imply equal tokens/FLOPs/time. Change no topology, serializer, objective,
training budget, hyperparameter or criterion based on new observations.

## Reference scope and evaluation

Keep both verified historical references at their fixed step600:

- Historical512: extract matching cache64 generations/teacher counters from
  stage 14's train128 evidence; retain full development128.
- Small16: retain stage 15's exact final common16 and development128 outputs/
  teacher counters. No reference refitting or new reference-model inference.

Three-way comparisons use ONLY identical common16 and development128 cases.
Full cache64 fitting compares new64 against historical512 on the SAME 64 cases;
the unmeasured small16 model on the additional 48 cases is explicitly absent,
not treated as zero. Report common16 as a matched training subset, not held-out.
All imported fit/replay costs and 6000 reference updates stay separate from
3000 new scientific CPU task-adaptation updates.

Save new model/optimizer checkpoints and complete-cache64 generations plus
teacher scores at 0,100,300,600. At 600 ONLY, evaluate full ID-development128;
derive common16 scores by exact case subsetting of final cache predictions and
teacher counters (no extra forward). Use unchanged unconstrained prompt-only
greedy decoder, 128-token cap, no hints/logit restrictions/truncation/head reset.
Keep invalid vectors wrong in unconditional semantic rates. Preserve digit/other
teacher scores, node/source/hop/signed/first-error counters and all four cells.
Teacher-forced digit tokens may span punctuation and use gold earlier answers.

## Frozen decisions and claim boundaries

Primary fitting criterion: ALL five arms at fixed update600 must have >=80%
exact in EVERY cache64 task/size cell, >=95% pooled valid vectors and full
coverage. With 16 cases/cell, >=80% requires >=13/16. Original development128
criterion remains separate (>=26/32 exact per cell, >=95% validity, full
coverage). Report both even if one fails; no best-checkpoint selection.
Report first observed pooled80% CACHE crossing at the four recorded steps;
unreached is censored, no interpolation. A transient crossing cannot replace
the fixed final per-cell decision. Do not compare crossing ratios across cache
sizes or arms as a confirmatory learning-speed result.

Descriptive comparisons include all same-case arm endpoints and raw counters,
without a p-value, independent-cohort CI or equivalence claim. All ID-development
measurements remain diagnostic, never select a checkpoint/cache or relax a target.
Regardless of outcome, no original random-rule readiness failure is repaired;
the categorical/random-rule bridge and eligible initializations remain required.
OOD and further full language suites remain gated. No GraphNCA acceleration,
cost recovery or mechanism is established by this single-cohort task-fit study.

## Execution and verification

Only selected cache64/common16 and ID-development128 enter new fitting/scoring.
Private input files are read directly, not regenerated assessment data during
fitting. Inherited provenance verification may rehash old assessment bytes.
For public data reconstruction, the unchanged full deterministic generator can
also recreate unused assessment/OOD rows in memory; none are passed to models,
predicted/scored, selected as outcomes or written as new benchmark evidence.
No official language test, remote/paid job, GPU run, email, push or external
publication occurs. New PPT/language updates are zero.

Preserve the first fit attempt, original inputs/outputs, all 20 checkpoints,
actual sampled IDs, losses/gradients/LR/token counts and timers. No silent restart
or continued fit. In a separate invocation, reload all new checkpoints and
reproduce all 1920 new generations/teacher scores and final/common16 decisions,
initial tensor/optimizer states, full update counts, data/order/source identities
and reference subsets. This is same-code replay, not another fit or independent
neural implementation. Independently parse generated answers and validate
counter/CSV accounting. Keep private weights, arrays and raw caches out of the
distribution; public-only replay regenerates data/order and reaggregates scores,
not neural logits. Tiny CPU regression fixtures are not scientific fits.
