# Stage 18: exploratory ID learning acceleration

Freeze configuration, scientific sources, data/order/target identities, complete
starting states and historical cost evidence BEFORE new fitting. This user-
authorized development study measures ID learning curves directly. It is not
the proposed stage17 error audit or a cache/budget ablation, and changes no
earlier frozen protocol, negative endpoint or confirmation gate.

## Paired training and new ID sample

Retain scratch/grid_nca/graph_nca/prefix_grid/prefix_graph from their original
stage13 step0 complete language-trained states, including learned tied heads,
with fresh AdamW states. Imported language stages share architecture, head
initialization, 2000 updates, 8192000 text tokens, sequence order and schedule.
Their PPT initializations differ by treatment; scratch has no PPT. No extra text
is assigned to scratch. This tests task adaptation AFTER that identical language
stage, not language-stage convergence itself, GNN execution speed or new PPT.

Keep the same six-layer/192-width/four-head transformer, GPT-2 vocabulary,
context1024, answer+EOS format/masks, full-model fp32 CPU AdamW (.9,.999), eps1e-8,
weight decay .01, clipping1, batch8, two threads, deterministic operations,
600 updates and warmup30/cosine .0003 to .00003. Reuse original numerical helpers.
All arms get identical actual examples/order/LR/update and token counts.
There is no early stopping or endpoint selection, even after target attainment.

Generate a fresh ID-only resample with seed2026103118 using unchanged v12 graph/
prompt/label helpers: 512 training and128 development examples, both BFS/Dijkstra
tasks, sizes8/10, cyclic-ER family, native view. Same distribution as before,
new ID sampling/order. Set assessment count to ZERO; no OOD rows are generated.
Keep original within-study train/dev WL-key exclusions including cuts. This is
one historic PPT/language initialization cohort with a new development sample,
not fresh independent rule/data/topology cohorts or untouched confirmation.
No claim of exclusion from every earlier development dataset is made; new starts
have not undergone any previous algorithmic adaptation. No old adapted state is
continued. Freeze IDs, labels, full order, actual token/padding counters and
example exposures; labels are cross-checked with Floyd-Warshall before fitting.

## Frozen semantic target and all-arm measurements

Fit ONE source-aware marginal predictor on TRAINING labels only: for each
task/size, choose the most frequent non-source distance (ties choose smallest
integer), output zero at the known source and that modal distance elsewhere.
It uses source identity and training marginals, not adjacency/path computation.
Freeze its predictions and per-cell development scores before any new neural fit.

Primary target at an observed ID checkpoint requires ALL FOUR task/size cells
to have unconditional NON-SOURCE node accuracy at least0.10 above that frozen
baseline's same-cell accuracy, pooled valid-vector rate>=.95 and coverage1.
Invalid vectors count every node wrong. No source-zero shortcut can satisfy the
non-source target. If any target exceeds1, fail before fitting, never reduce it.
This is a new prospective EXPLORATORY semantic target, not a relabeling or
relaxation of stages13/15/17's80% exact-vector calibration. Always retain exact
vectors, whole/non-source node scores, source/hop/signed/first-error counters,
per-cell validity and teacher-forced digit/other NLL/accuracy. Teacher scores
condition on gold earlier answers; digit tokens may include punctuation.

At0/100/200/300/400/500/600 save complete model/optimizer checkpoints and generate
all128 held-out ID answers using the original prompt-only unconstrained decoder,
128-token cap, no hints/head reset/truncation/logit restriction. Record actual
input/supervised/padded tokens, examples, training time, evaluation time and
adaptation elapsed time. Save every sampled ID/loss/LR/gradient. Measurement
overhead is included in elapsed but separated from training timers. No training
probe is substituted for held-out learning curves. Seven checkpoints/five arms:
35 checkpoints and4480 new predictions, plus separate replay of the same.

## First crossing, censoring and contrasts

First target attainment is the FIRST observed checkpoint satisfying the entire
target. Require the next scheduled observation also to satisfy it when present;
report that sustained-crossing sensitivity separately, never replace primary.
No interpolation, selected best checkpoint or post-hoc margin change. Unreached
is right-censored at600 with actual budget, not recorded as attainment at600.
Crossing at0 is reported as pre-existing ID performance with zero adaptation,
not infinite speedup. Keep final attainment and any later regression separately.

Report updates, example presentations, input/target/padded tokens, train seconds
and adaptation elapsed seconds to crossing. Report ratios ONLY if both arms
reach with positive denominators; otherwise null with explicit censoring/zero
reasons. Primary graph contrasts are graph versus BOTH grid and prefix_graph;
retain scratch and prefix_grid in all curves/tables. No p-values, cohort CI,
equivalence claim or seed selection: one inherited cohort is descriptive.
Show complete exact/node/NLL curves even if every arm is censored. No success
or confirmation is inferred just from a lower partial loss or fixed endpoint.

## PPT, language and generation cost accounting

Before fitting, bind original v4 cohort59 language/PPT summary files and its
data manifest. Extract PPT update/token/train/stage seconds; text/generation/
paired-preparation seconds; identical language token/update/train/stage seconds;
original cumulative stage cost and its documented exclusions. Scratch PPT is
zero. Charge full paired preparation to every synthetic arm, as original v4
did; this is conservative shared preparation, not isolated graph-generation cost.

Report adaptation-only crossing and separately total training tokens/updates
(PPT+language+adaptation) and accounted observed pipeline seconds (original
GPU cumulative seconds plus new CPU adaptation elapsed). Do NOT call summed
heterogeneous-hardware seconds controlled end-to-end throughput or cost recovery.
Keep GPU/CPU components, missing/unmeasured costs and exclusions explicit:
cloud queue, process/package startup, original calibration/initial snapshot and
final export/checkpoint verification/archive work are not all in original totals.
New plan/data preparation, checkpoint IO, replay and packaging wall costs stay
in the append-only logger, separately from crossing. Do not silently amortize PPT
over seeds or treat imported work as newly executed/zero-cost initialization.
Token sums are stage-count ledgers, not interchangeable FLOP or monetary units.

## Verification and boundaries

Retain the first attempt; no silent restart, extra updates or target adjustment.
Separate replay reloads all35 saved checkpoints and reproduces all4480 generated
answers, teacher scores, semantic target/crossing/censoring, source/input/order/
optimizer/token identities and imported costs. Same-code replay is not independent
neural inference or retraining. Complement it with independent decimal parsing,
source/non-source counters, fixed-baseline oracle checks and public-only score/
cost reconstruction. Private weights/optimizer/order arrays/raw caches remain
excluded from distribution; hashes, predictions, scores and cost evidence remain.

New scientific fitting:3000 CPU task-adaptation updates. No new PPT/language,
GPU/remote/paid job, official language test, OOD predictions, email, push or external
publication. User authorizes this exploratory test despite the failed original
synthetic-readiness bridge; no eligible-confirmation bit changes. Original
categorical/random-rule readiness and fresh-cohort/identical-stage/OOD selection
requirements remain necessary for the separate confirmation protocol. No outcome
here automatically opens OOD or a fresh full language suite.
