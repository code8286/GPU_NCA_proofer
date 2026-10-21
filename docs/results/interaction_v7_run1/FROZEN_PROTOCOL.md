# Stage 7: one spatial coupling, matched rule comparison

Declared 20 October 2026 before model fitting or assessment. Preserve v1-v6
sources, configurations, protocols and evidence. No language command exists.

## Treatment and fixed settings

Reference: x'(r,c) = 1 XOR x(r,c), the stage 6 binary complement rule.
Treatment: x'(r,c) = 1 XOR x(r,c) XOR x(r,c-1) XOR x(r,c+1), with periodic
horizontal boundaries on the same 4x4 lattice. This adds one neighbor-parity
coupling term. Both neighbors are distinct, and one can lie outside a 2x2 patch.
All cells update simultaneously. Both rules are involutions on this lattice;
the spatial map is bijective and avoids finite-time collapse to a constant.
Period-two dynamics are intentionally retained. This is one known interacting
rule, not arbitrary NCA rule learning, graph rewiring or language transfer.

Keep the v6 alphabet (binary; 18 tokens), grid, eight frames, context 47, 2x2
serializer, six-layer/192-wide/four-head backbone, tied embeddings, FP32 CPU,
two threads, deterministic algorithms, batch 2/accumulation 2, Adam, LR 1e-4,
warmup/cosine schedule, 1000 updates, final-checkpoint endpoint, eval cadence,
data seed 20261022, three model seeds 131/149/167 and split sizes 512/64/128.
Numerical training, masks and scores import the unchanged v4/v5 implementations.
Each of the twelve models gets 188,000 input tokens / 1000 updates. Total:
12,000 updates / 2,256,000 input tokens. No budget extension or checkpoint
selection from development losses is allowed.

## Matched data and phase leakage

Changing a rule changes phase-equivalent initial states. Use a NEW shared split
for BOTH rules and rerun the complement reference. Do not compare the historical
v6 endpoint causally to this treatment. Initial-state settings/sizes stay fixed,
but the sampler explicitly changes to avoid leakage under either rule.

For each initial grid x, jointly exclude {x, complement(x), spatial(x),
complement(spatial(x))} across all splits. The two maps commute, so these are
their complete joint orbits. There are 16,512 joint classes on 65,536 states.
Draw a seeded permutation of all states and accept its first member of each new
joint class until 704 initial states are obtained. Split in fixed 512/64/128 order.
No assessment likelihood is inspected when selecting classes. Both treatments
start from the EXACT same initial grids. Prefix-shuffled controls preserve each
initial frame, all markers and supervised patch counts. Their random permutation
streams match across treatments. Save all actual initial IDs, joint-class IDs,
arrays and hashes before training.

Within each seed, all four arms (reference/treatment x coherent/shuffled) start
from the same saved full weights and use identical minibatch index orders and
budgets. All seeds share this one split: they are initialization replicates,
not independent rule/data/topology cohorts. Log development loss every 100
steps, but do not compute assessment scores until ALL twelve arms finish.

## Frozen assessment

Score all 128 held-out initial conditions, 28 patch targets each (3584/cell),
using the first-frame mask and full trained context. Training also predicts
subsequent markers as before. For each rule/seed, cross-evaluate the initial,
coherent-trained and shuffle-trained checkpoints on both coherent and matched
shuffled trajectories: 36 cells total. Retain per-initial-condition loss,
accuracy and counts. Fit causal frequency priors/concentrations only on each
rule's training data, using the unchanged grid and first 256 training rows.

Unchanged readiness requires BOTH advantages >=0.01 nats/patch for all three
coherent-trained checkpoints under EACH rule:

1. Causal frequency NLL minus coherent-model/coherent-data NLL.
2. Same checkpoint's shuffled-data NLL minus coherent-data NLL.

Validate scoring for each rule with a known full-previous-frame oracle. Decode
the actually observed prior frame, apply its rule, and assign probability .99
to the correct next patch token and .01/15 to each other patch token. This is a
known-rule measurement control, not a trained model. It must pass both margins;
the exchangeable frequency-only negative must fail. A full-frame oracle replaces
the v6 own-patch transition-table measurement control because the spatial rule
cannot be represented by that table. This evaluation change applies to both
new reference and treatment and does not change transformer training.

Separately predeclare a COUPLING QUALIFICATION: fit an add-one 16x16 transition
table p(next patch | previous SAME patch) on training data only. This comparator
lacks adjacent-patch information. Each spatial coherent-trained model must beat
its held-out coherent NLL by >=0.01 nats to qualify evidence beyond this local
table. Report this independently of the two unchanged readiness checks. Bridge
completion requires readiness plus this qualification; failure must not be hidden
by renaming one as the other. This finite trained table is a diagnostic, not the
Bayes-optimal bound over all local-information predictors or a mechanism proof.
The reference table may beat the reference transformer; that is not a failure of
the spatial qualification. Report shuffle-trained readiness and ordering-gap
differences descriptively, without adding gates after seeing results.

## Evidence and decision

Freeze this protocol, config, historical/new source hashes, all data identities
and actual starting snapshots before model fitting. Preserve optimizer states
and checkpoint/summary/result receipts. Refuse overwritten runs or changed inputs.
Reload three initial and twelve final checkpoints and replay all 36 cells and
decisions with the same scorer. This is checkpoint replay, not independent loss
code or a training rerun. Do not treat trajectory rows as independent models or
compute a language-transfer p-value from these shared-data seeds.

If either readiness or spatial qualification fails, retain the result and inspect
the limiting comparison before declaring a new one-factor change. No retuning,
longer budget, seed exclusion or relaxed margin is allowed within this run.
If both pass, declare a separate bridge toward the original task, changing one
representation/rule-complexity factor at a time. A fixed known interacting rule
does not establish held-out-rule generalization, GraphNCA acceleration or cost
recovery. Original random-rule checkpoints retain failed stage 5 readiness.
Language training stays disabled regardless; no official language test is read.
