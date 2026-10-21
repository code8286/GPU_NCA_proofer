# Stage 8: matched first-transition loss weighting

Declared 20 October 2026 before weighted-model fitting. This is development on
the inspected stage 7 data, not fresh confirmation. Preserve v1-v7 evidence.
The sole treatment is objective weighting. No language stage exists.

## Treatment and matched controls

Use the exact stage 7 SPATIAL rule, cached arrays, joint-phase-disjoint split,
data/model seeds, 18-token alphabet, 4x4 grid, eight frames, context 47, serializer,
six-layer/192-wide/four-head model, actual initial checkpoints, Adam state reset,
LR/warmup/cosine schedule, batch 2/accumulation 2, FP32 CPU/two threads,
deterministic algorithms and final 1000-update endpoint. The original initial
frame is masked. Existing labels include subsequent patches AND frame markers.
No new examples, topology, serializer, augmentation or early stopping are added.

For each valid target at absolute position 6..11 (the complete first transition),
set loss weight to 6; set all remaining valid targets to 1. Masked targets have
weight 0. Minimize SUM(weight * cross-entropy) / SUM(weight). There are six first
transition targets and 36 later targets per sequence: the first transition's
loss mass increases from 1/7 to 1/2. Its 4 patch + 2 marker targets retain the
same overall patch/marker ratio (2/3 patches). This normalized reweighting does
not multiply the entire loss by six. Factor 1 reproduces the original objective.

Reuse the six verified stage 7 spatial standard-loss checkpoints (coherent and
prefix-shuffled for seeds 131,149,167) as imported controls. Their source,
configuration, full initial state, batch order, final weights, final-update
selection and all reused scores are verified; do not claim a fresh control
training run. Train six NEW weighted models, coherent and matched shuffled,
from those same initial states and batch indices for 1000 updates / 188,000
input tokens each. New training totals 6000 updates / 1,128,000 input tokens;
the compared twelve-model budget includes the same cost for imported controls.
Neural architecture/token/update budgets match; additional weighting arithmetic
means exact total FLOP identity is not claimed. Report actual new CPU timers,
separately from historical standard-control timers, without a GPU speed claim.

## Frozen endpoints and decision

Log UNWEIGHTED native development NLL every 100 updates and weighted training
loss separately. Do not select a checkpoint or retune the factor/budget from
these curves. Wait for all six new weighted models to finish before assessment.
Use the exact 128 original first-transition queries and the same 122 eligible
neighbor-flip pairs from stage 7, with the same fixed mask. They are inspected
development data. Never exclude a seed or change that mask to improve results.

Re-evaluate initial, standard/coherent, standard/shuffled, weighted/coherent and
weighted/shuffled checkpoints for every seed on BOTH full data versions: 30
full-trajectory cases, each with 3584 patch targets. Score those same 15 model
conditions on original and changed first-transition first-patch queries, with
only the initial full frame and next start marker available. Retain per-query
loss, correctness, paired-both-correct and expected argmax-change behavior.
All original standard-control metrics must replay exactly before comparison.

For EACH of the three coherent-trained seeds require BOTH primary improvements:

1. Standard first-patch NLL - weighted first-patch NLL >= **0.01 nats** on
   all 128 original queries.
2. Standard intervention NLL - weighted intervention NLL >= **0.01 nats**
   on the 122 eligible changed queries.

Also require weighted models to retain both unchanged full-trajectory readiness
advantages (>=0.01 nats versus the train-only causal frequency predictor, and
>=0.01 from coherent versus matched shuffled data). All three seeds must pass;
no pooling away a failed replicate or post-hoc accuracy threshold is allowed.
Report accuracy changes, full-trajectory tradeoffs and shuffled controls even
if NLL criteria pass. The change in weighting gain between coherent-trained
and shuffle-trained models is descriptive, not an added post-outcome gate.

These criteria test optimization of the known-rule first prediction; they do
not establish language acceleration, generalization to unseen random rules or
a unique internal circuit. The rule/model/data stay fixed, so no causal claim
about graph topology or serialization follows. The three seeds share one
inspected data split and are initialization replicates, not independent rule
cohorts. Do not manufacture a language-transfer significance test from query rows.

## Evidence and permitted next work

Freeze config/protocol/source identities, identical copied input hashes, parent
verified results and imported checkpoint hashes before weighted training. Preserve
all new final models/optimizer states and summaries. Reject changed sources,
arrays, plans or references. Reload all imported/new checkpoints and reproduce
all 30 full and 15 first-transition cases with the same scorers. This is replay,
not a new control training run or independently implemented loss function.

If the fixed criterion fails, preserve the result and declare one further
development change rather than extending this run. If it passes, inspect residual
accuracy/error tradeoffs before adding rule diversity. Fresh joint-phase-disjoint
data is required for later confirmation. No outcome authorizes language training;
the original random-rule models retain failed stage 5 readiness. The official
language test remains untouched. No remote or paid job is part of this study.
