# Stage 9: existing-checkpoint first-transition fit diagnostic

Declared 20 October 2026 before computing new train/development first-transition
scores. The stage 8 assessment outcome is already inspected. This is a post-hoc
development diagnostic, not confirmation or a replacement primary experiment.
Preserve the failed stage 8 result and all original evidence. No fitting occurs.

## Fixed inputs and endpoints

Use the verified stage 8 plan and existing initial, standard/coherent,
standard/prefix-shuffled, weighted/coherent and weighted/prefix-shuffled
checkpoints for seeds 131,149,167: 15 model conditions. Trained endpoints are
the original final 1000-update files. Keep the same spatial rule, vocabulary,
serializer, cached data and jointly phase-disjoint train/development/assessment
split (512/64/128 initial conditions). No augmentation, new shuffling, model
selection, optimizer update, language data or intervention query is introduced.

Score each model on BOTH existing coherent and prefix-shuffled data versions
in EACH split: 90 cells. Scoring a shuffled-trained model on its actual shuffled
training data is required to describe its fit; coherent evaluation alone would
measure a different target distribution. Inputs and checkpoint identities are
bound before scoring. Copy the six cached arrays into the private diagnostic
folder and preserve their origin hashes. Raw arrays/weights stay private.

## Measurements

1. First patch: reuse the unchanged stage 7 scorer. It sees tokens 0..6 only
   and predicts token 7, the first patch after the complete initial frame and
   next start marker. Report mean unweighted NLL, accuracy and per-query values.
   Every coherent assessment cell must reproduce the corresponding stage 8
   first-patch NLL/accuracy and per-query vectors EXACTLY (all 15 conditions).
2. Whole first transition: a causal forward pass on input tokens 0..10 predicts
   absolute positions 6..11. Report unweighted NLL and accuracy for its four
   patches (positions 7..10), marker NLL (positions 6,11), and the six-token NLL.
   Later target patches condition on earlier true patches: this is teacher-forced
   prediction, not rollout accuracy. The first patch from this longer pass must
   agree with the original short-prefix losses within 1e-5 nats per query and
   have identical argmax correctness. Small numerical differences due to matrix
   shape are retained; the short-prefix scorer is authoritative for first-patch
   comparisons. No token after the first transition is supplied.

Per-query vectors remain in compact evidence. Aggregate generalization gaps are
development minus train and assessment minus train for each model/data version.
These are reused, inspected splits; assessment is not a new untouched test.
Frequency/noise/marker gains are not inferred to be spatial-rule execution.

## Frozen descriptive comparison

Compare weighted minus standard first-patch NLL for each paired seed and
training arm on both data versions. Positive deltas mean weighting is worse.
Use a fixed 0.01-nat descriptive margin:

- Train delta >=0.01 and assessment delta >=0.01: `fit_regression_on_train_and_assessment`.
- Train delta <=-0.01 and assessment delta >=0.01: `training_gain_with_assessment_reversal`.
- Otherwise: `mixed_or_below_margin`.

Report every seed; never pool away a discordant seed. The principal diagnostic
slice is coherent-trained models on coherent data. Also report development
deltas, both controls, raw gaps and whole-transition measurements. Categories
describe observed endpoint likelihoods. They do not identify capacity,
gradient conflict, overfitting or a unique causal mechanism, and introduce no
new scientific acceptance threshold. Train and held-out gaps can coexist.

If worsening already occurs on training examples, held-out overfitting alone is
insufficient to explain the stage 8 regression. If training improves but
assessment worsens, a transfer/generalization reversal is observed. Either
outcome still needs a separately frozen treatment before any further fitting.
Inspect whole-transition results before attributing a single-patch finding to
the whole weighted objective.

## Integrity and scope

Freeze this protocol, implementation, parent results/verification, all input
and checkpoint hashes before scoring. Assess once after freezing; reopen all
15 checkpoints and replay all 90 cells in a separate invocation. Retain results,
receipts, source identities and errors. Replay uses the same scorers, not an
independent loss implementation or a training rerun. Check logged freeze-before-
assessment chronology. Keep all original plan/source/checkpoint files unchanged.

No result admits the original random-rule checkpoints or authorizes a language
stage. Any future synthetic intervention needs a new development declaration;
confirmation needs fresh joint-phase-disjoint data, and language transfer needs
fresh paired cohorts plus an identical full language stage. No remote/paid job,
GPU training or official-language-test access is part of this diagnostic.
