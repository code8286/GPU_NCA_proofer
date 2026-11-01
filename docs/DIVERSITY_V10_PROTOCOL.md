# Stage 10: training initial-condition diversity at fixed exposure

Declared 21 October 2026 before new fitting. This is development on the inspected
stage 7/8/9 held-out data, not new confirmation. Preserve all earlier outcomes.
No language stage is part of this study.

## Single treatment and fixed controls

Compare 512 versus 4096 available training initial conditions. The 4096 cache
contains the exact original 512 coherent trajectories and exact matched shuffles
as its first rows. Append 3584 distinct joint-phase classes, using the first
eligible representatives from a permutation of all 65536 states with expansion
seed 20261021. Exclude all original train/development/assessment classes and ALL
original neighbor-flipped assessment classes when choosing additions. Thus the
original 122 eligible interventions cannot become leaked training examples.
Keep the old intervention mask and every original held-out array unchanged.
Keep one representative per joint orbit under complement and spatial involutions.
No complexity selection or outcome-dependent filtering is permitted.

Generate new trajectories with the unchanged x' = 1 XOR x XOR left XOR right
periodic spatial rule. Retain binary 4x4 grids, eight frames, 2x2 patches, 18-token
alphabet, context 47 and the full six-layer/192-wide/four-head backbone. Generate
the entire nested prefix shuffle with the original data_seed+1 and verify the
first 512 shuffled rows are exact. Conditioning, markers and per-trajectory
supervised-prefix patch counts remain matched.

For seeds 131,149,167, retain standard loss (factor 1) and stage 8's normalized
sixfold complete-first-transition loss (factor 6), each on coherent and matched
prefix-shuffled data. Import the twelve verified final 512-condition controls
from stages 7/8. Train twelve NEW 4096-condition models from the exact saved
initial weights, never from a fitted endpoint. Report imported versus new work.

Keep Adam/reset state, LR/warmup/cosine schedule, FP32 CPU/two threads, batch 2,
accumulation 2 and 1000 updates / 188000 input tokens / 168000 supervised targets
per model. New training totals 12000 updates / 2256000 input tokens. The 24-model
comparison accounts separately for twelve historical budgets. No increased token
budget, topology, serializer or weighting-factor change is permitted.

Sample uniformly with replacement. Use the original minibatch RNG seed+101 and
4000 row draws. All four new arms within a seed share exact indices and initial
weights. Changing the cache bound changes sampled row identities: do NOT claim
the same sampled rows across cache sizes. Freeze expected order digests and
distinct-row counts for each seed/cache size. Verify the 512 digests against old
summaries and the 4096 digests/counts against actual training. Availability and
actual distinct exposure are reported separately. More diversity necessarily
changes example repetition at the fixed token budget.

## Assessment and fixed decisions

Use only final update 1000. Log original unweighted development loss every 100
updates, without checkpoint selection. Wait for all twelve new models before
assessment. Score the three initial states plus twelve imported and twelve new
final models (27 conditions). For each, retain full-trajectory coherent and
shuffled assessment scores (54 cells), first-patch original assessment and the
same 122 eligible neighbor interventions, and split-fit diagnostics below.
Imported coherent first-patch/intervention/full scores must reproduce stages
8/9 exactly. Standard 512 full scores reproduce stage 7 exactly.

Fit the causal trajectory-frequency predictor separately on each cache with the
unchanged train-only calibration policy. Its prior uses that cache's complete
training prefix counts, and concentration selection uses its first 256 rows.
This is a deterministic baseline recalibration from training data, not retuning
model hyperparameters. Report both priors/concentrations/frequency losses.

For EACH objective stratum, require ALL three coherent seeds to satisfy:

1. 512-cache original first-patch assessment NLL minus 4096-cache NLL >=0.01.
2. 512-cache intervention NLL minus 4096-cache intervention NLL >=0.01.
3. The 4096 coherent model retains both original full-trajectory readiness
   advantages >=0.01 nats (frequency and coherent-versus-shuffled likelihood).

Report standard and weighted strata separately. The study-wide primary requires
BOTH strata to pass. These are fixed all-seed development criteria, not a
significance test. Do not pool away failed seeds, select an objective after
results, change thresholds or extend budgets. Accuracy, control-arm diversity
gains and differences of gains are descriptive. Do not interpret a smaller
training/held-out gap by itself as better prediction.

## Fit probes and exposure caveat

Reuse the stage 9 short-prefix first-patch and teacher-forced four-patch/marker
scorers on four FIXED coherent probes: original train rows 0..511 (anchor), new
train rows 512..1023 (extension), original development 64 and assessment 128.
This yields 108 fit cells. The extension is out of the imported controls' cache,
but eligible in the larger cache. Cached training rows are not necessarily drawn
under sampling with replacement. Bind a per-query drawn/not-drawn mask from the
frozen row orders; retain probe means plus drawn-only and undrawn-only first-
patch metrics/counts where available. Empty subsets yield null, never zero loss.
These are cache probes, not an exhaustive evaluation of all 4096 training rows.
Initial and held-out probes have no training exposure. Do not call every cached
row an observed training example. No new shuffled fit probe or rule family is
introduced; shuffle-trained models remain evaluated on coherent fit probes and
both full assessment versions.

Short/long causal-prefix first-patch NLL must agree within the existing 1e-5
tolerance and have identical correctness. Later first-transition patches
condition on earlier true patches; their mean is not a first-patch or rollout
success measure. Preserve the full-trajectory/first-transition distinctions.

## Evidence and permissible conclusions

Freeze configuration, implementation/protocol, copied inputs, disjoint identities,
initial weights, imported checkpoint/assessment hashes and expected sampling
orders before fitting. Preserve new final parameters and optimizer states,
actual index order and exposure counts, all per-query scores, receipts and costs.
Reload checkpoints and replay every score and decision in a separate invocation
with the same scorers. Partition raw evidence files if needed to keep each public
file below the existing 5 MB boundary; weights/arrays remain private.

A positive result supports this diversity treatment on this fixed known rule
and inspected split; it does not demonstrate GraphNCA-specific acceleration,
unseen-rule generalization, GPU throughput or language cost recovery. A failure
is preserved without relaxing criteria. Original random-rule checkpoints remain
unready. Confirmation needs fresh joint-phase-disjoint data. Any language suite
still needs a separately frozen eligible configuration, fresh paired cohorts and
an identical full language stage. No remote/paid job, official-language-test
access or publication is authorized by this local development study.
