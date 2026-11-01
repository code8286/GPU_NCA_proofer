# Stage 14: matched ID training/development generated-error audit

Prospective diagnostic protocol, 31 October 2026. This follows the inspected
stage 13 failure. It is new development analysis of existing checkpoints and
data, not untouched confirmation. Freeze code, configuration, protocol, selected
IDs, original exposure counts and checkpoint/data hashes before new predictions.
No optimizer, fitting, OOD evaluation or checkpoint selection is performed.

## Fixed endpoints and matched probes

Use ALL five stage 13 final checkpoints at update 600: scratch, grid_nca,
graph_nca, prefix_grid, prefix_graph, historical cohort 59. Retain complete
backbones and inherited GPT-2 tied heads/vocabulary/context. Verify the original
stage 13 plan, result receipt, verification receipt, checkpoint identities,
actual optimizer steps and recorded common language/adaptation procedure.
Do not inspect other endpoints to select one for this audit.

Take all 128 original ID-development examples (64 graph instances), both BFS
and Dijkstra distance-vector tasks, 8 and 10 nodes: 32 examples in EACH of four
task/size cells. Select 32 TRAINING base graphs per size by ascending SHA-256 of
`selection_salt + ':' + base_id`, tie-break by base_id; include BOTH tasks for
each selected graph. This gives 128 training examples/64 graphs, matched to
development in task, size, family, view and sample counts. All arms use identical
probes. No label, model output, loss, degree or distance informs selection.

Training/development graphs remain structurally disjoint under stage 12's
conservative WL and cut exclusions. Independently check every selected distance
label using Floyd-Warshall. Match task/size composition, not graph instances,
degree, distances or token lengths: report these descriptive covariates. Repeated
topologies and shared source graphs across tasks remain correlated. Count actual
stage 13 training exposures for every probe ID from its retained sampled order;
require that all selected training examples were used and development IDs were
never used. These training probes cover one quarter of the cache; their results
do not establish fitting on all 512 examples or a population-wide causal effect.

## Measurements fixed before execution

Reopen five final checkpoints. On BOTH probes, regenerate prompt-only answers
using the unchanged unconstrained stage 12 decoder, 128-token budget and strict
distance-vector parser. Preserve token IDs, stop reasons, malformed answers and
coverage. Development generations must reproduce stage 13 final generations
exactly. Predictor sees ONLY ID/prompt. No hint, teacher-forced reference, output
mask, context extension, truncation, head replacement or new demonstration.

Report per task/size and equally weighted four-cell aggregates:

- Exact-vector accuracy, valid-vector rate, per-example mean node accuracy,
  and context coverage. Invalid/unsupported completions score zero node accuracy.
- Correct nodes at the source, unweighted one-hop neighbors and unweighted
  multi-hop nodes; report numerator and denominator. For weighted Dijkstra the
  hop categories still use unweighted BFS distance, NOT weighted distance value.
  Invalid outputs count wrong in these ALL-example node denominators too.
- For valid vectors only: underestimates, correct values and overestimates,
  negative predictions on reachable nodes, first wrong node position, and
  normalized first wrong position. Report invalid counts separately; do not
  silently omit malformed cases from unconditional accuracy.
- Teacher-forced answer/EOS NLL and exact next-token accuracy, with separate
  digit-bearing versus other token counts/accuracy/NLL. A digit-bearing token is
  any target token whose decoded raw bytes contain ASCII 0..9; EOS is other.
  Tokens may span signs/punctuation, so this is a token diagnostic, not a clean
  semantic integer-value accuracy or an independently decoded node prediction.
  Use the original prompt/answer/EOS encoding and masks. Gold previous answer
  tokens condition teacher-forced scores; they do NOT condition free generation.

Calculate train-minus-development accuracy differences and development-minus-
train NLL differences WITHIN each task/size cell, then macro-average cells.
Do not reuse stage 13's composition-mismatched first-64 pooled training NLL.
NLL itself remains target-token weighted inside each cell. No graph-level pairing
across splits, confidence interval, p-value or causal mechanism is claimed.

## Fixed interpretation categories

For each arm, training-PROBE competence requires >=80% exact match in EVERY
task/size cell, >=95% pooled valid vectors and 100% coverage. The development
criterion uses the same thresholds on its full original 128 examples. These
are descriptive flags, not relaxed replacements for stage 13's failed endpoint.

1. Training probe fails: `training_probe_competence_not_established`. Quantify
   any held-out differences too; failure is already visible on seen examples,
   so held-out generalization alone cannot explain all observed failure.
2. Both pass: `both_probes_competent`.
3. Training passes, development fails, pooled exact gap >=0.10:
   `held_out_gap_after_training_probe_competence`.
4. Training passes/development fails without that gap:
   `cell_or_format_failure_without_large_pooled_gap`.

Teacher-forced digit accuracy may reveal conditional prediction difficulty or a
free-generation discrepancy, but cannot uniquely identify capacity, data amount,
optimization, serialization, exposure bias, attention or NCA structure as cause.
No outcome clears original random-rule readiness or authorizes another language
suite. Preserve all five arms, original margins, negative result and OOD skip.

## Verification, provenance and costs

Zero new scientific optimizer updates. New CPU inference/scoring timers and full
command wall time are distinct from imported historical adaptation/PPT/language
costs. No GPU or learning-speed measurement is made. A separate process reopens
all five checkpoint files and exactly reproduces all 1280 train/development
generations, teacher-forced scores, per-case errors and interpretation flags.
Compare checkpoint hashes before/after; verify every retained optimizer step is
600. Same-code replay is not independent neural implementation or retraining.

Software tests must exercise source/hop error accounting including invalids,
matched outcome-blind selection, digit/other token accounting and teacher-forced
loss/accuracy against an ordinary full forward. Export compact evidence and
provide weight-free public regeneration/scoring of the probes and free-generation
errors. Weights, raw cached data and correspondence stay private. Preserve frozen
prior sources/receipts; no remote/paid training, official language test, push or
external publication. Any next treatment must be declared separately after this
diagnostic, with all controls retained and the categorical-readiness bridge intact.
