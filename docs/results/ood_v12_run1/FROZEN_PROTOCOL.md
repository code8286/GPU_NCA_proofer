# Stage 12: out-of-domain graph evaluation and historical calibration

Declared 30 October 2026 before new model predictions. This stage implements an
OOD evaluation layer following methodological feedback and the public
[CLRS-Text paper](https://arxiv.org/html/2406.04229v1). The paper motivates
training on controlled task distributions and evaluating final algorithmic
answers beyond those distributions. Its official
[implementation](https://github.com/google-deepmind/clrs/tree/d1c2ad7af8437c7536cf329d9cef8fdf93184d9d/clrs/_src/clrs_text)
is pinned locally with the Apache-2.0 license and checksums.

Private correspondence stays outside the publication selection. Feedback is
methodological input, not supervision, collaboration, endorsement or evidence
that graph pre-pre-training accelerates learning. No reply is sent by this work.

## What changes, and what stays separate

Add evaluation beyond the target training distribution to the research question:
does graph versus grid pre-pre-training improve an otherwise identical language
stage and subsequent graph-task adaptation, including generalization? Keep
scratch, grid, graph, prefix-grid and prefix-graph initializations. Compare
language-stage learning and total synthetic/language costs separately. If a
task-adaptation stage is used, its data order, head initialization, optimizer,
schedule and full budget must also match across arms. It is not pre-pre-training.

This stage has NO fitting command. It creates a reusable dataset/scorer/predictor
layer, validates its measurement controls and cross-evaluates five already-trained
language checkpoints in a tiny historical zero-shot calibration. It does not
replace stage 11's categorical bridge or clear original random-rule readiness.
Stage 11's weighted pass remains a fixed-binary-rule result; its broad criterion
still fails. No new language or algorithmic-adaptation run is authorized here.

## Explicit local task adaptation

Start with BFS distances and positive-integer-weight Dijkstra distances on
undirected simple graphs. Output the complete source-distance vector in the
declared input-node order, with -1 for unreachable nodes. This avoids selecting
one arbitrary shortest-path predecessor when ties exist. These are local
CLRS-inspired task/output adaptations, not official CLRS-Text generation or
scores: the official BFS/Dijkstra output and text conversion differ. Do not
compare our numbers with published CLRS scores. The separate `score-official`
adapter accepts official prompt/references JSONL exports and preserves their
final reference strings, rather than reinterpreting them as distance vectors.

Distance labels are generated using queue-based BFS or heap-based Dijkstra and
checked with an independently implemented all-pairs Floyd-Warshall oracle.
Use no algorithmic trace in the prompt. The predictor receives ONLY ID/prompt.
Teacher-forced reference likelihood is not an answer-correctness measurement.

## Frozen distributions and identities

Seed 20261030. ID sizes 8 and 10; size OOD sizes 12,16,20. The ID family is
Erdos-Renyi with p=min(0.7,2.5/(n-1)), conditioned on connectedness and at least
n+1 edges, capped at 3n edges. The extra cycle prevents a single cut from
creating a connected tree used in the held-out family shifts.

Family OOD at ID sizes uses a random recursive branching tree (maximum degree
at least 3), a randomly labeled path, or a forest of two equal-sized recursive
trees. The joint shift uses those families at the larger sizes. These are
separate cells; report each size/family/task, not just a pooled average.
The density/degree changes induced by these samplers are part of their declared
distribution shifts, not separately isolated causal treatments.

Training has 128 base graph instances per ID size; ID development has 32 per
size. Assessment has 32 per cell: two ID cells, three size cells, six family
cells and nine joint cells. Source is sampled uniformly, rejecting isolated
sources. Edge weights are uniform integers 1..9; BFS ignores weights. Both
tasks share a graph instance. Display labels are distinct random letters from
A..Z at EVERY size, so larger graphs do not introduce previously unavailable
label symbols. Answers follow input-node order, never an implicit spatial layout.

Use an unweighted Weisfeiler-Lehman fingerprint invariant to node relabeling to
exclude structurally matching graphs across train/development/assessment. Also
reserve each candidate's two task-specific edge-cut topologies before proceeding
to later splits, even where those control views are not written as training data.
This prevents intervention graphs from leaking across splits. Fingerprint
collisions may conservatively reject different graphs; this is not a complete
isomorphism canonicalizer or a proof that all non-isomorphic graphs differ.
Rejection exhaustion is an error, never permission to loosen the identity rule.

Repeated rooted/labeled/weighted versions inside a split are permitted and their
topology-cluster counts are retained. Paths in particular have one unrooted
topology per size. Thirty-two sampled instances are not thirty-two independent
topology cohorts. No example-level confidence interval is claimed. These are
development diagnostic data, not reserved untouched confirmation; the latter is
not generated. Freeze new confirmation seeds/distributions after development.

## Matched functional controls

Assessment has four views of each graph/task:

1. Native prompt and final distance vector.
2. Consistent random relabeling of all printed nodes/edges/source; answer in
   input-node order stays identical.
3. Edge-list reordering; graph and answer stay identical.
4. Remove the cheapest source edge under that task's weights. Strictly positive
   weights guarantee at least one distance changes. Keep the original query and
   report BOTH native and changed answers correct, not merely different answers.

The cut changes edge count/degrees, so this is functional input sensitivity,
not a degree-matched proof of a graph-specific pretraining mechanism. Conditional
correctness on both invariant views is required; consistency of two wrong
answers is reported separately and cannot count as competence.

Before neural prediction, require the graph oracle to score 100% on all cells
and paired tests. A cached-native-answer control is deliberately correct on
native/rename/edge-order and wrong on every changed graph; it must fail paired
native/cut correctness. A source-only/no-edge predictor must also fail that
paired test. These are measurement controls, not learned models or PPT evidence.
Malformed/missing/duplicate predictions cannot be silently discarded. Report
format validity, vector exact match, node accuracy, coverage and paired metrics.
Context overflow is explicitly reported and makes a comparison ineligible;
never truncate input graphs or reinterpret unsupported contexts as neural errors.

## Tiny historical zero-shot calibration, frozen before predictions

Use v4 cohort 59, fixed final language endpoints after the identical 2000-update
language stage: scratch, grid_nca, graph_nca, prefix_grid, prefix_graph. Verify
checkpoint hashes, actual common language-head initialization, language data
order, schedule and update/token counts. No endpoint is selected by OOD scores.
Keep the original failed stronger synthetic readiness and negative graph primary.

Take the first base graph from each declared assessment cell: ID cyclic ER n=8,
size cyclic ER n=16, family path n=8, joint path n=16. Both tasks and all four
views give 32 prompts per model /160 predictions. This single-cohort pilot is
not representative of the full suite and cannot support a statistical claim.

Retain the saved GPT-2 vocabulary and context 1024. Greedy decode at most 128
tokens from the prompt only; no few-shot demonstrations, hints, answer prefix,
logit mask, grammar constraint, model retuning or optimizer updates. Stop at EOS,
closing bracket, token cap or an irrecoverably invalid vector-prefix character.
The last option rejects an already-invalid completion; it does not force a
correct token. Record token IDs, text, stop reason and coverage. Project only the
last hidden position to the vocabulary using the same tied LM head; regression
check against the ordinary forward function. All complete prompts plus generation
budgets must fit without context extension. Reopen each checkpoint in a separate
invocation and reproduce every prediction, stop reason and score.

These WikiText-trained micro-models were not taught this graph-task interface.
A zero or invalid-format floor diagnoses inadequate task elicitation; it does
not refute graph transfer. Read-only historical evaluation is development work,
not admission of the unready synthetic configuration to another training study.

## Subsequent matched learning-speed study: not executed by this stage

After eligible synthetic initialization is frozen, run the SAME full language
stage for every arm. Then, if required to learn the task format, use identical
ID-only algorithmic task adaptation for every arm and log it as a third stage.
Choose schedules/checkpoints using ID development only. Report a predeclared
ID exact-match crossing (e.g. 80%) versus language/probe tokens and time; unreached
targets are censored, never interpolated or replaced after outcomes. Also report
fixed-budget OOD cell scores and generation + PPT + language + adaptation +
evaluation costs. Better OOD accuracy at one endpoint is not learning acceleration.

Use fresh paired rule/data/topology cohorts and at least six independent cohorts
before graph-specific confirmation. Require graph to beat BOTH grid and matched
graph shuffle on the predeclared primary OOD cells/criterion, retaining prefix-grid
and scratch. Freeze targets, minimum effects, cost accounting and multiplicity
rules before fitting; do not choose them from this historical zero-shot pilot.
All learned arms need retained stronger synthetic-readiness evidence. Do not
change topology, NCA rule family, serializer and PPT budget simultaneously.
The existing categorical-readiness bridge remains separately required.

No remote/paid job, official-language-test access, new fitting, push, email reply
or external publication is launched by this implementation.
