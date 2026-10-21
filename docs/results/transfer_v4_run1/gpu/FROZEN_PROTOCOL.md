# V4: prospective replication and exact-prefix controls

Prepared 19 October 2026, after inspecting v3. This is a new validation-stage
replication/mechanism study. It does not retroactively change v3's primary
grid-endpoint result, and it is not confirmation on an untouched language task.
The objective remains **acceleration of language learning through pre-pre-training**.
The graph NCA supplies synthetic PPT trajectories; it is not the language model.

## Frozen treatment and budgets

Use `configs/transfer_v4.json` with six new paired cohort seeds
**59, 71, 83, 97, 109, 127**. Each seed changes model initialization, synthetic
rule bank and topology together, just as in v3. These variance sources are not
separately estimated. No inspected v3 seed (17, 29, 43) enters this cohort.

Seven arms enter the identical language stage:

| Arm | PPT data | PPT updates | Language updates |
| --- | --- | ---: | ---: |
| scratch | None | 0 | 2000 |
| grid_nca | Grid trajectories | 1000 | 2000 |
| graph_nca | Rewired graph trajectories | 1000 | 2000 |
| shuffled_nca | V3 whole-trajectory grid shuffle | 1000 | 2000 |
| shuffled_graph | V3 whole-trajectory graph shuffle | 1000 | 2000 |
| prefix_grid | New exact-prefix grid shuffle | 1000 | 2000 |
| prefix_graph | New exact-prefix graph shuffle | 1000 | 2000 |

Preserve v3's transformer, 4096 input tokens/update, 1024 context, tokenizer,
WikiText revision/splits, optimizer settings, 100-update evaluation cadence,
32 held-out synthetic evaluation rules, rule counts, compression band, burn-in,
25% ordered-slot graph rewiring, joint grid/graph rule admission and fixed-node
2x2 serializer. Do not introduce graph-local serialization or tune topology here.
Grid/graph still share frozen rule weights and rollout random draws. Actual common
starting backbones and language heads are saved; each language optimizer starts
fresh, with matched data order, scheduler and full-backbone transfer.

Both existing shuffled arms remain to connect the new mechanism comparison to
v3. V3 results themselves remain immutable. Do not pool v2/v3 with these cohorts.

## Stronger frequency/format control

For **each individual trajectory**, leave the entire first conditioning frame
and every start/end marker unchanged. Uniformly permute the subsequent patch
tokens **within the exact prefix used by training and evaluation**, separately
from any unused suffix. The exclusive prefix boundary is `context + 1` = 1025:
next-token labels include the token at absolute index 1024. Marker positions
and the initial-context mask are identical across a parent and its control.

This matches both the full trained input-plus-target prefix histogram and its
supervised patch-target histogram exactly, per rule. It also preserves the
initial conditioning frame and therefore its histogram. In the default v3
layout, the 1026-token trajectory has only a final end marker outside the
1025-token input-plus-target prefix: **the old shuffle already preserves that
prefix's overall histogram**. Its actual mismatch is exchanging patches with
the masked initial frame, changing both conditioning and supervised-target
frequencies. The new control fixes that mismatch. Separate suffix shuffling is
a general safeguard when a configuration has unused suffix patches; it does
not repair a demonstrated suffix-leakage bug in the default v3 run.
No fresh rule filtering or
selection is performed for any shuffled arm. Grid and graph use the same
permutation seed within a cohort/split; train and validation use distinct seeds.

This control destroys patch ordering while retaining rule-specific frequencies,
formatting and conditioning/frequency associations. It does **not** isolate all
possible physical mechanisms, preserve per-frame histograms, or eliminate all
statistical dependence. A graph-versus-prefix benefit would support an effect
of order-dependent structure beyond these matched quantities, not prove that
local interaction physics is the unique cause. Native PPT loss alone is not a
cross-dataset measure of dynamics learning.

## Admission gate and stopping

Every structured grid and graph PPT model must improve held-out patch NLL by
at least 5% from initialization and beat both training-fitted unigram and
previous-frame-copy mixture controls by at least 0.01 nats. The selected PPT
checkpoint minimizes held-out patch NLL; charge all 1000 updates regardless.
Shuffled gates are diagnostics, not admission requirements. Any structured
failure stops the entire language suite, exports diagnostics and retains every
cohort. No automatic retry, rule-threshold relaxation, seed replacement or
post-result budget/target change is allowed. Tiny byte fixtures can bypass the
gate only under `purpose=execution_check` and cannot support a research claim.

## Prospectively fixed comparisons

For each new cohort, define **one common target**: that cohort's scratch model's
final validation NLL at 2000 language updates. Find each arm's first recorded
crossing at or below that target, including step zero, with no interpolation.
This replaces v3's different reference-arm endpoints **only for this new study**.
It avoids comparing graph/grid and graph/control speed at different losses.

The two co-primary comparisons are:

1. graph_nca versus grid_nca;
2. graph_nca versus prefix_graph.

For each, report the paired ratio `reference tokens / graph tokens` at the
same scratch target. Ratios above one favor graph. Publish all six rows, strict
wins/ties, geometric mean, and a one-sided exact sign test. Ties count as
non-wins, conservatively. Use alpha **0.025 per comparison** (Bonferroni family
bound 0.05 for the two tests). Both tests must pass and both geometric means
must exceed one before describing support for a graph advantage on this target.
With six cohorts this stringent rule requires six strict wins in each comparison:
`p = 1/64 = 0.015625`; five wins yields `p = 7/64 = 0.109375`.
This design detects consistent effects and can remain inconclusive despite a
positive mean; it is not a guarantee of adequate power for small effects.

If either arm misses the target in any cohort, retain that row as censored and
withhold the all-cohort ratio and sign-test claim. Missing cohorts are not wins.
A zero-token crossing makes a token ratio undefined; do not turn it into an
infinite speedup. The independent test unit is the paired cohort, not a batch,
checkpoint, validation token or bootstrap draw. Six cohorts do not establish
generalization to larger models, new graph families or other language tasks.

Report all arms versus scratch, grid versus prefix_grid, original shuffled
controls, final NLL/perplexity and the historical reference-endpoint comparisons
as secondary/descriptive outcomes. A graph advantage over grid/control does not
by itself establish a benefit over scratch; report that crossing separately.
An eventual confirmation experiment must reserve new data and its own criterion.
The official language test is neither downloaded nor evaluated in v4 defaults.

## Timing, evidence and execution

Report language-token savings as the primary learning-speed measurement.
Report language-stage wall time and cumulative generation + PPT + language
time separately, at the same target, retaining censored cohorts. Cumulative
cost conservatively charges the whole paired preparation to each synthetic arm;
it is not an isolated generator throughput benchmark or cloud-billing measure.
Retain v3's detailed timing exclusions in `environment.json`. Model/data
preparation and all checkpoint verification add to actual notebook runtime.

V4 runs 36 PPT jobs and 42 language jobs (78 selected stage checkpoints and
six actual initialization snapshots). Scaling the measured v3 stage workload
suggests roughly **7-8 hours on a comparable T4**, with variable preparation,
verification and export overhead beyond that estimate. This is an extrapolation,
not a new GPU measurement or service quota guarantee. The notebook prints a
disposable timing calibration on the actual assigned device before training.
Do not shrink cohorts or stages after inspecting new outcomes and still call
the result this frozen protocol; declare a new development version instead.

`analysis_plan.json`, its checksum in the run environment, and an archived
`FROZEN_PROTOCOL.md` are saved before benchmark model initialization/training.
They retain the config, study criterion and runner hashes. The verifier checks
plan/source/data identity, prefix-control invariants, topology/rule separation,
all selected/final checkpoint losses, actual initialization/transfer hashes,
matched language controls and primary analysis recomputed from raw curves.
Keep original ZIPs and all failed/stopped workflows. Completion is engineering
success, not evidence that acceleration occurred.

Upload `kaggle_nca_transfer_v4.ipynb`, enable one GPU and Internet, and Run All.
The last cell prints `WORKFLOW FINISHED: complete` or
`WORKFLOW FINISHED: stopped_at_synthetic_gate` and exports
`nca_transfer_v4_results_<timestamp>.zip`. No remote training is launched by
preparing this notebook locally. CPU checks only verify implementation.
