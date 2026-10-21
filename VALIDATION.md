# Local verification — 17 October 2026

This records the historical CPU diagnostic. The subsequently returned GPU pilot
is documented in [the audited review](docs/results/pilot1/REVIEW.md); no new GPU
benchmark was run during publication preparation. New check transcripts are in
`logs/checks.jsonl`, with scope and results in `LOGBOOK.md`.

- Scientific checks: **11 passed** (`python -m unittest discover -s tests -v`).
- Full model: **1,857,024 parameters**, four layers, width 192, context 128.
- Diagnostic budget: 60 updates per arm, batch 4, 30,720 total training tokens per arm, one paired seed (17).
- All five checkpoints reloaded; held-out test losses reproduced with **zero absolute error**.
- Initial parameter checksums match across all five arms.
- The 2.5 validation-nats/byte target was not reached. All time-to-target comparisons remain unresolved.

| Arm | Final validation NLL | Held-out test NLL |
| --- | ---: | ---: |
| scratch | 2.798420 | 2.889472 |
| reserve_text | 2.806562 | 2.896473 |
| grid_nca | 3.196839 | 3.298223 |
| graph_nca | 3.175359 | 3.277369 |
| shuffled_graph | 3.185969 | 3.284273 |

The short check favors text-only training over graph-NCA on language loss.
Graph NCA slightly improves loss relative to grid NCA and its shuffled control in this one-seed diagnostic,
but this is insufficient to claim a topology benefit, statistical significance, or GPU acceleration.
The Kaggle pilot uses the predeclared longer budget; do not treat this CPU diagnostic as that pilot.

## Historical evidence (original paths in the research workspace)

- `runs/cpu_verified/verification.json`: independently recomputed test losses and valid checkpoint checksums.
- `runs/cpu_verified/summary.csv`: metrics, parameter counts, initialization checksums, and checkpoint hashes.
- `runs/cpu_verified/report.json`: explicitly censored time-to-target comparisons.
- `runs/cpu_verified/curves.svg`: validation-loss traces with synthetic steps counted.
- `runs/cpu_verified/environment.json`: exact runtime and source identity.
- `data/cpu_check/manifest.json`: text ranges, checksums, graph identity, and generator diagnostics.

Experiment source SHA-256: `76b535acbf08050295648d5692d46690366c1f4802d5668c99edeca003cf1981`.

The separate six-update smoke run also completed and reloaded all five checkpoints successfully.

Compact publication copies of summary, report and checkpoint verification are
in `docs/results/cpu_check/`. Full datasets/checkpoints remain local. The source
hash recorded here identifies the historical experiment, not the entire evolving repository.

## Version 2 checks — 17 October 2026

These are new CPU/software/data-path checks, separate from the historical GPU
pilot above. They do not establish learning acceleration.

- All **33** unit/scientific/integrity checks passed, including **16** v2 tests.
- Offline v2 workflows completed for normal execution, a deliberately failed
  synthetic gate, and a predeclared confirmation-fixture path. Selected stage
  checkpoints reloaded with zero evaluation error; language head, optimizer,
  schedule and batch-order controls passed. A stopped gate produced no language
  checkpoints.
- Default-sized language model (12,306,048 parameters) accepted the real 50,257
  BPE vocabulary and completed a short forward/backward shape check. This is not
  a full-context GPU memory/performance measurement.
- Real pinned WikiText train/validation preparation passed: 2,403,644 training
  tokens and 248,461 validation tokens. The confirmation test was not downloaded.
- A CPU categorical-rule calibration retained 16 rules from its first 64 screened
  candidates within the declared band; its sampled fresh rollouts had no final
  fixed transition. This is a small generator check, not a corpus-wide guarantee.
- Historical `experiment.py` retains its original SHA-256.

Full commands, outputs, failed attempts and v2 source identities are appended in
`logs/checks.jsonl`. An initial OS-temporary-directory permission failure was
resolved by retaining offline-check artifacts under the workspace's ignored
`tmp/` directory. These checks preceded the returned v2 GPU archive below.

## Returned version 2 GPU evidence — 17 October 2026

The user ran the notebook on a Kaggle Tesla T4 and supplied
`nca_transfer_v2_results_20261017_065212_924967.zip`. This is imported GPU
measurement, distinct from the earlier CPU fixtures and from configured CI.
See [the results review](docs/results/transfer_v2_run1/REVIEW.md) for the
independent audit, raw curves and scope of verification.

All three grid-NCA seeds passed the synthetic gate. All nine language runs used
2000 updates / 8,192,000 input tokens each. Grid and shuffled pre-pre-training
both yielded a 1.178x geometric language-token speedup to the paired scratch
endpoint. Their final mean NLLs were 5.233321 and 5.231528 versus 5.266091 for
scratch. Grid's cumulative-time speedup was 0.886x, so the measured language
savings did not repay pre-pre-training at this horizon. No graph treatment or
official-test evaluation was performed.

Independent local CPU replay loaded all **15 development checkpoints**, with
maximum validation-metric error **1.9563e-7**. Archive CRCs, trusted source and
data identities, actual rule disjointness, shuffled histograms/markers, fitted
controls and raw crossing calculations passed. Actual PPT-transfer and batch
order hashes match. Initial/head/schedule hashes agree within the GPU run, but
are not all recreated bitwise by the newer Windows runtime; the retained
step-zero smoke core differs numerically by at most 7.45e-9. Failed regeneration
attempts and the corrected audit scope remain in the append-only check log.

## Graph-transfer v3 software checks — 17 October 2026

This is new CPU software evidence, separate from the returned v2 GPU result.
All 58 unit/scientific/integrity checks passed, including 25 v3 checks. They
cover degree-preserving graph rewiring, exact zero-rewiring grid recovery,
node-relabeling equivariance of the message update, paired random draws, joint
rule selection, independent caches, graph-only gate rejection and censored
graph-vs-grid reporting, alongside the stage controls retained from v2.

Offline complete, gate-stopped and explicit confirmation-fixture workflows
passed. The complete path uses two paired replicates, all five language arms
and 18 selected-stage checkpoint replays. Stored initial tensors are checked;
an intentionally corrupted initialization is rejected before model replay.
Actual rule-weight disjointness and shuffled histograms/markers are verified.
The fixture confirmation path does not download or inspect the official test.
Full commands and source identities remain in `logs/checks.jsonl`. CI changes
are configured only. These checks preceded the returned v3 GPU archive below.

## Returned graph-transfer v3 GPU evidence — 19 October 2026

The user supplied `nca_transfer_v3_results_20261018_161811_573163.zip`, a
completed Kaggle Tesla T4 run. All six structured models passed the synthetic
gate. All 15 language runs received the same 2000 updates / 8.192M input tokens.
Graph's final mean NLL is 5.232262, versus 5.237811 for grid and 5.234889 for
shuffled graph. Graph reaches the paired grid endpoint at 1800 / 2000 / not
reached for seeds 17 / 29 / 43. An all-seed primary speedup is withheld.
This is a small, mixed topology effect, not a robust graph-acceleration result.

Graph reaches the paired scratch endpoint with 10–20% fewer language updates;
its cumulative-time speedup is 0.871654x (about 14.7% more recorded time).
The official test remains absent. [The independent review](docs/results/transfer_v3_run1/REVIEW.md)
distinguishes imported GPU timing, new CPU replay, saved initialization checks,
raw crossing calculations and their limits. No new training was launched.

Independent CPU replay passed all **27 selected stage checkpoints and three
saved initializations**, with maximum metric discrepancy **1.8552e-7**. All
215 archive members passed CRC/path checks. The 11 source hashes, cached data,
joint filter/topology treatment, reconstructed batch orders, fitted controls,
gates, selection and raw reference crossings agree. Stored common initialization
and actual PPT-transfer hashes pass; cross-runtime binary LR-schedule regeneration
is not claimed. The original GPU verification remains unchanged. This is new
CPU verification of imported measurements, distinct from the prior execution
fixtures and from configured CI.

## V4 prepared study — new CPU checks, 19 October 2026

At preparation, v4 had only CPU checks. Its separate runner preserves
all measured v1/v2/v3 Python sources. Fourteen new control/analysis regressions
passed: supervised-prefix histograms and fitted unigram probabilities match,
initial frames/markers stay fixed, unused suffix tokens cannot enter training,
corruption is rejected, censored/zero-token targets withhold aggregate claims,
ties remain non-wins, and six versus five strict wins produce the fixed exact
sign-test probabilities. Model/generator/training numerical code and default
mechanistic settings match v3 (checkpoint schema differs).

An explicit default-layout regression distinguishes overall prefix counts from
supervised counts: v3's unused suffix is just an end marker, so its original
shuffle already preserves the overall prefix histogram. The changed masked
initial frame and supervised patch-target histogram are the actual control gap.

The logged CPU end-to-end fixture passed two complete paired cohorts, one
gate-blocked cohort, and one explicit byte-test-path cohort: **45 selected
stage checkpoints and four saved initializations** were replayed, plus any
distinct final PPT states. These tiny byte fixtures are execution checks and
are not language-transfer evidence or official-test access. Changed analysis
plans and corrupted initialization files are rejected before checkpoint replay.
Six default cohort topologies differ, and 16 paired rules per cohort passed the
unchanged admission filter from 64 candidates each, with prefix controls checked.
This small generation check does not assert full-cache yield or synthetic learning.

The [v4 protocol](docs/TRANSFER_V4_PROTOCOL.md) records the prospective common-target
comparisons, six new seeds, preserved controls, timing extrapolation and limits.
CI steps are configured separately; a local CPU pass does not assert a remote
CI run or a GPU speedup. Packaging/parity results are recorded in the logbook.

The final full local suite passed **72 tests**. An isolated extraction of all
25 notebook payload files also passed a complete CPU fixture (13 selected stage
checkpoints and one initialization), with the starter ZIP's exact inventory,
file bytes and CRC verified. This additional check confirms that the upload
does not depend on the surrounding research checkout. No new GPU measurement
was made by either workflow.

## Returned v4 GPU run and independent CPU audit — 19 October 2026

All 78 selected development stage checkpoints and six saved initializations
passed CPU replay of the returned T4 run, maximum metric error **2.7828e-7**.
All 454 ZIP members passed streaming CRC/path checks. Configuration, 14 runner
hashes, archived plan/protocol, cache identities, topology/joint filter, rule
separation, recreated shuffles and batch orders, gates, selection, actual common
heads/backbones and selected-PPT transfer agree. Raw crossings and frozen exact
sign-test decisions independently reproduce the imported report. The official
test is absent and the imported GPU verification is unchanged. Intermediate
language states were not saved; their logged curve values are not CPU-replayed.

The new causal-frequency diagnostic is post hoc and does not revise v4 admission.
It beats all structured PPT models on held-out patch loss, indicating that the
old gate cannot establish dynamics-specific learning. Its closed form matches
explicit causal sequential prediction; markers/unused future tokens are excluded.
The final local suite passed 78 software tests, distinct from the 78 checkpoint
replays. One Windows raw-path fixture failure and its correction are retained in
the append-only check transcript. See [the complete review](docs/results/transfer_v4_run1/REVIEW.md).

## Stage 5 readiness — completed CPU diagnostics, 19 October 2026

Eight new regressions and the complete **86-test** local suite pass. New checks
cover both readiness margins, non-finite rejection, the audited frequency joint
likelihood, known-rule positive/frequency-only negative controls, unchanged v4
patch-score agreement, unused-future exclusion, plan tampering and all-cohort
language blocking/acceptance with bound checkpoint identities.

The frozen real-data diagnostic completed **48 cells from 24 existing PPT
checkpoints**, including per-rule scores and native replay (maximum NLL error
1.6155661e-7). All 12 structured cohort decisions fail both 0.01-nat requirements;
the actual guard refuses language-start authorization. Seven compact evidence
copies retain frozen receipts and identities. This is CPU evaluation on inspected
v4 rules, not new neural training, GPU timing or official-test evaluation.
The transition-table positive control validates measurement but not transformer
learning. Full evidence and limits are in the
[stage 5 review](docs/results/readiness_v5_run1/REVIEW.md). CI is configured
separately; these local checks do not claim a remote CI execution.

## Stage 6 trainability — local implementation checks

The full local suite passes **94 tests**, including eight new phase-split,
known-transition/masking, exact-control, declared-configuration, source/input
tampering, held-out-access and overwrite-protection regressions. A retained
two-arm/two-update CPU fixture completes scoring and exact initial/final
checkpoint replay. Its tiny model and failed gate are execution evidence only.
The full-backbone task has its own frozen plan, final-only endpoint and evidence;
do not confuse fixture execution with a model-learning result. CI configuration
includes the fixture but does not establish a completed remote CI run.

The full task completed locally on CPU on 20 October: six full-backbone models,
1000 updates each, followed by final-only assessment. All three coherent-trained
models pass both readiness margins with 100% held-out patch accuracy. Separate
reloading of three starting and six final snapshots reproduces all **18 cells**
and decisions exactly, including optimizer update counts and bound source/input
identities. This is replay with the same scorer, not independently implemented
loss code or training reruns. Compact evidence retains 21 exact copies, receipts,
all per-initial-condition scores and six independently reconstructed batch orders.
V4/v5 source identities match their prior frozen plans. The descriptive control
inspection also records all three shuffle-trained models passing the absolute
margins; the review does not equate admission with coherent-training causality.
See the [full report](docs/results/trainability_v6_run1/REVIEW.md).

## Stage 7 spatial bridge — completed implementation checks

The complete local suite passes **104 tests**, including ten new checks for
exhaustive involution/bijection on all 65,536 grids, 16,512 joint orbit classes,
cross-patch neighbor influence, matched initial states and controls, scoring
masks, oracle ordering sensitivity, unchanged settings and frozen evidence.
A retained four-arm/two-update CPU fixture completes assessment and checkpoint
replay. It is execution evidence, separate from full-backbone learning.

The [main protocol](docs/INTERACTION_V7_PROTOCOL.md) fixes twelve 1000-update
models and separate readiness/own-patch-table comparisons. A
[supplementary diagnostic](docs/INTERACTION_V7_FIRST_TRANSITION.md), frozen
during training before assessment, targets the first transition and an eligible
neighbor intervention, where two-frame recall is unavailable. This does not
retroactively amend the main plan. CI has a fixture configured; no remote CI
execution is implied.

All twelve full models completed on CPU, followed by final-only assessment and
exact replay of all 36 primary cases from three initial/twelve final checkpoints.
The supplementary original/intervention queries also replay exactly for all 18
checkpoint cases. Receipt checks retain the timing evidence that its plan was
frozen during training and before primary assessment. Public provenance binds
40 unchanged private-run copies; batch orders, budgets, common initial states,
joint split classes and historical source identities independently agree.

Readiness, local-table qualification and the supplementary NLL criterion all
pass, with substantial remaining first-transition/intervention errors. This is
new CPU known-rule evidence, not a language-transfer result. Full outcomes and
limitations are in the [stage 7 review](docs/results/interaction_v7_run1/REVIEW.md).

## Stage 8 loss weighting — completed local study

The complete local suite passes **112 tests**. Eight new regressions cover
normalized half-mass weighting, patch/marker proportions, exact factor-one
loss/update/parameter/index-order equivalence, initial-context masking, invalid
weights/empty targets, locked treatment/decision settings and blocking assessment
until all weighted models finish. The initial focused invocation retained a
harmless scalar-conversion warning; its detached-scalar correction passes in the
full clean suite. A retained two-update CPU execution fixture completes replay;
its unequal imported/new budgets are explicitly excluded from scientific evidence.

Six new full-backbone CPU models completed the declared 1000-update endpoint.
The plan froze before new fitting; all six finished before assessment. Independent
reconstruction of sampled index orders matches the imported six standard controls,
as do actual initial weights, 188,000 input tokens and 168,000 supervised targets
per model. Exact data/query copies and source/config/checkpoint receipts pass.
All 30 trajectory and 15 original/intervention checkpoint cases replay exactly
with the frozen scorers, including exact agreement with original standard scores.
This is a separate scorer replay, not independent loss code or a training rerun.

All 21 copied evidence files have identical source/public hashes. The stage 7
integrity check also passes unchanged. The frozen primary fails on every coherent
seed despite passing trajectory readiness; no seed, weight, budget or endpoint
was changed. Full results, costs and limitations are in the
[stage 8 review](docs/results/loss_weight_v8_run1/REVIEW.md). This is development
evidence on inspected data. No remote CI result, GPU/language run, fresh
confirmation or new mathematical compiler execution is implied.

## Stage 9 split-fit diagnostic — completed CPU inference

Eight new scoring/comparison regressions pass; the full local suite passes
**122 tests**, including two subsequent starter-snapshot integrity regressions.
The eight scoring checks exercise actual causal prefix lengths, fixed target
positions, patch/marker NLL arithmetic, exclusion of all later frames, short/long
agreement on a real small backbone, invalid arrays, all control conditions,
comparison signs/margins and generalization-gap arithmetic.

The new protocol and source/input/checkpoint identities froze before computing
new split scores. All 15 existing initial/final checkpoints score both data
versions across train/development/assessment (90 cells), with zero new optimizer
updates. A separate invocation reloads them and reproduces every cell, per-query
vector, category and gap exactly using the same scorers. All 15 coherent
assessment first-patch vectors also agree exactly with stage 8. Longer causal
prefix NLL differences stay below 1e-5 (observed maximum 4.76837158203125e-7)
and correctness is identical. Replay is not independent loss code or retraining.

Seven copied evidence identities and all nested score aggregates/comparisons
pass, including logged freeze-before-scoring chronology. The mixed fit versus
held-out findings and whole-transition/first-patch distinction are retained in
the [review](docs/results/fit_v9_run1/REVIEW.md). This is post-hoc development on
inspected data, not new confirmation, a language gate, GPU measurement or remote
CI result. Raw inputs, checkpoints and historical sources remain preserved.
