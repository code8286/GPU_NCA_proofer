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

## Stage 10 diversity ablation — completed CPU study

Ten new scientific regressions pass, and the full root suite passes **134 tests**,
including two subsequent publication-policy checks. The scientific tests
cover exact nested trajectories/shuffles, joint-phase and intervention exclusions,
deterministic expansion/capacity limits, replacement-sampling traces, cache versus
actual exposure, null empty subsets, all-objective/all-seed decisions, fixed
settings and assessment blocking before every model completes. A tiny isolated
four-arm/two-update fixture with mocked provenance checks execution separately.

The scientific plan froze before training all twelve larger-cache endpoints;
assessment followed completion of every fixed 1000-update model. Initial states,
update/input/target budgets and actual sampling digests match declared controls.
Independent reconstruction verifies every larger-cache trajectory and shuffle,
4096 unique joint phases and unchanged eligible intervention exclusions.

All 54 full, 108 fit and 27 intervention cells replay exactly after loading 27
initial/final files. All common historical score vectors reproduce stages 8/9.
Nine per-seed raw files retain query scores/exposure masks; 42 copied evidence
identities, nested aggregates/subsets, both objective decisions and logged
freeze-training-assessment chronology pass. Replay uses the same scorers, not
independent loss code or a training rerun. No result is inferred from remote CI.

The weighted stratum passes, standard fails and study-wide primary fails. No
criterion, seed, endpoint, cache size or weighting factor changed after outcomes.
The [review](docs/results/diversity_v10_run1/REVIEW.md) preserves all controls,
costs, imperfect accuracy and coarse-readiness limitations. This is development
evidence, not language/GNN acceleration, confirmation or a new official test.

## Stage 11 fresh-phase confirmation â€” completed local CPU study

Nine new stage 11 and three recovery regressions and the full root suite pass 146 tests. They check fresh
joint-phase and intervention reservation, deterministic generation/capacity
limits, exact nested prefix shuffles and known neighbor targets, frozen settings,
all-seed/stratum decisions, assessment blocking before all fits, tiny eight-arm
execution with mocked provenance, and distinct CSV complete/readiness decisions.
The tiny fixture is software evidence, separate from the 24 scientific models.

The plan froze before all 24 full-budget fits. Independent reconstruction checks
fresh arrays/shuffles and exclusion of 4926 historical phase classes. All 128
intervention partners are reserved prospectively. Source/input/initial-state
receipts, per-model 1000 updates /188000 input tokens /168000 targets, shared
within-seed starts and actual row-order digests pass. Assessment follows all
fits; separate checkpoint replay reproduces all 54 full, 108 fit and 27
intervention cases, per-query vectors and decisions exactly with the same
scorers. This is not independent loss code or a rerun of fitting.

All 71 exported evidence hashes, nested arithmetic/subset masks, CSV decisions,
summary and frozen chronology pass. The interrupted status write and journaled recovery preserve 21 completed
endpoints; the restarted fit reproduces its recorded 800-update prefix exactly.
Charge discarded work: actual 24800 updates /4662400 inputs /4166400 targets,
versus 24000 /4512000 /4032000 contributing to final endpoints. The weighted
primary and standard/broader criteria remain separate. No post-outcome criterion or treatment changed.
The [review](docs/results/confirmation_v11_run1/REVIEW.md) records the outcome,
controls and costs. Historical stage 10 CSV ambiguity is documented separately;
its original JSON and review decisions remain correct and unchanged. Static
Lean source/retained-evidence checks pass without a new Lean compiler execution.
No remote CI, GPU/language run or fresh mathematical compilation is implied.

The standalone copy also passes all 146 tests and publication integrity for
866 selected files, 82 embedded sources and 12 pinned reference checksums.
All 19 Lean package/evidence files, 11 theory files and 16 paper sources/figures/
records are recursively included with exact root/Prod parity. Generated paper
output remains excluded and preserved locally. These checks copy and inspect
existing mathematical evidence; they do not execute Lean or LaTeX.


## Stage 12 OOD measurement — completed CPU validation

Fourteen new regressions and the full local suite pass **160 tests**. They
cover independent weighted/unweighted shortest-path oracles; all four graph
families and several sizes; guaranteed causal cuts; invalid edges; relabeling-
invariant structural exclusion including interventions; deterministic generation;
strict full-vector/official-reference semantics; missing/duplicate predictions;
wrong-but-consistent controls; censored ID-only target selection; future paired
stage metadata; last-position/full-forward agreement and explicit context refusal.

The protocol/config/data/source/checkpoint identities froze before new neural
predictions. Independent graph labels agree on all 5760 written examples; all
5120 assessment oracle predictions and 1280 paired native/cut cases pass. Cached-
native and source-only paired correctness is zero. A separate invocation reloads
all five historical v4 final language checkpoints and reproduces all 160 raw
generated predictions and scores. Same-code neural replay is not independent
neural inference or repeated fitting. These models make no valid task answers;
the floor cannot support a graph-transfer effect or no-effect conclusion.

Thirteen exact public/private evidence copies, per-cell CSV accounting, source
identity, original common language stages, frozen chronology and exclusion of
private correspondence pass. Original readiness failures remain unchanged.
No newly measured GPU training or language/official-test access is implied.
Private tokenizer cache setup and sandbox DNS failures and their resolution
remain logged. Seven pinned upstream files and Apache license verify separately.
The informal [control arguments](docs/theory/OOD_graph_controls.md) add no new
Lean or LaTeX compiler execution. See the [review](docs/results/ood_v12_run1/REVIEW.md).

The standalone distribution also passes all 160 tests, without the private
historical model folders. The checkpoint-independent dataset generator exactly
recreates all three frozen data files. Publication integrity passes for 907
selected files and 82 embedded sources. All 304 local links and full starter
content identities pass. Recursive source/retained-evidence inclusion and exact
root/Prod parity pass for 19 Lean, 12 theory and 16 paper files, including the new
informal control arguments; no Lean/LaTeX compiler or remote CI is executed.

## Stage 13: completed matched ID adaptation, 30 October 2026

Eight new pipeline tests and the full local 168-test suite pass. All five
historical language endpoints complete the separately frozen 600-update CPU
algorithmic-adaptation budget (3000 scientific updates), identical actual sample
order, LR schedule and per-arm 562855 input /96016 supervised answer-EOS tokens.
All final models score 0/128 exact ID answers with 98.4-100% valid vectors; every
per-task/size exact criterion fails, all 80% crossings are censored and OOD is
skipped. This is historical trainability calibration, not synthetic-readiness
admission, new language/PPT training or a graph-acceleration result.

A separate invocation reopens all 20 model/optimizer checkpoints and reproduces
all 2560 generated answers, token IDs/stops, scores and train/development NLL
exactly; optimizer steps and actual exposures pass. This is same-code neural
replay, not independent fitting or loss implementation. Independent JSON parsing
confirms the observed accuracy/format values; all 80 CSV cells, 32 exact evidence
copies, source/input identities, freeze-before-fit chronology, decisions and
private-artifact exclusions pass the evidence audit. Public score reproduction
regenerates ID dataset bytes and all 2560 answer scores without private model
weights/data, tokenizer cache or network. It does not replay neural inference.

The declared first-64 training NLL probe is eight-node-only while development
mixes sizes; do not interpret pooled NLL differences as a matched generalization
gap. Prior-receipt integrity checking hashes old assessment bytes but only ID
train/development is parsed for fitting; the frozen protocol's literal not-read
wording has a separate explicit clarification. No reserved official-language
test, fresh confirmation, new OOD predictions or remote/paid GPU is accessed.
See docs/results/adaptation_v13_run1/REVIEW.md and the usage guide.

Stage 13 final distribution: root AND standalone suites each pass 168 tests;
standalone public-only data/score reproduction passes all 2560 answers. All four
notebooks regenerate, 321 local links and the starter CRC/955 content hashes pass.
Publication checks 955 selected files/82 embedded sources. Recursive mathematical
source/evidence membership and parity pass for 19 Lean, 12 theory and 16 paper
files, without new compiler execution. Original stage 12/reference evidence stays
valid. Final direct sync/check follows the logged checks to preserve log parity.

## Stage 14: completed matched ID checkpoint-error audit, 31 October 2026

Eight new invariants and all 176 root tests pass. A separately frozen checkpoint-
only study scores five fixed stage 13 final models on 128 training and 128
matched development examples, 32 per task/size cell. All selected train IDs were
actually used 9-10 times; no dev ID was in optimizer batches. Each arm has 0/128
exact answers on BOTH probes. Node accuracy is 29.55-32.89% train, 18.95-21.60%
dev. Teacher-forced digit accuracy is 35.06-39.61% /24.36-28.09%, while other
accuracy is 100% /99.92-100%. Matching fixes the prior NLL composition issue,
though graph instances, degrees and target distances are not paired/equalized.
This establishes weak seen-example competence alongside held-out differences,
not a unique causal failure mechanism. No new scientific optimizer update or
OOD/GPU/learning-speed measurement occurs. Original readiness remains failed.

Separate invocation reopens all five checkpoints and exactly reproduces 1280
free generations/IDs/stops, teacher-forced scores, node errors and fixed flags;
model tensors and retained optimizer steps stay unchanged. This is same-code
neural replay, not independent implementation or fitting. The evidence audit
checks 22 byte-identical copies, independent decimal-vector/source/hop accuracy,
signed and first-node errors, token partitions/NLL, all 40 CSV cells/summary,
actual exposure, source/chronology identities and private-artifact exclusion.
Public-only generation/scoring reproduces exact probe bytes, independent labels
and all 1280 error scores without private weights/data/tokenizer cache/network.
It cannot reproduce neural or teacher-forced measurements without weights.
The plot was visually checked for readable labels and correctly scoped axes.

Stage 14 final distribution: both root and standalone suites pass 176 tests;
public-only standalone reproduction passes all 1280 errors. All four notebooks
regenerate; 335 local links, starter CRC/992 content hashes and 82 embedded source
identities pass. The prior stage 13 evidence audit remains valid. Mathematical
source/evidence inclusion and parity pass for 19 Lean, 12 theory and 16 paper
files without new compiler execution. Final direct sync/check follows the logged
checks, preserving the append-only log and all private/historical artifacts.

## Stage 15: completed small-cache overfit control, 31 October 2026

Eight new tests and all 184 root regressions pass. The nested 16-example cache
uses the original numerical adaptation procedure, all five original full step-0
models and fresh optimizers, fixed 600 updates/4800 presentations. Every arm
achieves 16/16 final cache vectors (4/4 per cell); the imported 512-cache reference
has 0/16 on those same examples. The fitting control passes. All development
results remain 0/128, with worse validity, node accuracy and NLL than the reference.
No cache-fitting success clears original synthetic readiness, full-ID competence
or OOD. There are 3000 new CPU task-adaptation updates; the reference 3000 are
imported. Actual token/padding differences are recorded, not an equal-FLOP claim.

Separate invocation reopens all 20 new checkpoints and exactly reproduces 960
cache/development predictions, teacher-forced scores, decisions/crossings, full
initial states, optimizer steps and actual order/token exposure; reference metrics
reconstruct from unchanged stage 14 evidence. This is same-code replay, not a
second fit or independent neural implementation. Fifty-two exact copies and all
140 CSV cells/summary pass independent decimal-vector/token/budget accounting,
source/config/start/reference identities, 300 presentations/example, freeze
chronology and publication privacy checks. Public-only data/label/order/reference
and 960 generated-score reproduction passes without private weights/data/cache/
network; neural and new teacher-forced reproduction require weights. Plot labels,
axes, target/crossing scope and zero held-out bars were visually inspected.

Stage 15 final distribution: root and standalone suites pass 184 tests each;
standalone public-only data/order/reference/score reproduction passes. All four
notebooks regenerate; 349 local links, starter CRC/1059 content hashes and 82
embedded source identities pass. Prior stage 14 evidence stays intact. Recursive
math source/evidence membership and parity pass for 19 Lean, 12 theory and 16
paper files without new compiler execution. Final direct sync/check follows all
logged checks, preserving append-only evidence and private-artifact exclusions.

Stage 16: all 193 root tests pass (eight new error-denominator tests and one
bounded exact-log-size policy test). The first pre-freeze invocation failed on
an invalid test-only weight-12 edge; its correction to weight8 is logged and
the failed record retained. No scientific source/config/endpoint changed after
freeze. Independent generated parsing and all-pairs hop computation reproduce
1440 imported outputs; 100 diagnostic cells and 50 paired rows/CSV counters,
first-error/teacher/role partitions, paired marginals and exact rendered review
pass. Teacher counters and prior neural replay remain imported evidence.
Zero new scientific model forwards, generations or updates; ordinary regression
fixtures are CPU execution checks, not new benchmark fits. Protocol regeneration
scope deviation is explicitly disclosed and original frozen wording preserved.

Stage 16 standalone validation also passes all 193 tests and public-only replay
of 1440 retained outputs without private weights/data, tokenizer cache or network.
Counter/CSV/review checks pass in both copies. Prior stage 15's 52 exact copies,
20/960 neural replay receipts, source/start/exposure/reference identities and
140 original cells remain valid; this check is not another neural replay.
All four notebooks regenerate. Documentation/starter checks pass 365 local links
and 1077 content hashes; standalone inventory verifies 1077 selected files and
82 embedded sources with pinned references/historical integrity. Recursive
source/evidence inclusion and root/Prod parity pass for 19 Lean, 12 theory and
16 paper files. No new Lean/LaTeX compiler or remote CI execution is claimed.
The bounded 10 MB exception preserves the now >5 MB append-only check history.
Regenerate the final starter after these notes, then sync/check directly after
all logged checks so no logger append creates distribution drift.

Stage 17: all 199 root tests pass, including six new nesting/reference-scope and
integer decision-boundary tests; the inherited numerical-update equivalence
checks pass before freeze. Five fixed CPU fits complete in 1,611.880 seconds.
Separate replay completes in 562.925 seconds and reproduces all 20 model/optimizer
checkpoints, 1,920 new cache/development generations/teacher scores, common16
subsets, starts, optimizer steps, actual order/LR/token counts and both reference
extractions exactly. This is same-code replay, not independent neural inference
or retraining. Scientific source/config/data/criteria remain unchanged after
freeze; ordinary regression fixtures are separate from the benchmark fits.

The evidence audit passes 72 byte-identical copies, 220 CSV rows/summary/review,
actual 75 presentations per example, both matched references, independent
generated vector/hop/sign/cell counts and frozen chronology/source/privacy.
Public data/labels/order/reference/generated-score reproduction passes without
private weights/data, tokenizer cache or network; teacher logits require weights.
The cache-fit curves, observed crossing line, all legends/axes, three-variant
held-out bars and overlap disclosure are visually checked. No new GPU, remote CI,
official language test or Lean/LaTeX compiler execution is claimed.

Stage 17 distribution checks: standalone also passes all 199 tests and public-only
data/order/reference/1,920-score reconstruction. Source/evidence inclusion and
root/Prod parity pass for 19 Lean, 12 theory and 16 paper files; saved verification
records are retained without a new compiler run. All four notebooks regenerate;
377 local Markdown links, starter CRC and all 1,164 inventoried content hashes
pass. No CI completion, push, remote/paid job or external publication is claimed.
Final starter regeneration and direct sync/check follow the final logger writes.

Stage18 (31 October–1 November 2026): all 208 root tests pass. Eight new tests
cover train-only marginals, source/format shortcuts, per-cell target requirements,
first/sustained crossings, right-censoring, step0 and component costs. The added
publication test bounds only the exact all-case result path at12MB, preserving
all other artifact exclusions/limits. Sixteen targeted new/inherited numerical
tests pass before freeze. Five matched fits complete in2561.106 seconds;
separate replay completes in970.423 seconds, verifying all35 checkpoints and4480
new generations/teacher scores, actual starts/optimizer/order/LR/token counts,
baseline/target/censoring/contrast and original cost identities. This is same-code
neural replay, not a second fit or independent neural implementation; timings
remain the original observations. No frozen scientific source/input/target changes.

All4480 saved answers independently parse/score with matching semantic/cell and
teacher-counter/cost aggregation. The first private preflight command fails only
on PowerShell quotation before executing its body; its corrected passing command
and the failed transcript remain logged. Public-only reconstruction passes ID-only
data/labels/order, train-only baseline, imported cost evidence, all4480 generated
scores,140 CSV rows and exact review without private weights/data/cache/network.
New neural logits require retained weights. The four-panel plot is inspected:
five legends, all seven points, gray baseline/black dotted target, both tasks/
sizes, full0..1 axes and coincident-curve disclosure are correct. No individual
semantic target is met at any observed cell; all arms stay right-censored.

Stage18 final distribution: standalone also passes all 208 tests and public-only
ID data/order/baseline/cost/4,480-score reproduction. All60 exact evidence copies,
140 cells,35/4480 replay receipts, actual token/optimizer identities and frozen
source/chronology/privacy pass the evidence audit. Stage17's72 exact copies,
20/1920 historical replay receipts,220 cells, references and fixed endpoints
remain valid. These preservation checks do not execute another neural replay.

All four notebooks regenerate. Starter CRC/1,239 content hashes and393 local
Markdown links pass; standalone inventory verifies1,239 files/82 embedded sources
and pinned-reference/historical integrity. Root/Prod recursive inclusion/parity
passes19 Lean,12 theory and16 paper source/evidence files. No new Lean/LaTeX,
remote CI, GPU, OOD or official-language-test execution is inferred. Regenerate
the final starter after these notes and finish direct sync/check AFTER all
logged commands, preserving append-only log parity and private-artifact rules.

Stage19 checkpoint-only diagnostic: all 213 root tests pass, including five new
selection/exposure/trace/decision tests; 13 targeted new/inherited tests pass
before freeze. Evaluate all five unchanged stage18 endpoints on128 seen cases
each; import the640 development outputs and exact-value teacher counters. The
evaluation command completes in265.287 seconds with zero optimizer updates.
Separate replay completes in213.673 seconds and exactly reproduces all five
endpoints/640 new predictions/teacher scores, fixed decisions and trace summaries.
This is same-code replay, not a new fit or independent neural implementation.

Public-only ID/probe/baseline/order reconstruction independently parses all1280
new/imported outputs and checks all3000 raw optimization rows,40 cell rows,
30 trace blocks, decisions/summary/review without private weights/data/cache/
network. The private evidence audit passes36 byte-identical copies, actual token
encodings, optimizer/count receipts and source/chronology/privacy. All five fail
semantic readiness on seen and held-out probes, with0/128 exact vectors on each.
The plot is visually inspected; palette counts confirm all four model/baseline
bar series, and all six fixed trace-block means/labels are retained. No new
PPT/language/GPU/OOD/official-test run or target/gate change is introduced.

Stage19 distribution: standalone also passes all213 tests and public-only probe/
baseline/1280-answer/3000-trace reconstruction. All four notebooks regenerate;
411 first-party local links, starter CRC/all1303 file hashes and standalone
1303-file/82-embedded-source integrity pass. Stage18's unchanged4480 output/
baseline/cost/censoring reconstruction passes without reexecuting neural replay.
Root/Prod recursive membership/parity retains19 Lean,12 theory and16 paper
source/evidence files. No new Lean/LaTeX compilation or remote CI is claimed.
The independently added scientific_transfer package and its own source-bound
evidence/receipts remain included under the existing publication selections;
its tests are not relabeled as newly run by this diagnostic. After final notes,
regenerate the starter and directly sync/check after the last logger append.
