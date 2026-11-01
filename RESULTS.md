# Results ledger

Updated 1 November 2026. This page summarizes reviewed evidence. Detailed
commands, failures and corrections are retained in [LOGBOOK.md](LOGBOOK.md)
and `logs/checks.jsonl`.


## Latest CPU diagnostic: seen/held-out fitting and trace audit stage 19

[Review](docs/results/fit_audit_v19_run1/REVIEW.md). All five unchanged stage18
update600 models miss the semantic target on an outcome-blind balanced128-case
SEEN training probe as well as the original held-out128 cases. Every arm has
0/128 exact answers on both probes. Seen whole-node accuracy is23.48–28.65%,
versus20.59–21.66% held out; source-excluded cell means and each probe's own
frozen-marginal advantages remain separately reported. This leaves seen-probe
fitting unresolved alongside the held-out gap, not generalization failure alone.
It does not identify a unique capacity, optimization or representation cause.

First50 update mean losses are3.038–3.320; last50 are0.834–0.863. Lower aggregate
answer/EOS loss coexists with weak seen digit prediction (34.0–37.7% under gold
earlier answers). All30 fixed optimization blocks and3000 actual update records
are retained; loss, clipping and LR summaries are descriptive, not evidence
that further updates necessarily repair semantic learning. Each selected case
has9 or10 actual presentations; all probe choices were frozen before prediction.

New work is640 seen-probe generations/teacher scores;640 held-out outputs are
imported exactly from stage18. Separate replay verifies all five endpoints and
640 new outputs/teacher scores, optimizer600/count receipts and fixed decisions.
All213 root tests pass. Zero new optimizer updates, language/PPT/GPU/OOD/official
test evaluations. No old target, checkpoint, budget or eligibility is changed.

## Previous CPU experiment: exploratory ID acceleration stage 18

The [new prospective protocol](docs/ACCEL_V18_PROTOCOL.md) freezes five-arm ID
learning curves and a training-only source-aware marginal baseline. All five
original complete language endpoints receive the SAME fresh ID512/128 data,
order, architecture, optimizer, schedule and 600 updates; evaluation is every
100 updates. The new semantic target is +10pp non-source accuracy over the
frozen baseline in every task/size cell, >=95% pooled validity and coverage1.
This is a new exploratory criterion, not retrospective success under the old
80% exact-vector calibration or synthetic-readiness/confirmation eligibility.

Fitting is complete: ALL five arms are right-censored at600, no individual
semantic cell target is met across140 observed cell points, and no finite
time-to-target ratio is estimable. Every final exact score is0/128; scratch has
one exact answer at update500 that does not persist to600. Final whole-node
accuracy is20.59% scratch,21.66% grid,20.78% graph,21.52% prefix-grid,20.61%
prefix-graph. All final vector-validity rates are100%. Keep all checkpoints,
partial curves, fixed targets and negative endpoints; censoring is not equality
or a proof that graph acceleration is impossible at all scales/budgets.

The common historical language stage has2000 updates/8192000 tokens per arm.
Each synthetic arm additionally has1000 PPT updates/4096000 tokens. Original
T4 generation/preparation/PPT/language/cumulative costs are hash-bound and
reported separately from new CPU times. Mixed-hardware execution sums are
component accounts, not controlled end-to-end throughput or cost recovery.
New fitting costs2561.106 seconds wall,1375.383 summed training seconds and
1095.985 evaluation seconds. Actual shared exposure:562823 input,96024 supervised
and696256 padded tokens per arm. New scientific task updates:3000; new PPT/
language/GPU/OOD/official-test work:zero. No earlier failure or gate changes.

All 208 root tests and independent parsing/counter/cost checks on 4,480 saved
outputs pass. Separate neural/teacher replay reproduces all 35 checkpoints and
4,480 predictions, targets/censoring/contrasts, optimizer/order/token identities
and historical costs exactly, in 970.423 seconds. Original timing observations
are retained; this is not retraining or independent neural implementation.
[Full review](docs/results/accel_v18_run1/REVIEW.md) and
[usage](docs/ACCEL_V18_USAGE.md) distinguish the verification layers.

## Previous CPU experiment: intermediate cache stage 17

[Review](docs/results/cache_v17_run1/REVIEW.md). The separately frozen64-example
cache uses the same five complete pre-adaptation states and600-update procedure.
Every model reaches64/64 exact cache answers (16/16 per cell), so the frozen
all-arm cache criterion passes. Every model still gets0/128 exact held-out answers
and fails the separate full-ID criterion. All observed pooled80% cache crossings
are at update300, without interpolation or a confirmatory speedup interpretation.

Three-way comparisons use the SAME common16 and development128 cases for all
variants. Imported512 additionally covers full64; imported16 has no measurements
on the extra48, which are absent rather than filled with zeros. New64 held-out
node accuracy is12.17–18.71%, validity59.38–97.66% and teacher-cell-mean answer/EOS
NLL2.0393–2.1876. NLL improves over imported16 in every arm; node changes versus16
are mixed. Imported512 retains better NLL AND node accuracy in every arm.
Graph node accuracy17.09% exceeds grid12.54%, but prefix-graph reaches18.71%.
This single-cohort descriptive endpoint cannot establish GraphNCA acceleration,
equivalence, algorithm generalization or cost recovery.

All20 checkpoints and1920 new generations/teacher scores replay exactly in a
separate invocation; no second fit is performed. New updates:3000 CPU task
adaptation. Reference updates:6000 imported. Each example is presented75 times;
actual per-arm tokens561300 input/96000 supervised/692880 padded. Costs, all
controls and all220 task/size cells remain retained. Root199 tests pass.
No new PPT/language/GPU/OOD or official-language-test evaluation occurs. Original
random-rule readiness, categorical bridge and all future transfer gates persist.

## Previous CPU analysis: retained-output error audit stage 16

[Review](docs/results/error_audit_v16_run1/REVIEW.md). The separately frozen audit
rescores 1440 IMPORTED endpoint outputs on identical cache16/development128 cases,
retaining five arms, small16/historical512 variants, four task/size cells and
pooled summaries. New model forwards, generations and updates are zero.

All small16 cache nodes and digit tokens remain correct. On held-out ID, valid-
vector node-micro accuracy is only 16.83–19.81%; teacher-forced digit token-micro
accuracy is 19.18–20.57%. Source/one-hop errors persist. Invalid outputs explain
part of the unconditional loss, but even parsed outputs and gold-prefix digit
prediction remain poor. This neither identifies a unique cause nor establishes
GraphNCA acceleration/no-effect. All development exact scores remain zero.

Independent generated parsing/hop recomputation agrees on all retained outputs;
100 diagnostic and 50 paired rows reproduce, including explicit zero-bin/null,
signed/first-error and valid/invalid denominators. Teacher counters are imported,
not independently neurally replayed. The inherited generator recreates unused
assessment/OOD rows in memory: this is disclosed as broader than the frozen
protocol's blanket no-OOD-data wording; none are scored or used for selection.
An intermediate64-cache study at fixed600 updates is proposed, not fitted.
Original readiness, separate categorical bridge and OOD gates remain intact.

## Previous CPU study: small-cache fitting control stage 15

All five original pre-adaptation states are fitted anew on a frozen nested
16-example cache using the unchanged stage 13 numerical procedure and 600-update
budget. Every final model gets 16/16 exact cache answers, including 4/4 in each
cell. The imported 512-cache reference scores 0/16 on the same examples.
There are 300 presentations/example, 4800 presentations/600 updates per arm;
actual new counts are 571200 input tokens, 96000 answer/EOS targets and 708488
padded positions, versus reference 562855 /96016 /691560. Diversity/content/
repetition and length changes are coupled consequences; no equal-FLOP claim.

Every new model still has 0/128 exact development answers. Development validity
is 51.56–66.41%, versus reference 98.44–100%; node accuracy is 8.61–12.34%, versus
18.95–21.60%; matched-cell answer/EOS NLL is 2.5892–2.8112 versus 0.9660–0.9983.
The fitting control passes while held-out metrics regress. This is one inspected
cohort/eight selected graphs and does not establish algorithm learning, graph
acceleration/no-effect or a unique cause. All 184 root tests pass. New scientific
updates: 3000 CPU task-adaptation updates; original reference 3000 are imported.
No new language/PPT/GPU/OOD measurement or official-language-test access occurs.
All 20 checkpoints and 960 new predictions/teacher scores replay exactly;
52 copied artifacts, 140 cells, independent budget/score accounting and public-
only data/order/reference/score reproduction pass.
Read [the stage 15 review](docs/results/overfit_v15_run1/REVIEW.md).

## Previous CPU diagnostic: matched seen/held-out error audit stage 14

All five saved final stage 13 models score 0/128 exact vectors on BOTH training
and development probes matched at 32 examples per task/size cell. Every selected
training example was previously used 9–10 times; development never entered the
optimizer. Training node accuracy ranges 29.55–32.89%, development 18.95–21.60%.
Teacher-forced digit-bearing token accuracy is 35.06–39.61% training and
24.36–28.09% development, versus other-token accuracy 100% /99.92–100%.
These matched-cell scores avoid the old eight-node-only NLL composition issue.

Every arm receives the training-probe-competence-not-established flag. Seen
examples already fail, with lower held-out scores too. This does not identify
optimization/capacity/data/representation/rollout as a unique cause. Match is by
task/size, not identical graphs, degree or distance. A quarter-cache sample is
not an exhaustive training census; this is one inspected historical cohort.
There are zero new scientific optimizer updates and no OOD/GPU/learning-speed
measurement. The proposed next study is a separately frozen small-cache overfit
control, changing cache size alone with all arms/settings/reference preserved.
All 176 root tests pass; all 1280 predictions and teacher-forced/error metrics
replay exactly. Twenty-two exact evidence copies, independent node/token error
accounting and 40 derived cells pass. Public-only probe/error reproduction passes.
Read [the review](docs/results/id_audit_v14_run1/REVIEW.md).

## Previous CPU diagnostic: matched ID adaptation stage 13

Five historical v4 cohort-59 final language models receive the same full-model
600-update answer-only supervised adaptation procedure, 512 training examples,
128 ID-development examples, actual batch order, fresh optimizer and LR schedule.
Each sees 562,855 unpadded inputs and 96,016 answer/EOS targets. All five final
models have 0/128 exact answers and zero correctness in every task/size cell;
valid-vector rates are 99.21875%, 100%, 100%, 98.4375%, 98.4375% for scratch, grid,
graph, prefix-grid, prefix-graph. All 80% ID crossings are censored, calibration
fails and OOD is skipped. Low teacher-forced NLL does not substitute for generated
answers. Preserve the original synthetic-readiness failure and negative graph
primary. See [the stage 13 review](docs/results/adaptation_v13_run1/REVIEW.md).

All 168 local tests pass. All 20 saved checkpoints reproduce 2,560 generations
and scores exactly in a separate invocation. Independent JSON correctness,
32 evidence copies and all 80 cells pass. Public-only data regeneration/scoring
also reproduces all outputs' scores without model weights or network.

This is 3,000 new local CPU task-adaptation updates, not new PPT/language training
or a confirmatory graph-acceleration test. Historical language stages remain
identical. The training NLL probe is eight-node-only while development mixes
sizes: their pooled difference is not a matched generalization-gap estimate.

## Previous CPU diagnostic: OOD graph evaluation stage 12

Implemented a separately frozen CLRS-Text-inspired graph evaluation layer.
Independent BFS/Dijkstra and Floyd-Warshall agree on 5,760 generated examples;
5,120 assessment views cover ID, size, family and joint shifts. Oracle exact
match and paired native/intervention correctness are 100%. The cached-native
control scores 75% pooled exact match but 0% paired native/intervention, exposing
why correct unchanged views alone are insufficient. Source-only also has 0%
paired correctness. These are measurement controls, not trained models.

The historical zero-shot calibration uses five paired final v4 language models,
cohort 59, on 32 prompts each. Scratch, grid, graph, prefix-grid and prefix-graph
ALL have 0/32 valid vectors and 0/32 exact answers. Context coverage is 100%.
All 160 generated token/text/stop records reproduce exactly from checkpoints.
No task-adaptation training was run, so this is a formatting/task-elicitation
floor, not a graph-transfer no-effect result or an interpretable OOD gap.

Fourteen new tests and the full local 160-test suite pass. Thirteen copied
evidence identities, independent accounting and frozen chronology pass. The
per-model CPU evaluation timers sum to 5.160180 seconds, including per-model
source/weight checks and loading, excluding preparation/controls/replay. These
are new CPU inference measurements, not newly measured training/GPU throughput.
No scientific optimizer updates, language/official-test access or GPU job occurs.

Read the [review](docs/results/ood_v12_run1/REVIEW.md),
[protocol](docs/OOD_V12_PROTOCOL.md) and [usage](docs/OOD_V12_USAGE.md).
The next learned test must first establish ID task competence with an identical
adaptation procedure, preserving the common language stage, all controls,
readiness gates and fresh confirmation. Local distance tasks are explicit
adaptations; official CLRS final references use a separate scoring adapter.

## Latest CPU study: fresh-phase confirmation stage 11

The weighted primary confirmation **passes** (3/3 paired seeds); standard passes 1/3. The broader both-objective criterion **fails**.
The weighted primary was declared prospectively from stage 10's development
finding; it does not replace that earlier failed both-objective primary.
All 24 endpoints were fitted anew, with three fresh paired initialization seeds
and one new shared split. Both objectives and coherent/shuffled controls remain.
Data excludes 4,926 historical joint phase classes; 128 native/changed assessment
pairs were reserved before selecting 64 development and 4096 training states.
The 512 cache is exactly nested; no historical endpoint is pooled into estimates.

| Objective | Cache | First NLL | First accuracy | Intervention NLL | Intervention accuracy | Full accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 512 | 1.359112 | 48.44% | 1.624712 | 40.10% | 92.06% |
| Weighted | 512 | 1.511389 | 48.44% | 1.678323 | 40.89% | 63.19% |
| Standard | 4096 | 1.306698 | 41.41% | 1.475631 | 28.65% | 91.45% |
| Weighted | 4096 | 1.105500 | 56.51% | 1.218692 | 47.66% | 72.72% |

These are final 1000-update endpoints at 188,000 input tokens and 168,000 targets
per model. Total new fitting: 24,000 updates / 4,512,000 input tokens /
4,032,000 targets contributing to final endpoints. The interrupted status write
adds 800 discarded updates /150,400 inputs /134,400 targets, so actual fitting
costs are 24,800 updates /4,662,400 inputs /4,166,400 targets. The last aborted
stage timer records 68.244014 seconds (a lower bound); preserve the failed
1519.694013-second command and successful 223.056-second recovery separately.
Recorded final-endpoint stage timers sum to 27.549
CPU minutes, excluding generation, assessment, replay and other checks; these
are not controlled throughput or time-to-target speedups. Actual larger-cache
exposure is 2561/2558/2564 distinct rows versus 512/512/512 smaller-cache rows,
from 4000 draws each. Retain actual masks and null empty-subset values.

All 54 full, 108 fit-probe and 27 intervention cases replay exactly from three
initial plus 24 final checkpoints. All 71 compact evidence copies, per-query
aggregates, paired decisions, common starts/budgets/sampling and frozen chronology
pass. The full local suite passes 146 tests, including nine stage 11 and three recovery checks.
The [review](docs/results/confirmation_v11_run1/REVIEW.md),
[summary](docs/results/confirmation_v11_run1/summary.json) and
[protocol](docs/CONFIRMATION_V11_PROTOCOL.md) retain every seed/control.
Stage 10's derived CSV has a documented
[column-collision caveat](docs/STAGE10_CSV_CAVEAT.md); its JSON/review remains
correct and its original evidence is preserved.

Declare a separate bridge toward interacting categorical rules, changing one task property at a time and retaining both objectives and controls.
Language remains gated. This fixed-rule confirmation does not establish
unseen-rule generalization, a dynamics-specific cause, GraphNCA acceleration or
cost recovery. No language/official-test evaluation, remote/paid run or
publication occurred. Earlier manuscript/presentation snapshots remain intact.

## CPU study: initial-condition diversity stage 10

Twelve new models completed 1000-update CPU training with a nested 4096-condition
cache, compared with twelve verified imported 512-condition controls. Rule,
objectives, architecture, actual initial weights, Adam/schedule and per-model
token budget remain fixed. All original trajectories/shuffles and held-out data
are exact; all 122 eligible interventions stay excluded from training.

The frozen study-wide primary **fails**: weighted loss passes all three paired
seeds, but standard loss passes only seed 167. Standard seeds 131 and 149 worsen
both first-patch and intervention NLL. No seed or objective is dropped.

| Objective | Cache | First-patch NLL | First accuracy | Intervention NLL | Intervention accuracy | Full accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 512 | 1.461689 | 47.40% | 1.608771 | 42.08% | 91.84% |
| Standard | 4096 | 1.711733 | 27.86% | 1.738204 | 25.96% | 90.33% |
| Weighted | 512 | 1.957809 | 32.03% | 2.006729 | 30.05% | 50.86% |
| Weighted | 4096 | 1.445994 | 45.57% | 1.443522 | 43.99% | 57.56% |

These are three-seed means on the inspected development assessment. The weighted
diversity gains are objective-specific, not proof of general acceleration. Its
larger-cache first/full accuracy still falls below the original smaller-cache
standard arm; that cross-objective comparison is descriptive. Larger-cache
shuffle models also pass trajectory readiness (3/3 standard, 2/3 weighted),
despite weak first-patch accuracy. Readiness alone remains insufficient.

Each model draws 4000 sequences. Actual distinct coverage is 2532/2533/2558
larger-cache rows versus 512/512/511 original rows. Cache availability is not
actual exposure; fixed anchor/extension probe masks distinguish drawn from
undrawn rows, and empty subset metrics are null. Both priors are train-only
frequency calibrations under the original policy.

All 54 full, 108 fit-probe and 27 intervention cases replay exactly from 27
initial/final files, including unchanged historical score vectors. Forty-two
compact evidence copies, nine per-seed raw files, source/input/sampling identities
and frozen chronology pass. The complete local suite passes 134 tests, including
two subsequent publication-policy integrity regressions.
New per-model CPU stage timers sum to 13.147 minutes; imported controls account
for 14.209 historical minutes separately. These are not throughput comparisons.
See the [review](docs/results/diversity_v10_run1/REVIEW.md),
[summary](docs/results/diversity_v10_run1/summary.json) and
[protocol](docs/DIVERSITY_V10_PROTOCOL.md).

Next: retain both strata and freeze fresh joint-phase-disjoint confirmation of
the objective-specific observation before broader claims. No additional study,
language stage, GPU/paid job or official-test evaluation has been launched.

## CPU diagnostic: split fit and generalization stage 9

No new model training occurred. All 15 existing initial/standard/weighted model
conditions were scored on coherent and shuffled train/development/assessment
data: 90 cells. The post-hoc diagnostic froze before new split scoring; its
assessment data were already inspected, so it is not confirmation.

| Coherent-model first-patch metric | Standard loss | Weighted loss |
| --- | ---: | ---: |
| Train NLL | 0.843632 | 0.810802 |
| Train accuracy | 75.39% | 83.46% |
| Development NLL | 1.315831 | 1.908040 |
| Development accuracy | 54.17% | 39.06% |
| Assessment NLL | 1.461689 | 1.957809 |
| Assessment accuracy | 47.40% | 32.03% |
| Assessment minus train NLL | 0.618058 | 1.147007 |

Three-seed means conceal a distinction: seeds 149 and 167 improve training
first-patch NLL but worsen on both held-out splits. Seed 131 worsens on training
data too. All three improve train AND assessment NLL/accuracy when averaging
the four teacher-forced patches of the whole first transition. Mean assessment
four-patch accuracy improves 42.90% to 46.48% despite first-patch regression.
This rules out describing the outcome as a uniform failure to fit every weighted
transition target; it does not identify the causal mechanism.

All 90 cells replay exactly after reopening the 15 checkpoints. All 15 coherent
assessment first-patch score vectors agree exactly with stage 8. Short/long
causal-prefix NLL differences are at most 4.76837158203125e-7 nats with identical
correctness. Seven copied evidence identities, nested aggregates and the frozen
chronology pass. The complete local suite passes 122 tests, including two
starter-snapshot integrity regressions. See the
[review](docs/results/fit_v9_run1/REVIEW.md), [summary](docs/results/fit_v9_run1/summary.json)
and [protocol](docs/FIT_V9_PROTOCOL.md).

Next candidate: a separately declared initial-condition diversity ablation,
preserving objectives, architecture, rule and update/token budget with all
controls. No such new treatment, GPU/paid job or language stage was launched.

## CPU diagnostic: first-transition weighting stage 8

The separately frozen sixfold first-transition objective ablation fails its
primary comparison: **0/3 coherent seeds** improve original AND intervention
NLL. Each worsens both metrics despite passing full-trajectory readiness.
Six new weighted CPU models each received 1000 updates / 188,000 input tokens,
matched to six imported verified stage 7 spatial controls. Data, actual starting
weights, batch order, rule, architecture and schedule remain identical.

| Mean coherent-model metric | Standard loss | Sixfold first-transition weight |
| --- | ---: | ---: |
| Original first-patch NLL | 1.461689 | 1.957809 |
| Original first-patch accuracy | 47.40% | 32.03% |
| Neighbor-intervention NLL | 1.608771 | 2.006729 |
| Neighbor-intervention accuracy | 42.08% | 30.05% |
| Both predictions correct | 19.95% | 11.20% |
| Full-trajectory patch NLL | 0.303136 | 1.347235 |
| Full-trajectory patch accuracy | 91.84% | 50.86% |

Lower NLL is better. Both shuffled controls remain; weighting yields no
consistent shuffled-model NLL gain. The coherent-minus-shuffled weighting gain
is negative for every seed. These controls are descriptive, not revised gates.
Recorded new CPU stage timers total 7.481 minutes; imported control timers total
6.729 historical minutes. These are not controlled throughput measurements.
The new command completes in 455.220 seconds, including initial verification.

All 30 full-trajectory and 15 original/intervention checkpoint cases replay
exactly. Source/input/budget/order identities and freeze-before-training-before-
assessment chronology pass. The local suite passes 112 tests. See the
[review](docs/results/loss_weight_v8_run1/REVIEW.md),
[summary](docs/results/loss_weight_v8_run1/summary.json) and
[protocol](docs/LOSS_WEIGHT_V8_PROTOCOL.md). This is inspected development data;
no weight/budget retuning, language stage or GPU run followed. The next proposed
diagnostic compares existing checkpoints' first-transition errors across train,
development and assessment before choosing another treatment.

## CPU diagnostic: matched spatial bridge stage 7

Twelve models completed fixed 1000-update CPU training. Complement and
neighbor-parity treatments share initial grids, full starting weights, batch
orders and all v6 model/representation/training settings. A new common split
excludes joint phase overlap under either rule; historical v6 is separate evidence.

| Rule | Mean patch NLL | Patch accuracy | Causal frequency NLL | Previous-own-patch table NLL |
| --- | ---: | ---: | ---: | ---: |
| Complement reference | 0.055023 | 100.00% | 2.352583 | 0.016619 |
| Spatial neighbor parity | 0.303136 | 91.84% | 2.106976 | 2.814026 |

All three spatial models pass both fixed readiness margins and the separately
declared own-patch-table qualification. Both rules repeat after two frames;
later-frame recall could beat that table without first-transition competence.
A supplementary diagnostic, frozen during training before assessment, therefore
scores the first target patch with no repeated frame available. All three
spatial models beat its table/uniform NLL comparators, but mean accuracy is
**47.40% original / 42.08% after a neighbor-bit intervention**, with both
predictions correct in only **19.95%** of the 122 eligible pairs. This is partial
known-rule learning. Spatial shuffle-trained controls pass the small supplementary
NLL margin too, with much weaker accuracy; admission alone is not a causal test.

All 36 primary cases and 18 supplementary checkpoint cases replay exactly.
The full local suite passes 104 tests. Recorded CPU stage time totals 13.464
minutes for 12,000 updates / 2,256,000 input tokens; these are not GPU speedups.
The next proposed one-factor study reweights the first transition while holding
rule/model/data/token budget fixed, using these inspected cases as development
data and reserving fresh data for confirmation. Language training stays disabled.
See the [stage 7 review](docs/results/interaction_v7_run1/REVIEW.md) and
[frozen main protocol](docs/INTERACTION_V7_PROTOCOL.md).

## Earlier CPU diagnostic: controlled trainability stage 6

The full v4 backbone learned the simplified binary complement rule. Three paired
seeds each completed coherent and prefix-shuffled training for 1000 updates.
All three coherent-trained final checkpoints pass both frozen 0.01-nat margins
on 128 phase-disjoint held-out initial conditions, with 100% patch accuracy.

| Seed | Coherent patch NLL | Same checkpoint on matched shuffle | Frequency advantage | Ordering advantage |
| --- | ---: | ---: | ---: | ---: |
| 131 | 0.052766 | 4.594601 | 2.252567 | 4.541834 |
| 149 | 0.053680 | 4.590817 | 2.251654 | 4.537137 |
| 167 | 0.062805 | 4.382613 | 2.242528 | 4.319808 |

The causal frequency baseline scores 2.305334 nats; mean model NLL is 0.056417.
All 18 initial/final assessment cells replay exactly from saved checkpoints.
The six CPU models total 6000 updates / 1,128,000 input tokens and 6.966 minutes
of recorded stage time. This is basic trainability on one non-interacting rule
with a smaller alphabet/grid/context, not random-rule generalization or language
acceleration. All three shuffle-trained checkpoints also pass the absolute
margins with much smaller ordering gaps and about 27% coherent accuracy; those
checks alone do not identify a causal training mechanism.

The next declared bridge should change only the transition to an interacting
binary rule. Language training remains disabled, and the original random-rule
readiness failures below remain intact. Read the
[full stage 6 review](docs/results/trainability_v6_run1/REVIEW.md) and
[frozen protocol](docs/TRAINABILITY_V6_PROTOCOL.md). V4 remains the latest GPU run.

## Earlier CPU diagnostic: checkpoint readiness stage 5

All 48 planned coherent/matched-prefix-shuffle cells completed using 24 saved
v4 PPT checkpoints, with no neural training or language evaluation. All 12
structured cohort decisions fail both frozen 0.01-nat readiness requirements.
The actual language-start guard rejects these checkpoints.

| Mean advantage, nats per patch | Grid | Graph |
| --- | ---: | ---: |
| Over causal frequency prediction | -0.587546 | -0.297951 |
| From coherent ordering | 0.000849 | 0.000987 |
| Ordering gap beyond prefix-trained control | 0.000092 | 0.000102 |

The maximum individual ordering advantage is 0.004817, below the requirement.
Small gaps have mixed signs and do not establish zero order dependence. The
known-rule measurement control passes, but uses a learned transition table and
does not establish transformer trainability. All 24 native cells replay within
1.6156e-7 NLL. This is development work on inspected rules; prior frequency
failures were already known. The next objective is a declared controlled
transformer-learning diagnostic before further GPU language training.
See the [stage 5 review](docs/results/readiness_v5_run1/REVIEW.md) and
[frozen protocol](docs/READINESS_V5_PROTOCOL.md). V4 remains the latest GPU result.

## Latest GPU measurement: prefix-control v4 development run 1

The returned seven-arm Kaggle T4 suite completed six fresh paired cohorts with
the frozen independent-stage budgets. All 78 selected stage checkpoints and six
actual initializations passed independent CPU replay; maximum metric discrepancy
was 2.7828e-7. All 454 archive members passed streaming CRC/path checks.

Both co-primary graph-advantage tests fail at the same paired scratch-endpoint
target: graph/grid has five token ties and one loss (ratio 0.990519x, p=1);
graph/prefix-graph has four ties, one win and one loss (1.000000x, p=0.984375).
No seed, target or gate is changed after inspection. This does not establish
equivalence or rule out smaller effects hidden by the 100-update cadence.

| Initialization | Mean final validation NLL | Language-token speedup versus scratch | Cumulative-time speedup versus scratch |
| --- | ---: | ---: | ---: |
| Scratch | 5.273956 | 1.000000x | 1.000000x |
| Grid | 5.230608 | 1.165316x | 0.864328x |
| Graph | 5.234337 | 1.154268x | 0.857989x |
| Original shuffled grid | 5.236298 | 1.143913x | 0.851626x |
| Original shuffled graph | 5.234172 | 1.165990x | 0.864376x |
| Prefix-matched shuffled grid | 5.235063 | 1.166591x | 0.864891x |
| Prefix-matched shuffled graph | 5.235680 | 1.154268x | 0.858035x |

Graph uses about 13.4% fewer language tokens at the scratch target but 16.6%
more recorded cumulative time. Shuffled data transfers similarly. A new post-hoc
causal frequency predictor also beats every structured PPT model on held-out
patch prediction; the declared global-unigram/copy gate was insufficient to
establish dynamics-specific competence. The immediate recommendation is an
order-sensitivity cross-evaluation using existing checkpoints before another
GPU language run. The official test remains untouched.

Read [the full v4 audit and next steps](docs/results/transfer_v4_run1/REVIEW.md),
[frozen primary analysis](docs/results/transfer_v4_run1/primary_analysis.json),
[new CPU diagnostic](docs/results/transfer_v4_run1/frequency_diagnostic.json)
and [copy provenance](docs/results/transfer_v4_run1/PROVENANCE.json).
These new cohorts are not pooled with earlier v2/v3 evidence.

## Earlier measurement: graph-transfer v3 development run 1

The user-run Kaggle T4 suite completed all five treatments across three paired
model/data/topology replicates. Every language stage received 2000 updates;
each synthetic treatment first received 1000 PPT updates. Graph and grid share
frozen rules admitted by the same joint complexity policy.

The **primary** target is each paired grid model's final language NLL. Graph
first reaches it at 1800 updates for seed 17, at 2000 for seed 29, and does not
reach it by 2000 for seed 43. The all-seed primary speedup is correctly withheld.
Mean final graph-minus-grid NLL is -0.005550 nats/token (about 0.55% perplexity
improvement). Graph-minus-shuffled-graph is -0.002627, with mixed paired signs.
These are small exploratory differences, not reliable graph/dynamics acceleration.

Additional comparisons to the paired **scratch** endpoint:

| Initialization | Mean final validation NLL | Geometric language-token speedup versus scratch | Geometric cumulative-time speedup versus scratch |
| --- | ---: | ---: | ---: |
| Scratch / no PPT | 5.266091 | 1.000x | 1.000x |
| Grid NCA | 5.237811 | 1.112x | 0.838x |
| Graph NCA | 5.232262 | 1.178x | 0.872x |
| Shuffled grid | 5.234889 | 1.112x | 0.835x |
| Shuffled graph | 5.234889 | 1.134x | 0.846x |

Synthetic initializations improve learning relative to no PPT in this setting,
but the extra cost is not recovered. Graph uses 10–20% fewer language updates
than scratch, yet about 14.7% more recorded cumulative time at that target.
The primary graph-versus-grid outcome above must not be replaced by the favorable
scratch comparison. The official test remains uninspected.

Read the [full v3 audit](docs/results/transfer_v3_run1/REVIEW.md),
[raw GPU report](docs/results/transfer_v3_run1/gpu/report.json) and
[derived aggregates](docs/results/transfer_v3_run1/aggregate.json).
CPU replay and imported GPU measurements are recorded separately. Original ZIP,
datasets, weights and GPU verification remain intact.
All 27 selected stage checkpoints and three saved initializations passed
independent CPU checks; maximum metric discrepancy was 1.8552e-7.

## Earlier measurement: transfer-v2 GPU development run 1

The returned Kaggle Tesla T4 experiment used three paired model seeds and
identical language stages of 2000 updates / 8,192,000 input tokens per arm.
Grid and shuffled arms first received 1000 synthetic updates. These are
imported GPU measurements, followed by independent local CPU checks.

| Initialization | Mean final validation NLL | Geometric language-token speedup | Geometric cumulative-time speedup |
| --- | ---: | ---: | ---: |
| Scratch / no PPT | 5.266091 | 1.000× | 1.000× |
| Grid NCA | 5.233321 | 1.178× | 0.886× |
| Shuffled NCA | 5.231528 | 1.178× | 0.883× |

The reference is each seed's scratch final validation NLL. Grid first reached
it at 1800, 1600 and 1700 language updates; shuffled reached it at 1700, 1600
and 1800. Crossings are observed every 100 updates, without interpolation.
Values above are rounded from [aggregate.json](docs/results/transfer_v2_run1/aggregate.json).

The result supports a modest synthetic-initialization transfer signal in this
development setting. Similar shuffled gains leave the contribution of NCA
dynamics unresolved. Grid's cumulative time is about 12.8% higher at the paired
reference, so cost recovery has not been demonstrated. No v2 graph treatment
or official-test evaluation was performed. Three seeds on one shared synthetic
cache and a small validation prefix do not establish a general speedup.

All 15 saved development checkpoints passed the earlier independent CPU replay,
with maximum metric discrepancy 1.9563e-7. Cross-runtime bitwise regeneration
of initialization/head/schedule bytes remains unverified; the failed attempts
are retained. The earlier v2 publication refresh checked saved evidence hashes;
it did not repeat training or checkpoint evaluation. The new v3 audit above is
separate from those v2 records.

Read the [full reviewed result](docs/results/transfer_v2_run1/REVIEW.md),
[CPU verification](docs/results/transfer_v2_run1/cpu_verification.json),
[stage-control audit](docs/results/transfer_v2_run1/stage_control_audit.json)
and [copy provenance](docs/results/transfer_v2_run1/PROVENANCE.json).
Original archives, datasets and weights remain preserved locally.

## Mathematical verification

The [LaTeX note](docs/theory/README.md) develops the mathematical principles,
conditional transfer arguments and worked numerical examples. The separate
[Lean folder](proofs/lean/README.md) contains 44 theorem statements and supporting
lemmas checked in the official Lean 4.35.0 web editor. Their exact source hashes,
allowed foundational axioms and failed first full check are retained in its
[verification record](proofs/lean/verification/README.md).

Formalization is partial; see the [coverage map](proofs/lean/PROOF_MAP.md).
Entropy, spectral theory and the calculus supplying Taylor remainder estimates
remain informal. No local modular Lake build or remote proof CI run is claimed.
The exact numerical toy results are mathematical examples, separate from the
GPU language-transfer measurements above.

## Historical results

- [Historical pilot 1](docs/results/pilot1/REVIEW.md): matched total budget with
  unequal language exposure; it did not demonstrate a graph benefit and is not
  relabeled as the corrected transfer experiment.
- [Local validation record](VALIDATION.md): historical CPU diagnostic, v2
  software/data-path checks and the later returned GPU audit, with their scopes
  kept explicit.

## Publication copies

Root files are canonical in the research workspace. `Prod` is regenerated from
the explicit publication selection and checked for SHA-256 parity. Standalone
publications contain this same results ledger and compact evidence. Raw corpora,
weights, dependency caches and unpublished experiment development are excluded.
