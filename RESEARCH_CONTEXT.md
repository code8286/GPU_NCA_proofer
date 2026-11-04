# Research context: GNNs in NCA pre-pre-training

Updated: 1 November 2026.

The research question is whether **graph-NCA data improves the transformer
initialization acquired during pre-pre-training**, before ordinary language
pre-training. Grid NCA and graph NCA are synthetic sources for that first stage.
Neither is a replacement for the subsequent language-model training stage.

Primary references to retain:

- Paper, version 1: https://arxiv.org/html/2603.10055v1
- Author repository: https://github.com/danihyunlee/nca-pre-pretraining/tree/main
- Pinned source revision inspected: `bdd1c71e04eb2e5a2c91481faf278e95846b4431`
  (12 March 2026).
- Local source snapshot and checksums:
  `sources/nca-pre-pretraining-reference/reference.json`, with files beneath
  `sources/nca-pre-pretraining-reference/bdd1c71e04eb2e5a2c91481faf278e95846b4431/`.

The correct primary comparison is:

| Initialization entering language pre-training | Synthetic pre-pre-training | Language pre-training |
| --- | --- | --- |
| Random / no pre-pre-training control | None | Same N text updates |
| Grid-NCA initialized transformer | K grid-NCA updates | Same N text updates |
| Graph-NCA initialized transformer | K graph-NCA updates | Same N text updates |
| Shuffled-NCA initialized transformer | K shuffled-data updates | Same N text updates |

Use identical language data order, optimizer initialization, learning-rate
schedule, batch/context, evaluation and stage length. Compare graph vs grid as
pre-pre-training treatments. A text-only model remains a valid **no
pre-pre-training control**, as in the paper; it must not get an unacknowledged
head start on the common language-stage clock.

Report transfer efficiency on language-stage tokens/time first. Separately
report cumulative synthetic + language cost and generation cost. The paper also
reports a combined token-cost metric in Appendix D; these questions must remain
distinct rather than removing cost accounting altogether.

Pilot 1 is preserved locally in `Kaggle Outputs/gnn_nca_results_1.zip`; compact
public evidence is in `docs/results/pilot1/`. All 15 runs and
checkpoints passed independent verification. Its design was a **matched total
budget replacement pilot**: scratch received 1000 text updates; synthetic arms
received 100 synthetic + 900 text. Its negative result is valid for that design
but is not a clean replication of the paper's transfer experiment. Its scratch
scheduler also restarted after 100 text updates, unlike the synthetic arms'
language-stage schedule.

See `METHODOLOGY_AUDIT.md` for the specific mismatches. Existing aligned logs show
no sustained graph benefit, so correcting the framing must not be described as
discovering a positive result. At 900 target-text updates, mean validation NLLs
are 1.7542 (scratch), 1.8002 (grid), and 1.8170 (graph), with the scratch schedule
still confounded. No matched-text final scratch test checkpoint was saved.

Do not overwrite original run files or silently relabel the old experiment.
Establish useful held-out-rule NCA learning, reproduce grid-NCA transfer at an
affordable scale, then introduce the graph modification. Avoid interpreting a
one-time generation timing difference as an optimized graph-kernel speedup.

## Implemented next run

The independent-stage implementation is now `transfer_v2/`, configured by
`configs/transfer_v2.json`, with notebook `kaggle_nca_transfer_v2.ipynb`.
See `docs/TRANSFER_V2_PROTOCOL.md`. It restores categorical NCA/rule filtering/
burn-in and uses a small Llama-style model, independent 1000-update PPT and
2000-update language budgets, grid/shuffle/no-PPT controls, stage-boundary
checkpoints and a held-out-rule prediction gate. Language heads, data order,
optimizers and schedules match across arms. The language confirmation split is
reserved by default. The returned v2 GPU development run is now measured; see
[its results review](docs/results/transfer_v2_run1/REVIEW.md). All three grid seeds
passed the synthetic gate and reached their paired scratch endpoint after
1600–1800 language updates versus 2000, with a geometric language-token speedup
of 1.178x. Shuffled NCA achieved the same aggregate token speedup. Mean final
validation NLLs were 5.266091 (scratch), 5.233321 (grid), and 5.231528 (shuffled).
This supports a small synthetic-initialization transfer effect in this setup;
it does not isolate local dynamics or establish a graph benefit. The recorded
cumulative cost of grid NCA was about 13% higher at the reference loss.
The official test remains unused. Graph NCA has not been run under v2.

## Graph transfer v3: measured development result

`transfer_v3/`, `configs/transfer_v3.json` and
`kaggle_nca_transfer_v3.ipynb` implement the next experiment; see
[the v3 protocol](docs/TRANSFER_V3_PROTOCOL.md). It compares no-PPT, grid, graph,
shuffled grid and shuffled graph before the same 2000-update language stage.
All synthetic arms receive 1000 PPT updates. Each paired seed gets an independent
rule cache/topology. Grid and graph share frozen rule weights and paired random
draws; selection requires both profiles to pass the same complexity band.
The graph changes donor connectivity while preserving degrees/ordered slots.

The primary comparison is graph-PPT versus grid-PPT; graph versus its shuffled
control is needed to assess ordered-trajectory structure. Comparisons to no-PPT
and cumulative cost remain additional results. Actual initial core/head weights
are saved, so cross-platform checks can verify those tensors directly. Both grid
and graph must pass the synthetic gate before any language run.

The returned v3 T4 run completed; see [its review](docs/results/transfer_v3_run1/REVIEW.md).
Graph's mean final NLL is 5.232262 versus grid 5.237811 and shuffled graph
5.234889 (scratch 5.266091). Against each paired grid endpoint, graph reaches
at 1800 / 2000 / not reached by 2000 for seeds 17 / 29 / 43. The primary
all-seed speedup is therefore censored. The small average gain does not yet
establish consistent topology-specific acceleration; shuffled controls remain
strong. Against the common scratch target, graph's token speedup is 1.178x,
but its cumulative time is about 14.7% higher. Preserve v2 separately: joint
rule eligibility changes the v3 grid cohort. Official-test confirmation remains
unused. Do not drop replicate 43 or redefine the primary target after inspection.

## V4 prefix controls and fresh replication: measured development result

The separate [v4 protocol](docs/TRANSFER_V4_PROTOCOL.md) tests graph PPT against
grid PPT and an exact-prefix shuffled graph control at the same scratch-endpoint
language-loss target. Six fresh paired cohorts retain both original shuffles,
with two additional controls preserving initial conditioning, marker formatting
and each trajectory's supervised-prefix token frequencies. Numerical model,
generator, filter, topology, serializer and budgets retain v3 defaults. V3's
primary result above is not redefined or pooled with the new cohorts.

The prospective plan fixes censoring and two co-primary sign tests before new
training. This is a validation-stage replication/mechanism study, not untouched
confirmation. The [returned v4 review](docs/results/transfer_v4_run1/REVIEW.md)
reports complete GPU execution and independent CPU replay of all 78 selected
stage checkpoints and six saved starting states, maximum metric error 2.7828e-7.
Both frozen primary graph comparisons fail: graph/grid has five token ties and
one loss, graph/prefix-graph has four ties, one win and one loss. Their geometric
token ratios are 0.990519x and 1.000000x; sign p-values are 1 and 0.984375.
Graph's mean final language NLL is 5.234337 versus grid 5.230608, prefix-graph
5.235680 and scratch 5.273956. Graph reaches the scratch target with about 13.4%
fewer language tokens, but about 16.6% more recorded cumulative time.

A new post-hoc CPU causal frequency predictor beats all 12 structured PPT
models on the same held-out patch targets (grid mean 7.572958 versus 8.160504;
graph 7.804035 versus 8.101986). This does not prove the models learned no
dynamics, but shows the original unigram/copy gate does not establish such
learning. Next, cross-evaluate existing PPT checkpoints on coherent/matched
shuffled trajectories, then strengthen readiness before another language suite.
Any serializer/topology or PPT-duration change needs a new declared development
protocol. Preserve all v4 outcomes and the reserved official test.

## Stage 5: completed checkpoint-only readiness diagnostic

The [new review](docs/results/readiness_v5_run1/REVIEW.md) reports all 48 frozen
CPU cross-evaluation cells from 24 saved v4 PPT checkpoints. None of 12 grid/graph
cohorts passes either the frequency or ordering requirement (each >=0.01 nats
per patch). Mean ordering advantages are 0.000849 grid and 0.000987 graph;
prefix-trained checkpoints show similar gaps. The known-rule table-based
measurement control passes, and the real all-cohort language-start guard rejects
the retained checkpoints. These are inspected development data, not confirmation
or new training. V1-v4 protocols, weights and measurements remain unchanged.

The next development objective is transformer trainability on a controlled
transition task, followed by one-factor representation/capacity diagnostics if
needed. The table control validates scoring, not transformer competence.
Freeze a new training protocol before running that diagnostic. Pause full
language suites until the selected synthetic configuration clears readiness;
any graph serializer/topology and cost-budget ablations remain separate studies.

## Stage 6: controlled transformer trainability passed

The [controlled-rule review](docs/results/trainability_v6_run1/REVIEW.md) reports
actual CPU transformer training: three paired seeds, coherent and prefix-shuffled
arms, fixed 1000-update endpoints and 128 phase-disjoint held-out initial
conditions. The full six-layer/192-wide backbone and v4 numerical training
pipeline are preserved; the binary alphabet, 4x4 grid and 47-token context
deliberately simplify the task. All three coherent-trained final models achieve
100% patch accuracy and pass both inherited 0.01-nat readiness advantages.
Mean coherent NLL is 0.056417 versus causal frequency 2.305334; ordering advantage
averages 4.466260 nats. All 18 initial/final assessment cells replay exactly.

This establishes a limited capability on a fixed self-local complement rule.
It does not isolate the cause of v4's random-rule failure. Shuffle-trained models
also pass the absolute margins with much weaker accuracy/order gaps, reinforcing
that admission is not a causal mechanism test. The next bridge is a separately
declared interacting binary rule with other settings fixed. Original random-rule
checkpoints retain failed stage 5 readiness; language training remains disabled.
There is no new GraphNCA, GPU or language-transfer result in this stage.

## Stage 7: matched spatial bridge and first-transition limit

The [new review](docs/results/interaction_v7_run1/REVIEW.md) reports twelve
fixed-budget CPU models comparing complement and a left/right neighbor-parity
term. All v6 alphabet/grid/context/serializer/model/training settings are locked.
Both rules use one new jointly phase-disjoint split and a newly trained reference;
historical v6 is not the treatment control. All three spatial seeds pass the
unchanged frequency/ordering readiness checks and a separately predeclared
comparison to a previous-own-patch table. Mean spatial patch NLL is 0.303136
versus frequency 2.106976 and local table 2.814026; accuracy is 91.84%.

Period-two recall can inflate later-transition performance. A separate diagnostic
was frozen during training before assessment, without changing the primary plan.
It measures only the first target patch and a neighbor-bit intervention outside
that patch. Its NLL criteria pass in all three seeds, but original/intervened
accuracy averages only 47.40% / 42.08%; both targets are correct in 19.95% of
eligible pairs. Shuffle-trained controls also pass its absolute NLL margins with
weaker performance. Do not equate admission with reliable spatial-rule execution
or coherent-training causality. All 36 main and 18 supplementary cases replay.

The next proposal is a separately frozen first-transition loss-weighting study,
holding the known spatial rule, model, data and token/update budget fixed and
retaining standard-loss/shuffled controls. Current inspected initial conditions
are development evidence; future confirmation needs fresh joint-phase-disjoint
data. No such new ablation or language study has been run. This does not change
original random-rule readiness or establish GraphNCA/GPU/language acceleration.

## Stage 8 — fixed-budget first-transition weighting (20 October 2026)

The separately frozen sixfold loss-weighting ablation is now complete. It
compares six new weighted coherent/shuffled models with the six verified
historical stage 7 spatial standard-loss checkpoints, preserving the exact
data, full initial states, batch order, model, optimizer/schedule and 1000-update
budget. The first transition contributes half the normalized training objective.
This reweights the entire first transition, including markers; patch/marker
weight proportions remain unchanged. All evaluation likelihoods are unweighted.

The primary fails for all three coherent seeds: original and neighbor-intervened
first-patch NLL both worsen. Mean original NLL rises 1.461689 to 1.957809;
intervention NLL rises 1.608771 to 2.006729. Mean accuracies fall 47.40% to 32.03%
and 42.08% to 30.05%, respectively; full-trajectory accuracy falls 91.84% to
50.86%. Both full-trajectory readiness margins still pass. This is further
evidence that these coarse trajectory gates cannot establish first-transition
competence, even on one known binary interacting rule.

All 30 full and 15 original/intervention checkpoint cases replay exactly;
standard-control scores match historical values. The fixed weight/endpoint is
retained. The complete local suite passes 112 tests. New CPU training stage
timers total 7.481 minutes; historical controls remain separate costs. See the
[stage 8 review](docs/results/loss_weight_v8_run1/REVIEW.md).
No topology, serializer, budget or language-stage change followed the failure.

The next proposed development task is an existing-checkpoint first-transition
fit/generalization diagnostic across train, development and assessment, declared
before computing those additional scores. It has not been executed here. The
negative weighting outcome alone does not identify gradient conflict, capacity,
serialization or another mechanism as its cause. Original random-rule readiness
remains failed. Any future transfer still requires a frozen eligible synthetic
configuration, fresh paired cohorts and an identical full language stage.

## Stage 9 — existing-checkpoint fit versus generalization (20 October 2026)

The proposed stage 8 follow-up is complete without any new training. A separate
post-hoc development protocol froze before scoring new train/development first-
transition errors, retaining all 15 initial/standard/weighted model conditions,
both data versions and all three existing splits. All 90 cells replay exactly;
the coherent assessment vectors reproduce stage 8. No held-out split is newly
untouched confirmation data.

Two coherent seeds (149,167) improve training first-patch NLL with weighting
but worsen development and assessment; seed 131 worsens training as well.
Mean first-patch train accuracy rises 75.39% -> 83.46% while assessment falls
47.40% -> 32.03%. Mean assessment-minus-train first NLL grows 0.618058 ->
1.147007. Both imperfect fitting and generalization differences remain; a single
pooled explanation is unwarranted.

All four first-transition patches tell a different story: every coherent seed
improves train and assessment patch NLL/accuracy with weighting. Assessment
four-patch accuracy rises 42.90% -> 46.48%, even as first-patch accuracy worsens.
Later patch predictions condition on earlier true patches. Aggregate transition
learning is not equivalent to correct initial spatial prediction or rollout.
Marker NLL is separately retained. This does not establish gradient interference,
memorization, capacity or a specific circuit as the unique cause.

The [stage 9 review](docs/results/fit_v9_run1/REVIEW.md) retains every control,
per-query score, mixed descriptive category, source identity and replay record.
The local suite passes 122 tests, including two starter-snapshot integrity checks.
The proposed next single-variable treatment is
training initial-condition diversity with objectives, architecture, rule and
update/token budget fixed, retaining standard/weighted and shuffled controls.
It needs a new frozen declaration; it has not been configured or fitted here.
The failed stage 8 endpoint and original random-rule readiness remain unchanged.
Language training remains gated; later confirmation and transfer still need
fresh paired data/cohorts and an identical full language stage.

## Stage 10 — initial-condition diversity (21 October 2026)

The separately frozen diversity ablation is complete: 4096 nested available
training conditions versus 512, with the known spatial rule, objectives, full
backbone, actual initial states and per-model 1000-update/token budget fixed.
Twelve new CPU models retain both standard/weighted objectives and coherent/
shuffled controls against twelve imported endpoints. All original rows and
shuffles are exact. New phases exclude held-out and all eligible intervention
classes; the old 122-pair mask is unchanged. This is inspected development data.

The predeclared study-wide primary FAILS. All three weighted seeds improve
original AND intervention NLL by >=0.01 while retaining trajectory readiness;
standard loss improves only seed 167, with both first/intervention NLL worse for
131 and 149. Weighted first accuracy rises 32.03% -> 45.57%, intervention 30.05%
-> 43.99%, full trajectory 50.86% -> 57.56%. Standard first accuracy falls
47.40% -> 27.86%, intervention 42.08% -> 25.96%. Do not remove the failing
objective or call this a general diversity/learning-speed success.

The weighted larger-cache arm still has lower first/full accuracy than the
original standard smaller-cache arm (descriptive cross-objective comparison).
Larger-cache shuffle controls pass readiness in 3/3 standard and 2/3 weighted
seeds despite weak first-patch performance. These coarse gates alone cannot
establish dynamics-specific competence or a unique mechanism.

Actual distinct rows drawn are 2532/2533/2558 from the 4096 cache versus
512/512/511 originally, each from 4000 draws. All four new arms share indices
within each seed, but cache sizes cannot share identical sampled examples.
Common anchor/extension probes retain drawn/undrawn masks and null empty subsets;
they are not exhaustive full-cache fit measurements. Both frequency priors are
calibrated from their own training cache under the unchanged policy.

All 54 full, 108 fit-probe and 27 intervention cases replay exactly, retaining
every imported score vector, endpoint and criterion. The local suite passes 134
tests, including two publication-policy regressions. See the
[stage 10 review](docs/results/diversity_v10_run1/REVIEW.md).
The next step is separately frozen fresh phase-disjoint confirmation with both
objective strata/controls retained, focused on the observed objective-specific
effect. No such confirmation was launched. The manuscript and presentation
remain their earlier authored snapshots; they do not silently acquire stage 10
coverage. Original random-rule readiness remains failed, and language transfer
still requires fresh paired cohorts and an identical full language stage.

## Stage 11 â€” fresh-phase confirmation (21 October 2026)

The weighted primary confirmation **passes** (3/3 paired seeds); standard passes 1/3. The broader both-objective criterion **fails**.
The separately frozen weighted hypothesis follows stage 10's development
finding; that earlier broad primary remains failed. Fresh seeds 179/193/211
share one new split, excluding 4926 historical joint phase classes. All 128
intervention pairs are reserved outside fresh training/development. All 24
512/4096 Ã— standard/weighted Ã— coherent/shuffled models are new fits at the
unchanged rule/model/serializer/1000-update budget. No inspected endpoint was
continued or imported as a control.

All 54 full, 108 fit-probe and 27 intervention cells replay exactly. Sixty-six
copied records, nine raw evidence files and source/input/order/cost receipts
remain retained. The local suite passes 146 tests. See the
[stage 11 review](docs/results/confirmation_v11_run1/REVIEW.md).
The historical stage 10 CSV has a documented
[column caveat](docs/STAGE10_CSV_CAVEAT.md); its nested JSON/review and negative
broad primary are unchanged. Declare a separate bridge toward interacting categorical rules, changing one task property at a time and retaining both objectives and controls.
This is conditional confirmation on one known binary rule and one fresh split,
not evidence of GraphNCA acceleration or unseen-rule transfer. Original random-
rule checkpoints remain unready. Any language study needs a separately frozen
eligible synthetic configuration, fresh paired cohorts and an identical full
language stage. No language, official-test, remote/paid run, push or publication.


## Stage 12 — explicit OOD graph evaluation (30 October 2026)

Add an evaluation layer motivated by methodological feedback and CLRS-Text:
test a learned procedure outside its ID distribution, separating size shifts,
graph-family shifts and their joint shift. Preserve the question of graph/grid
PPT before the identical language stage. Any additional algorithmic adaptation
must also use identical data/order/optimizer/schedule/budget across arms; it is
a third stage, not synthetic PPT or additional hidden language exposure.

The implemented local diagnostic uses BFS/Dijkstra full distance vectors,
explicitly different from official CLRS-Text outputs. A separate official-export
scorer preserves official prompt/reference semantics. Paired relabeling,
edge-order and guaranteed answer-changing edge cuts distinguish invariance and
functional correctness. Independent all-pairs labels agree on 5760 written
examples. All 5120 assessment views are development evidence; future untouched
confirmation remains ungenerated. Graph-isomorphism-invariant conservative
exclusion includes intervention graphs, with repeated topology counts retained.

A predeclared historical zero-shot calibration uses the five final v4 language
arms in cohort 59, without any new fitting or task teaching. All 160 predictions
are invalid-format and incorrect; every input fits the original context. Exact
same-code checkpoint replay passes. This cannot establish graph equivalence,
no-effect or learning acceleration: the task interface is not learned. The
next learned calibration must establish ID competence before comparing OOD.

Read [stage 12 review](docs/results/ood_v12_run1/REVIEW.md),
[protocol](docs/OOD_V12_PROTOCOL.md) and [usage](docs/OOD_V12_USAGE.md).
Stage 11 results and original failed random-rule readiness stay intact. A future
confirmatory transfer study still needs eligible synthetic initialization, its
separate categorical bridge, fresh paired cohorts, identical downstream stages,
predeclared targets/censoring and total-cost accounting. No new training,
official-language-test access, remote/paid job or publication occurred here.

## Stage 13: matched historical ID adaptation completed

The [new calibration](docs/results/adaptation_v13_run1/REVIEW.md) trains all five
retained cohort-59 final language models with identical 600-update answer-only
supervised task adaptation, full inherited model/head policy, actual sample order,
optimizer and schedule. Every model scores 0/128 exact ID answers at the fixed
endpoint, despite 98.4–100% valid vectors. The 80% pooled/per-task-size competence
criterion fails and OOD is skipped. There are 3000 new CPU adaptation updates,
zero new language/PPT updates and no GPU measurement. Historical costs and old
negative results remain separate. Low NLL/format acquisition is not graph-task
competence or graph-specific acceleration. Original random-rule readiness stays
failed; future transfer requires its separate categorical bridge and fresh paired
cohorts. Before another fit, use existing checkpoints for a separately frozen,
task/size-matched training/development generated-error audit. Read the
[protocol](docs/ADAPTATION_V13_PROTOCOL.md) and
[usage](docs/ADAPTATION_V13_USAGE.md). Preserve the explicit
[integrity wording clarification](docs/ADAPTATION_V13_INTEGRITY_CLARIFICATION.md).

## Stage 14: matched ID checkpoint-error audit

The [new development diagnostic](docs/results/id_audit_v14_run1/REVIEW.md) uses
all five fixed stage 13 final models without fitting. Matched training and dev
probes have 128 examples each, 32 per task/size cell; salted graph-ID selection
is outcome-blind and genuine earlier exposure is verified. Every arm has zero
exact answers on both probes. Training node accuracy is 29.55–32.89% versus dev
18.95–21.60%. Conditional digit-token accuracy is weak even with gold previous
answer tokens, while other-token accuracy is nearly perfect. Reliable predictions
on seen examples are not established; the held-out difference remains descriptive,
not a unique cause or graph-specific treatment conclusion.

All 176 root tests pass. No optimizer update, new language/PPT/GPU measurement,
OOD prediction or untouched confirmation is introduced. The original random-rule
readiness failure and categorical bridge remain. The next proposal is a separately
frozen small-cache overfit positive control with model/tokenizer/objective/budget
held fixed and the original cache reference/all arms retained from common prior
starts. That new fit has not been launched. See the
[protocol](docs/ID_AUDIT_V14_PROTOCOL.md) and [usage](docs/ID_AUDIT_V14_USAGE.md).

## Stage 15: small-cache fitting positive control

The separately frozen [new study](docs/results/overfit_v15_run1/REVIEW.md)
restarts all five original stage 13 step-0 models and changes the training cache
from 512 to 16 nested, outcome-blind examples. Numerical updates, codec/model,
head policy, objective, optimizer and 600-update budget are fixed. Each selected
example receives 300 presentations. Actual token/padding differences and coupled
cache diversity/content/repetition are preserved in cost accounting. The imported
512-cache reference is scored on the exact same selected examples and full dev.

All five small-cache models pass the fixed final criterion with 16/16 exact
cache answers; the historical reference has 0/16. All new models still score
0/128 exact development answers, with worse validity/node/NLL than the 512-cache
reference. This establishes fitting capability on this limited cache, not a
unique cause of full-cache weakness or reliable algorithm generalization.
All 184 root tests pass. There are 3000 new CPU task-adaptation updates, zero
new PPT/language/GPU/OOD measurements; original reference costs remain imported.
The categorical/random-rule bridge, failed stronger readiness and OOD gates
remain intact. Any next diversity/budget/generalization treatment must be frozen
separately, with all controls retained. See the
[protocol](docs/OVERFIT_V15_PROTOCOL.md) and [usage](docs/OVERFIT_V15_USAGE.md).

## Stage 16: retained-output semantic and token error audit

The [frozen audit](docs/results/error_audit_v16_run1/REVIEW.md) independently
parses/hop-scores 1440 imported outputs, including every fixed endpoint arm,
cache-size variant and task/size cell. No scientific fitting or inference is
performed. Explicit invalid/valid-vector denominators show held-out semantic
weakness even among parsed vectors (16.83–19.81% node-micro for small16).
Teacher-forced digit token-micro accuracy is also low (19.18–20.57%), and source/
one-hop errors persist. A sole formatting or long-path explanation is inadequate;
no unique causal diagnosis follows. Keep all prior failed endpoints and gates.

The frozen no-OOD-data wording was broader than implementation: unchanged
deterministic generation also recreates unused assessment/OOD rows in memory.
The [disclosure](docs/ERROR_AUDIT_V16_REGENERATION_NOTE.md) preserves that wording
and executed sources. No assessment/OOD scoring, prediction, selection or model
input occurs. The next proposed cache64/fixed600 development treatment requires
its own freeze and all controls, not an extension or retuning of this audit.

## Stage 17: intermediate-cache fitting study

The [separately frozen experiment](docs/results/cache_v17_run1/REVIEW.md) changes
cache size to 64 examples at the same 600 updates, from all five original full
pre-adaptation states. Every arm reaches 64/64 exact cache answers, with all
four cells 16/16. Every arm still has 0/128 exact development answers. All
pooled cache crossings are first observed at update 300; no speedup ratio is
inferred from these coarse observations or across different training caches.

The SAME common16/dev128 cases retain both imported references; only reference512
covers the full64. New64 held-out NLL improves over reference16 in every arm,
while reference512 has better NLL AND node accuracy in every arm. Graph node
accuracy is 17.09% versus grid 12.54% and matched prefix-graph 18.71%. All are
descriptive single-cohort endpoints, not a graph acceleration/equivalence claim.
Full-ID competence, categorical bridge and original random-rule readiness remain
unmet. No OOD, new language/PPT run or official-language-test evaluation occurs.

All 20 checkpoints/1,920 new predictions and teacher scores replay exactly in a
separate invocation; 199 root tests and independent counter/public score checks
pass. Public reproduction needs no private weights/data/cache/network. New
scientific updates are 3,000 CPU task-adaptation updates; 6,000 reference updates
are imported. The run changes no frozen target or prior evidence. Future work
must declare a separate treatment, preserving all controls and existing gates.

## Stage18: direct exploratory ID acceleration measurement

User authorizes the exploratory study despite the unchanged original synthetic-
readiness failure; this is a prospective exception for development measurement,
not a change to the separate confirmation protocol. The
[frozen plan](docs/ACCEL_V18_PROTOCOL.md) uses the same five original language-
trained states, new shared ID sampling/order, 600 updates and seven held-out
evaluations. All language stages remain identical before this new task stage.
No new PPT or language stage is trained. The target requires non-source semantic
accuracy +10pp above a source-aware train-marginal baseline in all four cells,
95% pooled validity and coverage1. Exact-vector calibration is retained separately.

All five fits complete without reaching that target: every arm is censored at
600, all140 individual semantic cell targets also remain unmet, and speedup
ratios are undefined. Full curves/endpoints stay intact; do not replace the
threshold or cap to obtain a ratio. The result measures task adaptation after
the language stage, not language convergence or standalone GNN speed. It does
not establish graph equivalence, universal harm/impossibility or confirmation.

Include full imported T4 generation/preparation/PPT/language costs and actual
new CPU token/time counts, separating heterogeneous-hardware components and
unmeasured overheads. All208 root tests and independent4480-output parser/
counter/cost checks pass; separate neural replay verifies all 35 checkpoints
and 4,480 predictions/teacher scores, cost/target decisions and optimizer budgets.
Original categorical/random-rule bridge, eligible starts, fresh paired cohorts,
all controls, ID-only selection, OOD criteria/censoring/multiplicity and full
cost requirements remain intact for confirmation. No OOD run occurs.

## Stage19: checkpoint-only fitting and generalization diagnostic

The [new frozen diagnostic](docs/FIT_AUDIT_V19_PROTOCOL.md) measures all five
stage18 final models on128 outcome-blind seen cases and imports the unchanged
development128 reference. It uses the same original train-fitted marginal
predictor, scored separately on each probe's own cases; old held-out targets
remain exact. All five fail the semantic target on BOTH probes, with zero exact
answers on seen data as well as development. Seen whole-node accuracy is only
23.48–28.65% despite prior exposure. Held-out generalization alone cannot explain
all failure; neither this result nor declining mixed-token loss identifies a
unique cause or guarantees that more training is the solution.

All213 root tests pass. Separate five-checkpoint replay verifies640 new seen
generations/teacher scores, optimizer/count identities, interpretations and trace
summaries. The640 dev outputs are imported, not new neural measurements. All600
updates per arm and fixed blocks/first50/last50 summaries are retained. There is
no fitting, new language/PPT/GPU/OOD/official-test evaluation or confirmation-gate
change. A later longer-budget treatment must freeze data/model/objective/controls,
schedule, target/censoring and costs separately, preserving all old endpoints.

## Separate scientific-transfer milestone completion (2 November 2026)

The numerical/specification milestone2 for the explicitly declared
steady_hybrid_v2 is complete: all four cases pass original numerical variants
and independent full-system49/65 spectral checks, with separate exact replay
except timers. The failed first-order spectral attempt remains retained. This
is not faithful reproduction of the unresolved CEJA manuscript or experimental
validation. See docs/results/scientific_transfer_completion/REVIEW.md.

Milestone3 now implements a matched numeric-interface runner and prepares all
five actual original v4 cohort59 language starts, common adapter tensors,32/8
ID-only reference profiles and fixed400-update schedule/competence/censoring.
The real fit guard rejects before a fitting marker or scientific update because
the retained original random-rule starts still fail stronger synthetic readiness.
No old stage18 exception is inferred. Planned exposure is not measured training;
new numeric query tokens are not language tokens. All26 new/retained scientific
unit tests and213 root regressions pass locally. No new scientific neural fit,
GPU/PPT/language/OOD/official-test result, graph speedup or cost recovery exists.

## Milestone 3 exploratory execution and milestone 4 ID assessment

All five original cohort 59 arms completed 400 matched numeric adaptation
updates in the separate local CPU exploratory execution. Every arm passed
0/8 ID development profiles at the endpoint and never reached the frozen
competence criterion. All crossing times are censored at 400. The ID gate
keeps OOD evaluation pending. Separate replay verifies 25 checkpoints and
200 predictions. See [the review](docs/results/scientific_m3_exploratory/REVIEW.md).
