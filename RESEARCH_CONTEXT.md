# Research context: GNNs in NCA pre-pre-training

Updated: 20 October 2026.

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
