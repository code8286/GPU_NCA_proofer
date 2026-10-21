# Results ledger

Updated 20 October 2026. This page summarizes reviewed evidence. Detailed
commands, failures and corrections are retained in [LOGBOOK.md](LOGBOOK.md)
and `logs/checks.jsonl`.

## Latest CPU diagnostic: split fit and generalization stage 9

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
