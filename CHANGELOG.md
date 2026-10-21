# Changelog

Public changes are summarized here; measurements and failures are in
[LOGBOOK.md](LOGBOOK.md).

## Unreleased

### Existing-checkpoint first-transition fit diagnostic stage 9 — 2026-10-20

- Declare an inference-only comparison across all three existing splits, both
  data versions and all 15 initial/standard/weighted model conditions.
- Preserve failed stage 8 weighting evidence and all coherent/shuffled controls.
  Freeze descriptive fit-versus-generalization comparisons before new scoring.
  No training or language-stage entry point is introduced.
- Completed all 90 CPU inference cells and exact checkpoint replay; all 15
  coherent assessment score vectors agree with stage 8. The full suite passes
  122 tests, including eight new scoring/comparison and two snapshot regressions.
- Two seeds improve training first-patch NLL but worsen held-out NLL; one also
  worsens training. Whole-transition patch prediction improves while held-out
  first-patch prediction worsens. Preserve the mixed result and all controls.
- Make starter packaging bind file members and hashes to one checked inventory,
  preventing manifest/member races when sources appear during packaging.

### Presentation through research stage 8 — 2026-10-20

- Added a 128-slide research presentation with reviewed v3/v4 GPU findings,
  stronger readiness checks and completed CPU diagnostics through stage 8.
- Retained the negative weighting outcome, actual code and fourth notebook
  excerpts, new recorded plots and mathematical interpretations alongside the
  earlier derivations, dataset views and partial Lean coverage.
- Preserved the earlier presentation and all four original editable charts
  with their actual workbooks, and refreshed the slide/source guide and Prod.

### First-transition loss-weighting stage 8 — 2026-10-20

- Added a separately frozen sixfold first-transition objective ablation, keeping
  the interacting rule, cached inputs, initialization and training budget fixed.
- Reuse six verified spatial standard-loss checkpoints as historical matched
  controls; train six weighted coherent/shuffled models. Language remains gated.
- Eight objective/matching regressions and a retained local execution fixture
  pass; the full local suite passes 112 tests.
- Completed all six new 1000-update CPU models. The frozen primary fails in
  every coherent seed: original and intervention NLL both worsen. Mean first
  accuracy falls 47.40% to 32.03%, intervention 42.08% to 30.05%, and full
  trajectory 91.84% to 50.86%. Preserve the negative result without retuning.
- Exact replay of 30 full and 15 first-transition cases, including imported
  standard controls, passes. Retain 21 copied evidence files, CSVs and a figure.

### Matched spatial-interaction bridge stage 7 — 2026-10-20

- Added a separate paired complement/neighbor-parity diagnostic with fixed v6
  model, representation and training settings, preserving all prior evidence.
- Guard joint rule-phase leakage with one shared split and a newly trained
  complement reference. Keep both readiness checks and separately predeclare
  a spatial qualification against a previous-own-patch predictor.
- Ten new structural/integrity regressions and a four-arm tiny CPU execution
  fixture pass; the full local suite passes 104 tests.
- Completed twelve full-budget CPU models and exact replay of 36 primary cases.
  Spatial readiness/local-table checks pass; mean trajectory accuracy is 91.84%.
- Separately froze a first-transition/neighbor-intervention diagnostic during
  training before assessment, preserving the primary plan. All 18 cases replay;
  NLL checks pass but original/intervened accuracy is only 47.40% / 42.08%.
  Preserve this partial-learning limit; no language/GPU run followed.

### Controlled transformer trainability stage 6 — 2026-10-20

- Added a separate known-rule transformer diagnostic preserving the full v4
  backbone and numerical training pipeline, with explicitly simplified data.
- Fixed three paired model seeds, coherent/shuffled arms, final 1000-update
  checkpoints, phase-disjoint held-out initial conditions and both readiness
  margins before training. No language stage exists in this diagnostic.
- Eight new regressions and a retained tiny CPU execution fixture pass.
  The complete local software suite passes 94 tests.
- Completed six full CPU models: all three coherent-trained checkpoints pass
  both margins with 100% held-out patch accuracy. All 18 assessment cells replay
  exactly. Preserved compact evidence, source/input receipts and learning curves.
- Recorded that shuffle-trained controls also pass the absolute margins with
  much weaker performance; admission is not a mechanism test. No language/GPU
  run or random-rule checkpoint clearance follows from this simpler task.

### Production refresh and mathematics inclusion — 2026-10-19

- Refreshed the standalone publication from current canonical research sources.
- Recorded the standing requirement to include all new Lean proofs, proof
  documentation/evidence and LaTeX mathematical documentation in every Prod
  update, using recursive publication selection and checksum verification.

### Checkpoint-only readiness stage 5 — 2026-10-19

- Added a separate frozen diagnostic and reusable language-start guard requiring
  both frequency and coherent-ordering advantages, without editing v1-v4 runners.
- Completed 48 CPU cross-evaluation cells from 24 existing v4 PPT checkpoints.
  All 12 structured cohorts fail both fixed 0.01-nat requirements; language
  training remains blocked. A learned known-rule transition-table control passes.
- Retained per-rule evidence, receipts, compact provenance and paired plots.
  Eight new regressions pass; the complete local suite passes 86 tests.
  No new neural training, GPU run or official-test evaluation was performed.

### Returned prefix-control v4 results — 2026-10-19

- Added a safe returned-archive auditor, independent common-target/sign-test
  reconstruction, full 78-checkpoint/six-initialization CPU replay and compact
  evidence with SHA provenance. Both frozen graph-advantage comparisons fail.
- Added a post-hoc causal frequency diagnostic that outperforms all structured
  PPT models, without changing the declared v4 gate or primary result.
- Added paired-effect plots, a full review and checkpoint-first next steps;
  updated research status while preserving all earlier evidence and runner code.
- Retained an initial Windows raw-ZIP-path test failure and its correction;
  the final local software suite passes all 78 tests. No new GPU run was launched.

### Prefix controls and fresh replication, v4 — 2026-10-19

- Added a separate seven-arm v4 runner and self-contained Kaggle notebook with
  six fresh paired cohorts, preserving measured v1/v2/v3 sources and evidence.
- Added exact supervised-prefix frequency/format controls with unchanged initial
  frames; retained both original shuffled arms and the v3 model/generator settings.
- Froze two common-target comparisons, censoring and multiplicity rules; archive
  the prospective plan and verify it with checkpoints and raw-curve analysis.
- Added offline complete/gate-stop/test-fixture checks and control/statistical
  regressions. These are CPU implementation checks; v4 has no GPU result yet.

### Returned graph-transfer v3 results — 2026-10-19

- Added a reproducible archive/metric/checkpoint audit and compact v3 GPU/CPU
  evidence with paired graph-versus-grid and graph-versus-shuffle plots.
- Graph has a small mean final NLL advantage, but the primary grid-endpoint
  comparison is censored in one of three replicates; no all-seed speedup is claimed.
- Retained language-stage gains versus no PPT separately from cumulative cost;
  the official test, prior runners, checkpoints and returned archives are preserved.
- Restored the missing original presentation from its unchanged, checksum-verified
  Prod copy, repairing the previously recorded publication-sync blocker.
### Detailed research presentation — 2026-10-17

- Expanded the supplied descriptive presentation into a detailed research deck
  with the original material, literature, mathematical derivations, Lean proof
  excerpts, actual notebook/code references, dataset views and measured results.
- Retained historical and fixture plots with explicit scope, distinguished
  language-stage acceleration from cumulative cost, and covered the prepared
  v3 graph methodology separately from measured v2 outcomes.
- Added a slide/source coverage index and preserved the original presentation.

### Graph transfer next stage — 2026-10-17

- Added versioned `transfer_v3/` and a self-contained graph-vs-grid Kaggle
  notebook, preserving v1/v2 runner sources and returned measurements.
- Five PPT initializations feed identical language stages. Grid/graph share
  frozen rules selected by joint complexity eligibility; three paired seeds
  use independent synthetic caches and degree-preserving graph topologies.
- Added saved starting tensors, graph and shuffle integrity checks, graph-vs-grid
  censored reference comparisons, graph-specific unit/integration checks and
  configured CPU CI. The official test remains unused; no v3 GPU result is claimed.

### Results publication refresh — 2026-10-17

- Added a top-level results ledger with reviewed v2 GPU metrics, audit limits
  and the separate mathematical verification status.
- Made configuration, test and helper selection explicit so unpublished
  experiment development does not enter the reviewed publication implicitly.
- Refreshed both notebook bundles and the canonical-root/Prod result copies;
  validation details and check outcomes are retained in the logbook.

### Returned transfer-v2 development results — 2026-10-17

- Retained the original Kaggle archive and extracted evidence; added a separate
  reproducible CPU audit and compact GPU records with provenance.
- Grid pre-pre-training reached the paired scratch endpoint with 10–20% fewer
  language updates across three seeds. The shuffled control had comparable gains;
  dynamics-specific acceleration and graph benefits remain unestablished.
- Recorded cumulative grid time was about 13% higher at the reference loss.
  The language confirmation split remains unused; historical pilot 1 is unchanged.

### Lean proof documentation — 2026-10-17

- Added a dedicated Lean 4 proof project, a LaTeX coverage map, reproducible
  browser bundle, axiom audit, verification records and a separate proof CI job.
- All 44 statements passed in the official Lean 4.35.0 web editor; retained
  records include the corrected source hashes and the failed first full check.
  A local modular build and remote proof CI have not been run.
- Selected algebraic, graph, serialization and analytic-toy claims are formalized;
  full entropy, spectral and smoothness arguments remain explicitly unformalized.
- Excluded Lean dependency caches and compiled objects from publication.

### Mathematical documentation — 2026-10-17

- Added a standalone LaTeX theory project with graph-update proofs, information
  bounds, exact serialization calculations, conditional transfer results and
  worked convergence/cost examples. No simulation or training code added.
- Added a numerical arithmetic ledger and documented which conclusions are
  proved under assumptions versus still-open language-transfer hypotheses.

### Added — 2026-10-17

- Independent-stage `transfer_v2/` runner and `kaggle_nca_transfer_v2.ipynb`:
  categorical rule-disjoint NCA, complexity filtering and burn-in, held-out
  prediction gate, saved PPT checkpoints, matched language heads/data/schedules,
  and separate language/cumulative acceleration reports.
- Pinned WikiText-2/GPT-2 preparation with reserved confirmation data, disposable
  timing calibration, synthetic diagnostics plots and stage-checkpoint verification.
- V2 scientific regressions, complete/gate-stopped/confirmation-fixture integration
  workflows, checked extra dependencies and a detailed v2 protocol.
- Standalone `Prod/` copy, explicit file selection, SHA-256 inventory and drift checks.
- Apache-2.0 license, third-party notices, contributor guide, issue/PR templates,
  CPU CI, line-ending policy and expanded ignore rules.
- Append-only check transcripts and an offline five-arm smoke check.
- Compact historical CPU/GPU evidence with provenance.

### Changed — 2026-10-17

- Reorganized README around the research question, setup, contribution and limitations.
- Qualified stale CPU-only validation wording to acknowledge the returned GPU pilot.
- Added checkable notebook generation, pinned reference retrieval and explicit
  artifact paths for historical helpers.
- Preserved historical `experiment.py` byte-for-byte.

### Known limitations

- The returned v2 GPU benchmark is exploratory and uses three model seeds with
  one shared synthetic cache. Graph rewiring remains untested in v2.
- V2 uses a small model and a shared synthetic cache, not a full paper reproduction;
  step-level resume is not implemented.
- New publication checks use CPU. CI and notebook GPU execution are configured,
  not newly completed remote measurements.
