# Changelog

Public changes are summarized here; measurements and failures are in
[LOGBOOK.md](LOGBOOK.md).

## Unreleased

### Scientific-transfer material/radiation milestone - 2026-11-01

- Visually audit manuscript equations: withdraw the extraction-based suspected
  Pr mismatch; document printed density/momentum issues and unresolved author
  intent. Faithful replication remains disabled.
- Add explicitly derived steady_hybrid_v2 with final-volume mixture conventions,
  conservative nonlinear radiation and named dimensionless wall-flux outputs.
- Add source-bound precheck freeze, four-case separate tolerance/mesh/domain
  verification and independent Chebyshev Homann/thermal limiting reference.
  Preserve v1 artifacts and all historical neural results; no neural/OOD run.
- All four numerical cases pass; independent wall errors are below2.3e-9
  (velocity) and8.1e-10 (thermal). All19 scientific tests and213 root regressions
  pass locally. Retain exact freeze/result/provenance evidence and explicit
  partial/pending milestone decisions; no manuscript validation claim.

### Scientific-transfer solver foundation - 2026-11-01

- Add separate `scientific_transfer/` source/config/tests and development docs:
  declared steady bridge, collocation solver, domain/mesh checks, immutable
  split/source manifests, full-profile scoring and supplied-guess correction.
- Audit unresolved manuscript equations; keep faithful manuscript replication
  and neural training disabled. Historical gates/results remain unchanged.
- Select this source directory for distribution; runtime solves stay under
  ignored runs. Extend logged command provenance with scientific source hashes.

### Seen/held-out fitting and trace diagnostic stage 19 — 2026-11-01

- Complete all five frozen endpoint evaluations: every arm misses semantic
  readiness on its seen probe as well as development, with0/128 exact vectors
  on both. Preserve mixed-token loss declines and all per-cell/counter differences.
- All213 root tests pass; five checkpoints/640 new seen predictions and teacher
  scores replay exactly. The640 development outputs remain imported evidence.
- Public-only reconstruction verifies1,280 new/imported answers,3,000 original
  trace rows,40 cells and30 blocks without private weights/data/cache/network.
  Preserve36 exact copies and all old checkpoints/targets/eligibility flags.
- Standalone also passes all213 tests and public reproduction. Preserve the
  separate scientific-transfer package added by another session; include all
  current Lean/theory/paper sources and records without new compiler claims.

- Prepare a separately frozen checkpoint-only diagnostic for all five stage18
  final models. Use outcome-blind balanced128 seen cases and unchanged imported
  dev128, probe-specific scores of the same frozen marginal predictor, and all
  600 optimization updates. No fitting, target relaxation or endpoint selection.

### Exploratory ID acceleration stage 18 — 2026-10-31 to 2026-11-01

- Complete all five matched fits and seven held-out evaluations per arm. All
  targets remain unreached: every arm is right-censored at 600 updates, with
  undefined ratios and full cost accounting. No threshold/budget/control changes.
- All 208 root tests pass; separate replay verifies 35 checkpoints/4,480
  predictions, teacher scores, target/censoring/cost and optimizer identities.
- Public-only ID data/order/baseline/cost/generated-score reconstruction passes;
  retain all 140 cell rows and 60 exact evidence copies. Costs at the full budget
  are explicitly separate from unobserved target-crossing costs.
- Standalone also passes all 208 tests and public-only reproduction. Preserve
  recursive Lean/theory/paper sources and verification records with root/Prod
  parity; no new compiler, GPU, OOD, remote CI or external publication is claimed.

- Prepare a prospective five-arm ID learning-curve study from the same original
  language endpoints, with a fresh ID sample and identical 600-update training.
  Freeze a per-cell non-source semantic advantage over a training-only marginal
  predictor, complete crossings/censoring, both graph contrasts and stage costs.
- Keep earlier calibration/confirmation gates and all historical endpoints intact.
  Imported T4 PPT/preparation/language costs remain separate from new CPU costs.

### Intermediate cache study stage 17 — 2026-10-31

- Complete all five 600-update CPU fits: every arm gets64/64 cache answers and
  0/128 exact held-out answers. Preserve this fitting-only pass and failed full-ID
  endpoint. Prefix-graph's held-out node score exceeds graph's in this cohort;
  no graph-specific acceleration, equivalence or readiness claim is inferred.
- All 199 root tests pass. All 20 checkpoints/1,920 predictions and teacher
  scores replay exactly; 72 copies, 220 cells, actual exposure/token accounting
  and public-only reconstruction of both references and new scores pass.
- Standalone validation also passes all 199 tests and public-only reproduction.
  Retain proof/theory/paper sources and records with root/Prod parity, without
  claiming new compiler or remote CI execution.

- Prepare a separately frozen 64-example/600-update development treatment from
  all five original pre-adaptation states. Reuse the exact numerical helpers,
  codec, objective and targets. Retain imported16/512 references on common16
  and full development128; reference512 additionally covers cache64.
- Preserve original endpoints and all readiness/OOD gates; actual token/padding
  and 75 presentations/example are accounted separately from imported costs.

### Retained-output error audit stage 16 — 2026-10-31

- Complete separate generated-parser/hop replay of 1440 imported outputs;
  100 diagnostic cells and 50 paired rows, full counters and review reproduce.
  Even valid held-out vectors have poor node accuracy; source/one-hop and gold-
  prefix digit failures persist. Preserve all gates and original endpoints.
- All 193 root and standalone tests pass; public-only replay, counter/CSV/review
  accounting and prior stage 15 evidence pass. Retain recursive Lean/theory/paper
  sources and verification records with root/Prod parity, without new compilers.
- Disclose that the unchanged deterministic generator recreates unused OOD/
  assessment rows in memory, broader than the frozen no-OOD-data wording; none
  are scored, predicted or used for selection. Preserve protocol and sources.

- Freeze all five arms, both fixed update-600 variants, the same cache/dev
  examples and all task/size cells for digit/source/hop/format error analysis.
  No fitting, inference, new targets or eligibility changes are introduced.
- Preserve the entire append-only check history with a bounded 10 MB allowance
  for the exact `logs/checks.jsonl` path only; all other file limits and private
  artifact exclusions remain. Boundary regression checks cover the exception.

### Small-cache overfit control stage 15 — 2026-10-31

- Complete all five CPU fits: 16/16 cache answers each, versus 0/16 for the
  same-example 512-cache reference. Held-out exact remains 0/128 and continuous
  metrics worsen. Preserve this fitting-only pass and all readiness/OOD gates.
- All 184 root tests pass. Twenty checkpoints/960 predictions and teacher scores
  replay exactly; 52 copies, 140 cells and independent numerical/exposure/reference
  accounting pass. Add public-only reproduction and reviewed plots.

- Freeze a one-factor 512-to-16 cache control from all five original complete
  pre-adaptation states. Preserve numerical training settings/600 updates and
  account for actual token/padding differences and 300 presentations/example.
- Retain verified imported 512-cache reference on the exact same examples.
  Add eight numerical/selection/reference/criterion checks, source/order receipts,
  checkpoint replay, independent reporting and public-only score reproduction.
  Cache fitting alone cannot clear OOD or original synthetic readiness.


### Matched ID error audit stage 14 — 2026-10-31

- Complete all five matched checkpoint audits: zero exact answers on seen and
  held-out probes, weak digit prediction despite near-perfect other tokens.
  Preserve all negative results and gates. All 1280 generations/teacher-forced
  scores replay exactly; 22 copies, 40 cells and independent accounting pass.
  Public-only probe/error reproduction and readable diagnostic plots are added.
  Next proposal: a separately frozen small-cache overfit control; no new fit.

- Freeze a checkpoint-only seen/held-out audit of all five final stage 13 models.
  Match 128 train/development probes by task/size and verify actual train exposure.
- Add unconditional source/hop/value error accounting and teacher-forced digit/
  other-token scores, generated-error public reproduction and replay. Eight new
  tests and the 176-test local suite pass. No scientific fitting or new OOD run.


### Matched ID adaptation stage 13 — 2026-10-30

- Complete all five 600-update CPU fits. Format validity reaches 98.4-100%,
  but every endpoint has 0/128 exact ID answers; preserve the failed criterion,
  censored targets and OOD skip. All 168 local tests pass; all 20 checkpoints/
  2560 generations replay exactly. Export 32 exact evidence copies, audited
  curves/cells/review and a weight-free public score/data-reproduction helper.
- Document the size composition of the training NLL probe and clarify old
  assessment integrity hashing without changing the original frozen protocol.

- Freeze and implement a bounded five-arm historical task-interface/trainability
  calibration after the identical saved language stage: common answer-only SFT,
  full-model/head retention, batch order, optimizer and schedule. Keep original
  random-rule readiness failed and all earlier evidence intact.
- Record generated-answer curves, fixed per-cell competence/format criteria,
  censored crossings and exact exposure/costs. OOD is gated by all final ID arms.
  Add eight loss/gradient/masking/decision checks and checkpoint replay/export.


### Out-of-domain graph evaluation stage 12 — 2026-10-30

- Complete CPU measurement checks and the frozen five-model historical zero-shot
  calibration: 5,760 independent-oracle labels, 160 replayed predictions and
  160 passing local tests. Preserve the uniform invalid-format floor; no graph
  effect/no-effect or acceleration claim. Export verified evidence and runnable
  usage; retain all old results, privacy exclusions, costs and setup failures.

- Add CLRS-Text-inspired graph-size, family and joint-shift evaluation, strict
  generated-answer scoring, paired relabel/order/cut controls, official-export
  scoring and future matched-stage/censored-crossing guards.
- Pin seven official reference files and Apache-2.0 license. Private correspondence
  remains excluded. A frozen historical zero-shot calibration uses existing
  checkpoints only; no new pretraining, language or task-adaptation run.

### Fresh-phase confirmation stage 11 â€” 2026-10-21

- Freeze a prospective confirmation of the stage 10 weighted diversity finding
  with fresh initialization seeds and data excluding 4,926 historical joint
  phase classes. Reserve all 128 intervention pairs before selecting training.
- Fit both 512/4096 caches, both standard/weighted objectives and both coherent/
  shuffled arms anew: 24 models at the unchanged 1000-update budget. Keep the
  weighted primary and broader both-objective criterion distinct, preserving
  stage 10's failed primary. Assessment waits for all declared endpoints.
- Seven new regressions and the full 141-test local suite pass. Preserve private
  arrays/weights, frozen source receipts and actual row exposure. Add compact
  report, provenance and replay-accounting helpers. No language command or
  remote/paid training is introduced.
- Document a historical stage 10 derived-CSV column collision without changing
  preserved evidence or its correct JSON/review outcome. Stage 11 keeps complete
  comparison and readiness decisions separate, with a passing regression.

- Completed all 24 fresh CPU fits. Weighted confirmation passes (3/3);
  standard passes 1/3 and the broader both-objective criterion fails.
  All 189 scoring cases replay exactly; 71 evidence copies and chronology pass.
  The full local suite passes 146 tests. Preserve every control and cost.
- Preserve a Windows status-write interruption and journaled recovery: 21
  endpoints stay unchanged, the unfinished fit restarts from its frozen start,
  and its 800-update metric prefix reproduces. Charge all 800 discarded updates;
  the overlapping unused recovery preparation launches no fitting.

### Initial-condition diversity stage 10 â€” 2026-10-21

- Declare a nested 512-to-4096 training-condition ablation with the spatial rule,
  objectives, full model and 1000-update/token exposure fixed.
- Retain twelve imported standard/weighted coherent/shuffled controls; fit
  twelve new larger-cache models. Freeze all-seed criteria for BOTH objectives.
- Ten new regressions pass, including phase/intervention exclusions, exact nested
  shuffles, actual sampling coverage and an isolated four-arm execution fixture.
  No language, remote or paid training is introduced.
- Completed all twelve larger-cache CPU models at the fixed budget. Weighted
  loss passes all three paired comparisons; standard passes only one. Preserve
  the failed study-wide primary and the objective-specific weighted gain.
- All 54 full, 108 fit and 27 intervention cases replay exactly from 27 saved
  initial/final files. Retain 42 copied evidence files, nine raw per-seed records,
  actual sampling coverage, exposure-aware probes, CSVs and figures.
- The complete local suite passes 134 tests, including two publication-policy
  checks. Exclude generated paper build output while retaining sources and figures.
  Earlier manuscript/presentation
  snapshots and every historical model/result remain preserved.

### Research manuscript through stage 9 â€” 2026-10-20

- Added an author-attributed A4 research manuscript, editable LaTeX/content
  sources, eight scientific figures and source/validation records.
- Integrated mathematics, partial Lean coverage, versioned GPU findings and
  controlled CPU studies through stage 9, retaining negative primary outcomes.
- Added an exact position-wise loss decomposition distinguishing first-patch
  deterioration from improved teacher-forced aggregate transition likelihood.
- Selected the new source project for Prod; rendered PDFs remain local output
  artifacts under the existing publication policy.

### Existing-checkpoint first-transition fit diagnostic stage 9 â€” 2026-10-20

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

### Presentation through research stage 8 â€” 2026-10-20

- Added a 128-slide research presentation with reviewed v3/v4 GPU findings,
  stronger readiness checks and completed CPU diagnostics through stage 8.
- Retained the negative weighting outcome, actual code and fourth notebook
  excerpts, new recorded plots and mathematical interpretations alongside the
  earlier derivations, dataset views and partial Lean coverage.
- Preserved the earlier presentation and all four original editable charts
  with their actual workbooks, and refreshed the slide/source guide and Prod.

### First-transition loss-weighting stage 8 â€” 2026-10-20

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

### Matched spatial-interaction bridge stage 7 â€” 2026-10-20

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

### Controlled transformer trainability stage 6 â€” 2026-10-20

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

### Production refresh and mathematics inclusion â€” 2026-10-19

- Refreshed the standalone publication from current canonical research sources.
- Recorded the standing requirement to include all new Lean proofs, proof
  documentation/evidence and LaTeX mathematical documentation in every Prod
  update, using recursive publication selection and checksum verification.

### Checkpoint-only readiness stage 5 â€” 2026-10-19

- Added a separate frozen diagnostic and reusable language-start guard requiring
  both frequency and coherent-ordering advantages, without editing v1-v4 runners.
- Completed 48 CPU cross-evaluation cells from 24 existing v4 PPT checkpoints.
  All 12 structured cohorts fail both fixed 0.01-nat requirements; language
  training remains blocked. A learned known-rule transition-table control passes.
- Retained per-rule evidence, receipts, compact provenance and paired plots.
  Eight new regressions pass; the complete local suite passes 86 tests.
  No new neural training, GPU run or official-test evaluation was performed.

### Returned prefix-control v4 results â€” 2026-10-19

- Added a safe returned-archive auditor, independent common-target/sign-test
  reconstruction, full 78-checkpoint/six-initialization CPU replay and compact
  evidence with SHA provenance. Both frozen graph-advantage comparisons fail.
- Added a post-hoc causal frequency diagnostic that outperforms all structured
  PPT models, without changing the declared v4 gate or primary result.
- Added paired-effect plots, a full review and checkpoint-first next steps;
  updated research status while preserving all earlier evidence and runner code.
- Retained an initial Windows raw-ZIP-path test failure and its correction;
  the final local software suite passes all 78 tests. No new GPU run was launched.

### Prefix controls and fresh replication, v4 â€” 2026-10-19

- Added a separate seven-arm v4 runner and self-contained Kaggle notebook with
  six fresh paired cohorts, preserving measured v1/v2/v3 sources and evidence.
- Added exact supervised-prefix frequency/format controls with unchanged initial
  frames; retained both original shuffled arms and the v3 model/generator settings.
- Froze two common-target comparisons, censoring and multiplicity rules; archive
  the prospective plan and verify it with checkpoints and raw-curve analysis.
- Added offline complete/gate-stop/test-fixture checks and control/statistical
  regressions. These are CPU implementation checks; v4 has no GPU result yet.

### Returned graph-transfer v3 results â€” 2026-10-19

- Added a reproducible archive/metric/checkpoint audit and compact v3 GPU/CPU
  evidence with paired graph-versus-grid and graph-versus-shuffle plots.
- Graph has a small mean final NLL advantage, but the primary grid-endpoint
  comparison is censored in one of three replicates; no all-seed speedup is claimed.
- Retained language-stage gains versus no PPT separately from cumulative cost;
  the official test, prior runners, checkpoints and returned archives are preserved.
- Restored the missing original presentation from its unchanged, checksum-verified
  Prod copy, repairing the previously recorded publication-sync blocker.
### Detailed research presentation â€” 2026-10-17

- Expanded the supplied descriptive presentation into a detailed research deck
  with the original material, literature, mathematical derivations, Lean proof
  excerpts, actual notebook/code references, dataset views and measured results.
- Retained historical and fixture plots with explicit scope, distinguished
  language-stage acceleration from cumulative cost, and covered the prepared
  v3 graph methodology separately from measured v2 outcomes.
- Added a slide/source coverage index and preserved the original presentation.

### Graph transfer next stage â€” 2026-10-17

- Added versioned `transfer_v3/` and a self-contained graph-vs-grid Kaggle
  notebook, preserving v1/v2 runner sources and returned measurements.
- Five PPT initializations feed identical language stages. Grid/graph share
  frozen rules selected by joint complexity eligibility; three paired seeds
  use independent synthetic caches and degree-preserving graph topologies.
- Added saved starting tensors, graph and shuffle integrity checks, graph-vs-grid
  censored reference comparisons, graph-specific unit/integration checks and
  configured CPU CI. The official test remains unused; no v3 GPU result is claimed.

### Results publication refresh â€” 2026-10-17

- Added a top-level results ledger with reviewed v2 GPU metrics, audit limits
  and the separate mathematical verification status.
- Made configuration, test and helper selection explicit so unpublished
  experiment development does not enter the reviewed publication implicitly.
- Refreshed both notebook bundles and the canonical-root/Prod result copies;
  validation details and check outcomes are retained in the logbook.

### Returned transfer-v2 development results â€” 2026-10-17

- Retained the original Kaggle archive and extracted evidence; added a separate
  reproducible CPU audit and compact GPU records with provenance.
- Grid pre-pre-training reached the paired scratch endpoint with 10â€“20% fewer
  language updates across three seeds. The shuffled control had comparable gains;
  dynamics-specific acceleration and graph benefits remain unestablished.
- Recorded cumulative grid time was about 13% higher at the reference loss.
  The language confirmation split remains unused; historical pilot 1 is unchanged.

### Lean proof documentation â€” 2026-10-17

- Added a dedicated Lean 4 proof project, a LaTeX coverage map, reproducible
  browser bundle, axiom audit, verification records and a separate proof CI job.
- All 44 statements passed in the official Lean 4.35.0 web editor; retained
  records include the corrected source hashes and the failed first full check.
  A local modular build and remote proof CI have not been run.
- Selected algebraic, graph, serialization and analytic-toy claims are formalized;
  full entropy, spectral and smoothness arguments remain explicitly unformalized.
- Excluded Lean dependency caches and compiled objects from publication.

### Mathematical documentation â€” 2026-10-17

- Added a standalone LaTeX theory project with graph-update proofs, information
  bounds, exact serialization calculations, conditional transfer results and
  worked convergence/cost examples. No simulation or training code added.
- Added a numerical arithmetic ledger and documented which conclusions are
  proved under assumptions versus still-open language-transfer hypotheses.

### Added â€” 2026-10-17

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

### Changed â€” 2026-10-17

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


## 2026-10-21 â€” Fresh-phase confirmation (stage 11)

Declared a new confirmation protocol retaining both objectives and controls, fresh phase-disjoint data/seeds, unchanged per-model budgets and separate weighted/both-objective criteria. Results pending; no language stage.
