# Changelog

Public changes are summarized here; measurements and failures are in
[LOGBOOK.md](LOGBOOK.md).

## Unreleased

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
