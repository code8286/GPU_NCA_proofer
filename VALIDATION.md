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
are configured only. No v3 GPU training has run.
