	# Results ledger

Updated 17 October 2026. This page summarizes reviewed evidence. Detailed
commands, failures and corrections are retained in [LOGBOOK.md](LOGBOOK.md)
and `logs/checks.jsonl`.

## Latest measurement: transfer-v2 GPU development run 1

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
are retained. The current publication update checks the saved evidence hashes;
it does not repeat training or checkpoint evaluation.

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
