# Contributing

Contributions to reproducibility, controls, diagnostics and documentation are
welcome. Read [the research context](RESEARCH_CONTEXT.md) and
[methodology audit](METHODOLOGY_AUDIT.md) first. `experiment.py` preserves pilot 1;
`transfer_v2/` implements the corrected independent-stage transfer experiment.
Read [the v2 protocol](docs/TRANSFER_V2_PROTOCOL.md) before changing its controls.

## Setup and checks

Use Python 3.12 and a virtual environment; install `requirements.txt`. The README
gives Windows and Unix commands. `requirements-tested.txt` records the direct
versions checked here. The broader dependency ranges are not a claim that every
combination has been tested. Install an appropriate CUDA PyTorch build separately
for GPU work. CPU checks need no cloud account.

Run from the repository root:

```bash
python scripts/run_logged.py --label scientific-tests -- python -m unittest discover -s tests -v
python scripts/run_logged.py --label offline-smoke -- python scripts/smoke_check.py
python scripts/run_logged.py --label notebook-parity -- python build_notebook.py --check
python scripts/run_logged.py --label v2-offline -- python scripts/smoke_transfer_v2.py
python scripts/run_logged.py --label v2-notebook -- python build_transfer_notebook.py --check
```

The offline smoke check creates an isolated temporary run using generated text,
trains all five arms, reloads checkpoints and exports plots. It does not download
data or measure a transfer benefit. Do not use Python `-O`: historical scientific
checks use assertions.

## Propose changes

Open an issue, then submit a focused pull request with:

- The problem, resulting behavior and any changed scientific assumptions.
- Commands, hardware, outputs, failures and remaining limitations.
- An entry in `LOGBOOK.md` and `CHANGELOG.md` under `Unreleased`.
- Regression coverage for changed controls, integrity checks or measurements.

Declare budgets, embedding/optimizer policy, data/rule/topology seeds and metrics
before confirmatory runs. Report language-stage and cumulative costs separately.
Use fresh data/run directories, retain failed runs and document why they failed.
Never tune a protocol to improve an already inspected test result. Submit small
metrics and provenance; link large artifacts in an issue or release with SHA-256
checksums rather than committing corpora, checkpoints or ZIPs.

## Publication maintenance

In the original research workspace, edit root files, regenerate both notebooks
with `build_notebook.py` and `build_transfer_notebook.py`,
then run `python scripts/sync_prod.py`. `Prod/` is the standalone repository to
publish. `python scripts/sync_prod.py --check` detects drift without writing.
Finish with these sync commands directly after logged checks; wrapping a sync
in the logger appends another log entry after the copy and therefore creates drift.

In a standalone checkout, run `python scripts/check_release.py --refresh` after
changes and tests, then `python scripts/check_release.py`. The inventory includes
only distributable files. Do not publish the parent research workspace.

For release preparation, `python scripts/git_roundtrip_check.py` also checks a
temporary Git staging/checkout cycle, preserving historical byte checksums across
line-ending settings. In the research workspace pass `--root Prod`. It creates
no commit or remote and removes its temporary repository afterwards.

By submitting material for inclusion, you contribute it under Apache-2.0 unless
explicitly stated otherwise. Only submit material you may share. Preserve
third-party notices and provenance. Discuss results respectfully, include
negative evidence and make reproduction possible for other contributors.
