# GNNs in NCA

Research code for testing whether graph neural cellular automata (NCA) data can
improve a transformer's initialization **before language pre-training**.
We compare graph, grid and shuffled synthetic trajectories with text controls.
The GNN is a frozen random-rule data generator, not a component inside the LM.

**Status:** the returned **version 2 GPU transfer experiment** completed on a
Tesla T4. Grid-NCA pre-pre-training reached the paired no-PPT endpoint in
1600–1800 language updates instead of 2000 across three seeds. Shuffled NCA
achieved comparable gains, so this does not establish a dynamics-specific benefit.
Recorded cumulative time was about 13% higher for grid NCA at that reference.
See [the v2 results review](docs/results/transfer_v2_run1/REVIEW.md).
The [results ledger](RESULTS.md) collects current measurements, mathematical
verification and historical evidence in one place.
The detailed [research presentation](docs/presentation/GPU_NCA_Research_Detailed_v2.pptx)
and its [slide/source index](docs/presentation/README.md) cover the theory,
proofs, implementation, notebooks, visualizations and reviewed results.
Graph rewiring has not been tested in v2. The historical
`experiment.py` still implements **matched-total-budget pilot 1**.
The first GPU pilot did not demonstrate a graph benefit. Its scratch model saw
1000 text updates; NCA arms saw 100 synthetic + 900 text updates. The language
stage and scheduler confounds prevent interpreting that result as a clean test
of equal-language-stage pre-pre-training transfer.

Start with [research context](RESEARCH_CONTEXT.md), [methodology audit](METHODOLOGY_AUDIT.md),
and [historical GPU evidence](docs/results/pilot1/REVIEW.md). The v2 result is a
small development experiment with one shared synthetic cache. No general acceleration,
reasoning improvement or successful reproduction of the reference paper is claimed.

## Next run: graph versus grid pre-pre-training, version 3

Upload **[kaggle_nca_transfer_v3.ipynb](kaggle_nca_transfer_v3.ipynb)** to Kaggle,
enable one GPU and Internet, then Run All. This next stage tests graph-NCA PPT
against grid-NCA PPT, with shuffled controls for both and a no-PPT control.
All five enter the same 2000-update language stage after 1000 PPT updates
(zero for scratch). Three paired model/data/topology replicates use common
frozen grid/graph rules selected with the same complexity filter.

The transformer and language settings match v2. Save actual common starting
weights; verify transferred backbones, common heads and all stage checkpoints.
Language training starts only if every grid and graph seed passes the synthetic
gate. The official test remains untouched. Based on the returned T4 run, allow
roughly 2.5–3 hours plus variable preparation/export overhead; the notebook
measures a fresh estimate before training.

Completion prints `WORKFLOW FINISHED: complete` or
`WORKFLOW FINISHED: stopped_at_synthetic_gate`, followed by the download link for
`nca_transfer_v3_results_<timestamp>.zip`. Read [the v3 protocol](docs/TRANSFER_V3_PROTOCOL.md).
V3 has local software checks; a new GPU outcome has not been measured.

## Previous run: pre-pre-training transfer, version 2

Upload **[kaggle_nca_transfer_v2.ipynb](kaggle_nca_transfer_v2.ipynb)** to Kaggle,
enable one available GPU and Internet, then Run All to repeat v2. For the next
graph experiment use v3 above; `kaggle_gnn_nca_pilot.ipynb` is historical pilot 1.

Default: 1000 synthetic updates followed by **2000 equal language updates**,
three paired seeds, with zero synthetic updates for scratch. Categorical NCA,
rule filtering, burn-in, a small RoPE/RMSNorm/SwiGLU transformer, and fresh
WikiText/GPT-2 data replace the first pilot's simplified setup. The confirmation
test split stays untouched by default.
The shared backbone has a 4.58M-parameter synthetic model and a 12.31M-parameter
language model because the two vocabularies require different embedding tables.

The notebook measures a timing estimate on your actual GPU before training.
It saves PPT checkpoints, checks unseen-rule prediction against fitted controls,
and starts the language comparison only if the declared synthetic gate passes.
Language-stage acceleration and cumulative training costs are reported separately.

The last cell exports `nca_transfer_v2_results_<timestamp>.zip`. It reports either
`complete` (language benchmark completed) or `stopped_at_synthetic_gate`
(diagnostics completed; no language benchmark). Both archives are useful to
review; neither status automatically means the hypothesis succeeded.

Read [the v2 protocol](docs/TRANSFER_V2_PROTOCOL.md). Local CLI dependencies are
in `requirements-v2.txt`; additional checked versions are in
`requirements-v2-tested.txt`. Offline checks do not require network packages:

```bash
python scripts/run_logged.py --label v2-scientific -- python -m unittest discover -s tests -p test_transfer_v2.py -v
python scripts/run_logged.py --label v2-integration -- python scripts/smoke_transfer_v2.py
```

## Quick start

The separate [mathematical note](docs/theory/README.md) develops the graph-NCA
concept in LaTeX, including proofs, transfer conditions, counterexamples and
worked numerical derivations.
The companion [Lean proof folder](proofs/lean/README.md) contains 44 checked
statements, a [LaTeX coverage map](proofs/lean/PROOF_MAP.md), reproducible proof
sources and retained verification evidence. Its coverage is partial: the full
entropy, spectral and smoothness arguments remain informal.

Python **3.12** is the checked interpreter. A CPU is sufficient for tests and
smoke checks. Install dependencies in a virtual environment from this directory.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/run_logged.py --label unit-tests -- python -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/run_logged.py --label offline-smoke -- python scripts/smoke_check.py
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_logged.py --label unit-tests -- python -m unittest discover -s tests -v
python scripts/run_logged.py --label offline-smoke -- python scripts/smoke_check.py
```

`requirements.txt` defines supported dependency ranges. `requirements-tested.txt`
records the direct versions actually checked on Windows CPU. CUDA requires a
suitable PyTorch build; do not infer GPU reproducibility from CPU checks. Do not
use Python `-O`, which disables assertions in historical scientific checks.

The offline smoke check uses generated fixture text in a temporary directory.
It exercises all five arms, cache checks, training, reporting, checkpoint
verification and plot export without a download. Temporary artifacts are removed;
full captured output is retained in `logs/checks.jsonl`. Fixture losses are not
research results. CI is configured for Python 3.12 on Linux and Windows; a CI
configuration is not evidence that those remote jobs have run.

## Run the historical pilot

Use a fresh data directory and output directory. To retain a CPU smoke run with
the default corpus (first preparation downloads Tiny Shakespeare):

```bash
python experiment.py all --config configs/smoke.json --data data/smoke_001 --out runs/smoke_001 --device cpu
python verify.py --data data/smoke_001 --out runs/smoke_001 --device cpu
python plot_results.py --out runs/smoke_001
```

Supply `--text /path/to/corpus.txt` to avoid downloading. The text must be long
enough for every configured split. Only reuse a cache with its original config.
`prepare`, `run` and `report` can be called separately. `prepare` rewrites its
selected cache: always choose a fresh path for changed data/configuration. Failed
runs are not automatically resumable; retain them and choose a new output path.

For the full historical GPU pilot, set deterministic CUDA configuration before
starting Python:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python experiment.py all --config configs/pilot.json --data data/pilot_001 --out runs/pilot_001 --device cuda
python verify.py --data data/pilot_001 --out runs/pilot_001 --device cuda
python plot_results.py --out runs/pilot_001
```

In PowerShell use `$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'` before those commands.
Wrap commands in `scripts/run_logged.py --label NAME -- ...` to retain their full
output. Training already writes per-evaluation metrics and environment identity.
Runtime and cloud availability must be measured; no quota or cost is promised.

### Kaggle

Import `kaggle_gnn_nca_pilot.ipynb`, select an available GPU and enable Internet
for the corpus download. It embeds the runner, scientific tests, configurations,
license and essential documentation. It uses one GPU and runs the historical
pilot. Save a version with outputs and download the generated result archive to
retain checkpoints. Check the notebook's stated protocol before spending compute.

Regenerate embedded sources with `python build_notebook.py`; check parity with
`python build_notebook.py --check`. Generation also makes a local starter ZIP,
which is excluded from source publication. Notebook GPU execution was not rerun
as part of preparing this repository.

## Model, controls and measurements

The default causal byte LM has **1,857,024 parameters**, four blocks, width 192,
four attention heads and context 128. All stages share 274 output symbols:
256 bytes, 16 binary patch symbols and two frame markers. Reported perplexity is
byte perplexity, not BPE-token or word perplexity.

| Arm | First 100 updates | Next 900 updates | Token embeddings at boundary |
| --- | --- | --- | --- |
| `scratch` | Target text | Target text | Preserved |
| `reserve_text` | Disjoint same-domain text reserve | Target text | Preserved |
| `grid_nca` | Periodic grid trajectories | Target text | Restored to initial values |
| `graph_nca` | Rewired graph trajectories | Target text | Restored to initial values |
| `shuffled_graph` | Frequency-preserving shuffled trajectories | Target text | Restored to initial values |

Every arm restarts its optimizer/schedule at the boundary. Each seed sees
4,096,000 total training tokens per arm; three paired seeds yield 15 runs.
The graph keeps eight typed incoming/outgoing neighbors per node, with no self
or duplicate neighbors. The grid uses the same message-passing implementation;
this is a topology study, not a sparse-kernel versus convolution speed benchmark.
One cached graph/data seed and fixed spatial patch ordering limit generalization.
Random rules are not filtered for useful dynamics or tested on held-out rules.

The default corpus has disjoint byte ranges: 20% text reserve, 60% target train,
10% validation and 10% test. The declared target is 2.5 validation nats/byte.
First recorded crossings are used without interpolation; unreached comparisons
remain unresolved. Reports include target-text tokens, total training tokens,
training seconds and defined pipeline seconds. Pipeline timing includes synthetic
preparation, text preprocessing, setup, training and validation, but excludes
network/package startup, cloud queues and final checkpoint writing. See the
[historical protocol](docs/PILOT_PROTOCOL.md) for exact accounting and limitations.

The next controlled protocol separates K synthetic updates from N common
language updates, skips synthetic training for scratch and matches language data
order, head initialization, optimizer and schedule. Held-out-rule synthetic
learning and boundary checkpoints must be established before a confirmatory
claim. Previously inspected test data must not be reused for selecting that claim.

## Evidence and research records

- [Validation record](VALIDATION.md): historical CPU diagnostic and current check scope.
- [Compact historical evidence](docs/results/README.md): original metrics, summaries,
  verification records, retrospective plots and SHA-256 provenance.
- [Logbook](LOGBOOK.md): changes, tests, failures, observations and inferences.
- [Check transcripts](logs/checks.jsonl): append-only machine records of commands/results.
- [Changelog](CHANGELOG.md): contributor-facing changes.

Retain full `environment.json`, data manifest, metrics, `summary.csv`, `report.json`,
plots, checkpoints and `verification.json` for actual experiments. Do not commit
raw corpora, token arrays or checkpoints. The complete historical archive remains
local; no public download location has yet been established. Compact published
metrics permit recalculation but cannot replace checkpoint/data verification.

## Repository layout and contributing

```text
experiment.py                  Historical generation/training/reporting runner
verify.py, plot_results.py     Checkpoint verification and figure export
configs/, tests/              Declared budgets and scientific/regression checks
scripts/                      Logged checks, offline smoke, publication maintenance
build_notebook.py              Standalone Kaggle notebook generator
sources/                      Pinned upstream MIT reference, not runtime code
docs/results/                 Compact evidence and provenance
.github/                      CPU CI and contribution templates
```

Historical helper scripts accept `--help`; `audit_kaggle_output.py` and
`diagnose_prepretraining.py` require the original full artifacts. `write_validation.py`
exports a new diagnostic report without replacing historical validation by default.
`fetch_reference.py` defaults to the pinned revision; do not overwrite reference
provenance as part of ordinary checks.

See [CONTRIBUTING.md](CONTRIBUTING.md). Reproducibility fixes, held-out-rule
learning diagnostics, faithful grid controls and the corrected stage protocol
are useful starting points. Keep negative results and failed tests in the logbook.

In the original research workspace, edit root sources and run:

```bash
python build_notebook.py
python scripts/sync_prod.py
python scripts/sync_prod.py --check
```

`Prod/` is the selected standalone repository. Publish its contents, including
hidden `.github`, `.gitignore` and `.gitattributes`, rather than the parent
workspace. `PUBLICATION_MANIFEST.json` inventories every selected file. In a
standalone checkout, use `python scripts/check_release.py --refresh` after
changes/tests, then `python scripts/check_release.py`. New publishable files
must be selected in `publication.json`. Ignore rules alone do not select a release.

## License and references

Original project code and documentation: **Apache License 2.0**, see
[LICENSE](LICENSE) and [NOTICE](NOTICE). The pinned upstream snapshot remains
MIT-licensed; see [third-party notices](THIRD_PARTY_NOTICES.md).

Scientific reference: [Training Language Models via Neural Cellular Automata, v1](https://arxiv.org/html/2603.10055v1),
with [author code](https://github.com/danihyunlee/nca-pre-pretraining).
This project is an independent small-scale graph-topology study.
