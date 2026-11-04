# GNNs in NCA

Research code for testing whether graph neural cellular automata (NCA) data can
improve a transformer's initialization **before language pre-training**.
We compare graph, grid and shuffled synthetic trajectories with text controls.
The GNN is a frozen random-rule data generator, not a component inside the LM.

**Latest stage: the seen-example fitting target also fails.**
[Stage 19](docs/results/fit_audit_v19_run1/REVIEW.md) evaluates the five unchanged
stage18 endpoints on a frozen, balanced128-example training probe. Every arm
gets0/128 exact answers and misses the semantic target on both seen and held-out
cases. Seen node accuracy is23.5–28.7%, while held-out is20.6–21.7%. Aggregate
training loss fell, but gold-prefix digit prediction remains weak; held-out
generalization alone cannot explain all observed failure. No unique cause is
identified. All213 root and standalone tests pass and five endpoints/640 new answers replay
exactly. No model update occurs. See the [protocol](docs/FIT_AUDIT_V19_PROTOCOL.md)
and [usage](docs/FIT_AUDIT_V19_USAGE.md); all earlier endpoints/gates remain intact.

**Previous stage: exploratory ID acceleration measured; all arms are right-censored.**
[Stage 18](docs/results/accel_v18_run1/REVIEW.md) freezes a fresh shared ID
sample, identical 600-update training for all five arms, seven learning-curve
checkpoints and a per-cell non-source advantage over a training-only marginal
baseline. No arm reaches the target, so time-to-target ratios remain undefined.
All final exact scores are 0/128; final node accuracy is 20.6–21.7%. This is
an exploratory censored result, not evidence of graph equivalence or impossibility.
Original T4 PPT/generation/language costs are retained alongside new CPU costs;
mixed-hardware sums are component accounts, not controlled throughput.
All 208 root and standalone tests pass; all 35 checkpoints/4,480 predictions and teacher scores
replay exactly. Read the [protocol](docs/ACCEL_V18_PROTOCOL.md) and
[usage](docs/ACCEL_V18_USAGE.md). Earlier calibration/readiness failures and the
separate confirmation/OOD gates remain unchanged.

**Earlier stage: the 64-example cache fits; full held-out competence still fails.**
[Stage 17](docs/results/cache_v17_run1/REVIEW.md) runs all five original
pre-adaptation starts for the same 600 updates on a frozen nested 64-example
cache. Every arm gets **64/64 cache answers and 0/128 held-out answers**. Held-out
NLL improves over the 16-example cache for every arm, while the 512-example
reference retains lower NLL and higher node accuracy in every arm. Graph's node
score is 17.1%; its prefix control reaches 18.7% in this cohort. No graph-specific
acceleration follows. All 20 new checkpoints/1,920 predictions and teacher scores
replay exactly; all 199 root and standalone tests pass. Both imported references use the SAME
common16/dev128 cases; unmeasured small16-on-extra48 cases are absent, not zero.
Read the [protocol](docs/CACHE_V17_PROTOCOL.md) and
[usage](docs/CACHE_V17_USAGE.md). OOD and synthetic readiness remain gated.

**Earlier stage: retained-output error audit completed.**
[Stage 16](docs/results/error_audit_v16_run1/REVIEW.md) rescores 1,440 saved
endpoint outputs, with all five arms, both cache-size variants and all task/size
cells retained. Small-cache held-out node accuracy stays only 16.8–19.8% even
within valid vectors; teacher-forced digit accuracy is 19.2–20.6%. Source and
one-hop errors also remain. Formatting failure alone cannot explain the endpoint.
No new fitting or model inference occurs; OOD and synthetic readiness stay gated.
See the [protocol](docs/ERROR_AUDIT_V16_PROTOCOL.md),
[usage](docs/ERROR_AUDIT_V16_USAGE.md) and explicit
[data-regeneration disclosure](docs/ERROR_AUDIT_V16_REGENERATION_NOTE.md).

**Earlier stage: small-cache fitting control passes; held-out performance remains poor.**
[Stage 15](docs/results/overfit_v15_run1/REVIEW.md) restarts all five original
language endpoints and changes only the training cache, 512 to 16 examples, at
the same 600-update procedure. Every arm reaches **16/16 correct cache answers**;
the verified 512-cache reference has 0/16 on those same examples. Each small-cache
example is presented 300 times. Actual token/padding differences are recorded.
All new models still have **0/128 development answers**; validity drops to
51.6–66.4% and node accuracy to 8.6–12.3%, below the larger-cache reference.
This demonstrates fitting capability on the selected set, not unseen-graph
algorithm execution or GraphNCA acceleration. All 184 root tests pass; 20 checkpoints and 960 new predictions replay exactly.
The same-cache reference, actual exposure and public-only reproduction pass.
OOD and original synthetic readiness remain gated. See the
[protocol](docs/OVERFIT_V15_PROTOCOL.md) and [usage](docs/OVERFIT_V15_USAGE.md).

**Previous stage: matched training/development error audit completed.**
[Stage 14](docs/results/id_audit_v14_run1/REVIEW.md) evaluates the five saved final
models with matched 128-example probes (32 per task/size cell). Every arm has
**0/128 exact answers on both seen training and development examples**. Training
node accuracy is 29.6–32.9%; development is 18.9–21.6%. Teacher-forced digit-token
accuracy remains weak even with correct earlier answers supplied, while other
tokens are nearly perfect. Seen-example competence remains unresolved alongside
lower held-out performance. This does not identify a unique cause. There are
zero new scientific optimizer updates; OOD and original readiness remain gated.
All 176 local tests pass; all 1,280 audit predictions and teacher-forced metrics
replay exactly. Public-only probe/error reproduction passes too. See the [protocol](docs/ID_AUDIT_V14_PROTOCOL.md) and
[usage](docs/ID_AUDIT_V14_USAGE.md). The proposed next fit is a separately frozen
small-cache overfit control with other training settings held fixed.

**Previous stage: matched ID adaptation completed; task criterion fails.**
[Stage 13](docs/results/adaptation_v13_run1/REVIEW.md) gives all five retained v4
language models the same 600-update algorithmic-adaptation procedure and actual
batch order. Each receives 562,855 input tokens and 96,016 answer/EOS targets.
All final models have **0/128 exact ID answers**, despite 98.4–100% valid vectors.
Formatting is learned, but the frozen graph-task criterion is not met. OOD is
skipped, all arms/endpoints remain, and original random-rule readiness still
fails. This is local CPU interface/trainability calibration, not a new language
stage or graph-specific acceleration claim. All 168 local tests pass; 20
checkpoints reproduce 2,560 answers exactly. Public scoring also reproduces
without private weights. See the
[protocol](docs/ADAPTATION_V13_PROTOCOL.md) and
[usage](docs/ADAPTATION_V13_USAGE.md).

**Previous stage: out-of-domain graph evaluation is implemented.**
[Stage 12](docs/results/ood_v12_run1/REVIEW.md) adds CLRS-Text-inspired BFS/Dijkstra
size, graph-family and joint shifts, with relabel/order/intervention controls.
Independent graph oracles verify all 5,760 generated examples. In a frozen
zero-shot calibration, five retained v4 language checkpoints each score 0/32
valid answers: they were never taught this task interface. All 160 predictions
replay exactly. This is a formatting floor, not a graph-transfer no-effect test.
The local suite passes 160 tests. See the [protocol](docs/OOD_V12_PROTOCOL.md)
and [runnable usage guide](docs/OOD_V12_USAGE.md). No new scientific fitting,
Gmail reply or GPU job occurred. Private correspondence stays excluded.
A meaningful learned OOD comparison first needs matched ID task competence,
stronger synthetic readiness and separately frozen fresh confirmation.

**Previous stage: fresh-phase weighted confirmation passes.**
All 24 CPU models completed the fixed 1000-update budget: both 512/4096 caches,
both standard/weighted objectives and coherent/shuffled controls from three
fresh paired starts. New data excludes 4,926 historical joint phase classes;
all 128 intervention pairs were reserved outside training/development.
The weighted primary confirmation **passes** (3/3 paired seeds); standard passes 1/3. The broader both-objective criterion **fails**.
Read the [stage 11 review](docs/results/confirmation_v11_run1/REVIEW.md) and
[protocol](docs/CONFIRMATION_V11_PROTOCOL.md). All 54 full, 108 fit-probe and 27
intervention cases replay exactly; the local suite passes 146 tests.
This is one known binary rule and one fresh split, not a language-learning
speedup or GraphNCA result. Language remains gated. Stage 10's failed broad
primary remains intact; its historical derived CSV has a documented
[column caveat](docs/STAGE10_CSV_CAVEAT.md), with correct JSON/review decisions.
Original random-rule checkpoints still fail
[stage 5 readiness](docs/results/readiness_v5_run1/REVIEW.md).
Historical notebooks retain their original gates.

**Latest GPU result:** the returned **version 4 experiment** completed all seven arms
across six fresh paired cohorts on a Tesla T4. Both predeclared graph-advantage
tests fail: graph/grid has five token ties and one loss; graph/prefix-matched
shuffle has four ties, one win and one loss. Graph uses about 13.4% fewer
language tokens than scratch but about 16.6% more recorded cumulative time.
See [the v4 review and next steps](docs/results/transfer_v4_run1/REVIEW.md).
All 78 selected development checkpoints and six actual starting states passed
independent CPU replay. A new post-hoc causal frequency predictor beats every
structured PPT model on held-out synthetic patch loss, so the original gate
does not establish dynamics-specific learning. No graph-specific acceleration
or cost recovery is demonstrated. Earlier [v3](docs/results/transfer_v3_run1/REVIEW.md)
and [v2](docs/results/transfer_v2_run1/REVIEW.md) remain separate evidence.
The [results ledger](RESULTS.md) collects current measurements, mathematical
verification and historical evidence in one place.
The detailed [research presentation](docs/presentation/GPU_NCA_Research_Detailed_v8.pptx)
and its [slide/source index](docs/presentation/README.md) cover the theory,
proofs, implementation, notebooks, visualizations and reviewed results through
stage 8 (20 October 2026), including the negative weighting outcome.
The updated [IEEE template paper through stage 19](docs/paper/transactions_2026_11_02/README.md)
adds fresh confirmation, graph-task adaptation and the latest fitting audit,
with the separate scientific-transfer numerical foundation. It includes the
Word article, mathematical LaTeX companion, retained-result figures and provenance.

The accompanying [research paper and editable LaTeX project](docs/paper/README.md)
cover the completed work through stage 9, with source-grounded mathematics,
reviewed results and explicit verification limits.
Graph rewiring was introduced in v3, retaining grid and shuffled controls. The historical
`experiment.py` still implements **matched-total-budget pilot 1**.
The first GPU pilot did not demonstrate a graph benefit. Its scratch model saw
1000 text updates; NCA arms saw 100 synthetic + 900 text updates. The language
stage and scheduler confounds prevent interpreting that result as a clean test
of equal-language-stage pre-pre-training transfer.

Start with [research context](RESEARCH_CONTEXT.md), [methodology audit](METHODOLOGY_AUDIT.md),
and [historical GPU evidence](docs/results/pilot1/REVIEW.md). V3 has three paired
model/data/topology replicates, v4 has six fresh cohorts, and v2 used one shared synthetic cache. No general acceleration,
reasoning improvement or successful reproduction of the reference paper is claimed.

## Stage 10: initial-condition diversity

The separate `diversity_v10/` study expands the training cache from 512 to 4096
nested, jointly phase-disjoint initial conditions while retaining the exact
original rows, held-out data and eligible neighbor interventions. Standard and
sixfold-weighted objectives each retain coherent and prefix-shuffled arms.
Twelve new 1000-update models are compared with twelve verified imported controls.
Read the [frozen protocol](docs/DIVERSITY_V10_PROTOCOL.md).
The [completed review](docs/results/diversity_v10_run1/REVIEW.md) preserves the
objective-specific result, failing standard seeds, all controls and actual coverage.

The only training treatment is available initial-condition diversity. Uniform
sampling with replacement changes actual examples and repetition at fixed
updates/tokens; the same seed/draw count does not mean identical sampled rows
across cache sizes. Actual exposure counts and probe masks are retained.
Both objective strata must satisfy the predeclared all-seed criteria for the
study-wide primary. No result automatically enables a language stage.

Repetition requires the private verified stage 7/8/9 run folders and a fresh
output directory; public compact evidence excludes arrays and checkpoint weights:

```text
python scripts/run_logged.py --label diversity-repeat -- python -m diversity_v10.run all --out runs/diversity_repeat
```

## Stage 9: first-transition fit and generalization

The inference-only `fit_v9/` diagnostic retains initial, standard/weighted and
coherent/shuffled models across all three existing splits and both data versions.
It reuses the exact first-patch scorer and separates teacher-forced first-transition
patch and marker predictions. No optimizer update, new dataset or language stage
is introduced. This is inspected development evidence, not fresh confirmation.
Read the [completed review](docs/results/fit_v9_run1/REVIEW.md) and
[frozen protocol](docs/FIT_V9_PROTOCOL.md).

Repeating it requires the verified private stage 7/8 data and checkpoint folders.
Use a fresh output directory; the public compact evidence excludes raw arrays
and weights:

```text
python scripts/run_logged.py --label fit-repeat -- python -m fit_v9.diagnostic all --out runs/fit_repeat
```

The next candidate treatment varies training initial-condition diversity alone,
holding objectives, rule, architecture and update/token budget fixed with all
controls. It needs a separate frozen protocol and has not been run. Language
training remains gated.

## Stage 8: first-transition loss weighting

The separate `loss_weight_v8/` experiment changes only the first transition's
normalized loss weight from one to six, giving it half the training loss.
It retains the stage 7 spatial rule, exact cached inputs, initial weights,
batch order, backbone and 1000-update endpoint. Six new weighted models are
compared with six verified imported standard-loss checkpoints; coherent and
prefix-shuffled training controls remain paired for every seed. Read the
[frozen protocol](docs/LOSS_WEIGHT_V8_PROTOCOL.md).
The [completed review](docs/results/loss_weight_v8_run1/REVIEW.md) records the
negative result, all controls, per-query scores and whole-trajectory regression.

A repeat requires the private verified `runs/interaction_v7_run1` data and
checkpoints and a fresh output directory. The public compact evidence alone
does not include those arrays/weights. This repeats the same inspected data;
fresh confirmation needs its own declared split and protocol.

```text
python scripts/run_logged.py --label loss-weight-repeat -- python -m loss_weight_v8.run all --out runs/loss_weight_repeat
```

The frozen primary requires >=0.01 nats improvement in original AND intervened
first-patch NLL for all three coherent seeds, retaining trajectory readiness.
Final checkpoints are evaluated without selection. Language remains disabled.

## Stage 7: matched spatial-interaction bridge

The separate `interaction_v7/` experiment adds left/right neighbor parity to
the complement rule. The new reference and spatial models share initial grids,
weights, minibatch order and all v6 model/representation/training settings.
Their common new split excludes phase overlap under both rules; the historical
v6 endpoint is not used as the controlled reference. Twelve CPU models each
receive the fixed 1000-update budget. Read the
[main protocol](docs/INTERACTION_V7_PROTOCOL.md) and separately frozen
[first-transition diagnostic](docs/INTERACTION_V7_FIRST_TRANSITION.md).
The [completed review](docs/results/interaction_v7_run1/REVIEW.md) retains all
primary and supplementary results, including the remaining first-transition errors.

The first-transition check prevents interpreting two-frame recall alone as
neighbor-rule learning. It scores the first target patch and a neighbor-bit
intervention before repeated frames are available. A repeat uses one fresh
folder and freezes both plans before fitting:

```text
python scripts/run_logged.py --label interaction-repeat -- python scripts/run_interaction_v7.py --out runs/interaction_repeat
```

Primary readiness, the own-patch-table qualification and the supplementary
first-transition result remain separate. Language training stays disabled.

## Stage 6: controlled transformer trainability

`trainability_v6/` tests the actual full v4 backbone on the simple binary
cell-complement rule used by the measurement control. The alphabet, lattice and
context are deliberately simplified. Three paired seeds each train coherent
and prefix-shuffled arms for 1000 updates. Final checkpoints are assessed on
phase-disjoint initial conditions after all arms finish, with the same two
0.01-nat readiness requirements. This stage has no language-training command.
Read the [frozen protocol](docs/TRAINABILITY_V6_PROTOCOL.md) and
[completed CPU review](docs/results/trainability_v6_run1/REVIEW.md) for its limits.

This diagnostic runs locally on CPU with PyTorch and NumPy; no GPU, account,
corpus download or external pretrained weights are needed. For a repeat, retain
the original run and select a fresh directory:

```text
python scripts/run_logged.py --label trainability-repeat -- python -m trainability_v6.run all --out runs/trainability_repeat
```

The command freezes inputs before fitting and replays every initial/final
assessment cell after training. Existing random-rule checkpoints retain their
failed stage 5 readiness regardless of this simpler capability check's outcome.

## Stage 5: checkpoint-only readiness

The separate `readiness_v5/` package uses the retained v4 audit, cached held-out
trajectories and selected PPT checkpoints. It freezes inputs and code before
48 coherent/matched-shuffled comparisons, validates a known-rule positive and
frequency-only negative control, and requires both advantages to reach 0.01
nats per patch in every structured cohort. It has no training command.

With the retained private v4 artifacts at their documented location, use a
**new** output directory for a repeat; never overwrite the frozen first run:

```text
python scripts/run_logged.py --label readiness-repeat -- python -m readiness_v5.run all --out runs/readiness_repeat
```

The defaults and reproduction limits are in the protocol. Public evidence
contains hashes and scores, not weights or corpora. A future language runner
must call `readiness_v5.run.require_readiness(review, out)` before accepting
the returned checkpoint identities. That guard currently rejects all 12
structured cohorts; it does not bypass the separate study-design step.

## Reproduce historical version 4: stronger controls and fresh replication

Upload **[kaggle_nca_transfer_v4.ipynb](kaggle_nca_transfer_v4.ipynb)** to Kaggle,
enable one GPU and Internet, then Run All to reproduce the completed experiment.
Its objective is to test whether
ordered graph trajectories accelerate language learning beyond grid trajectories
and matched frequency/formatting structure. Six fresh paired cohorts enter
identical language stages, retaining both original shuffled controls and adding
two controls that preserve the exact supervised-prefix counts and initial frame.
The model, topology, serializer, rule filter and stage budgets retain v3 settings.

The [v4 protocol](docs/TRANSFER_V4_PROTOCOL.md) fixes two comparisons at the same
paired scratch-endpoint target before training. All cohorts, ties and missed
targets remain in the result. The run archives its analysis plan and verifies
checkpoints, controls and raw-curve analysis. This is a prospective validation
replication/mechanism study; the official test remains reserved.

The returned run recorded about **6.92 hours across saved stage timers**, plus
preparation, verification and export work. The notebook prints a fresh estimate
on the assigned GPU; that estimate is not a runtime guarantee. Completion prints
`WORKFLOW FINISHED: complete` or `WORKFLOW FINISHED: stopped_at_synthetic_gate`
and exports `nca_transfer_v4_results_<timestamp>.zip` (potentially several GB).
The [returned review](docs/results/transfer_v4_run1/REVIEW.md) recommends a
checkpoint-level order-sensitivity diagnostic before another full GPU suite.

## Repeat the graph versus grid experiment, version 3

Upload **[kaggle_nca_transfer_v3.ipynb](kaggle_nca_transfer_v3.ipynb)** to Kaggle,
enable one GPU and Internet, then Run All. This stage tests graph-NCA PPT
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
The first returned GPU outcome is documented in the v3 review above. Completion
does not by itself establish a graph or dynamics advantage.

## Previous run: pre-pre-training transfer, version 2

Upload **[kaggle_nca_transfer_v2.ipynb](kaggle_nca_transfer_v2.ipynb)** to Kaggle,
enable one available GPU and Internet, then Run All to repeat v2. Later controlled
experiments are preserved above; consult the latest review for next research steps.
`kaggle_gnn_nca_pilot.ipynb` is historical pilot 1.

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

Every Prod update includes new and existing Lean proofs, supporting documentation
and retained verification evidence in `proofs/lean/`, together with LaTeX sources
and mathematical documentation in `docs/theory/`. Both trees are selected
recursively by default; add any new locations to `publication.json`. Local build
caches and private artifacts remain excluded. Copying these materials does not
constitute a new proof check or document compilation.

## License and references

Original project code and documentation: **Apache License 2.0**, see
[LICENSE](LICENSE) and [NOTICE](NOTICE). The pinned upstream snapshot remains
MIT-licensed; see [third-party notices](THIRD_PARTY_NOTICES.md).

Scientific reference: [Training Language Models via Neural Cellular Automata, v1](https://arxiv.org/html/2603.10055v1),
with [author code](https://github.com/danihyunlee/nca-pre-pretraining).
This project is an independent small-scale graph-topology study.
