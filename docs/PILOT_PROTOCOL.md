# Historical pilot 1 protocol

This preserves the original implementation description. The runner still uses
this fixed-total-budget design. Read `../METHODOLOGY_AUDIT.md` for its
limitations; this is not the corrected equal-language-stage protocol. Provider
quota statements below were planning context, not current allocation promises.

# Does graph-NCA pre-training make a small language model faster?

**Version 1 protocol note (17 October 2026):** this runner is a fixed-total-budget
replacement pilot, not a clean reproduction of the paper's pre-pre-training
transfer comparison. Read [the methodology audit](../METHODOLOGY_AUDIT.md) and
[retained research context](../RESEARCH_CONTEXT.md) before another run. The corrected
equal-language-stage protocol is documented there; it has not yet been
implemented in this version's Kaggle notebook. Existing results are preserved.

This is an experiment, not a promised positive result. The first deliverable is a
small causal transformer, reproducible checkpoints, and a Kaggle notebook that
can support or reject a narrowly stated speed claim.

## Assessment of the proposed plan

The cloud-first approach and matched-budget comparisons are sensible. These parts
need correction:

1. **NCA transfer and GNN benefit are different hypotheses.** Grid NCAs are already
   local message-passing systems. Re-expressing the same grid update as a GNN does
   not by itself introduce a new model or guarantee a speedup. A test checks that
   the grid message-passing implementation exactly matches a convolutional NCA.
2. **A short GPU runtime and a complete $2 benchmark are estimates requiring a
   measurement.** They depend on hardware, model, total arms, seeds, evaluation,
   and preparation. Do not repeat the quoted prices or runtime as established facts.
3. **Cloud allocations vary.** Kaggle documents a weekly GPU quota around 30 hours,
   subject to availability; check the account's current quota and accelerator.
   Colab explicitly does not guarantee GPU type, runtime, or availability.
4. **Match the right resource.** Equal architecture, batch, context, and optimizer
   steps roughly match transformer arithmetic. They do not match total cost when
   synthetic-data creation is extra. This pilot records that extra cost.
5. **Noise is a weak control.** Shuffle the same graph trajectories while retaining
   their per-trajectory patch frequencies and frame marker locations.
6. **One trained model cannot establish a comparative claim.** Reuse one model
   architecture across arms and paired seeds; use held-out text and a declared
   validation-loss target. A tiny pilot does not establish reasoning improvements.

Relevant prior work is [Training Language Models via Neural Cellular Automata
(2026)](https://arxiv.org/abs/2603.10055), with [author code](https://github.com/danihyunlee/nca-pre-pretraining).
It reports successful synthetic-to-language transfer at substantially larger
scales. This project is a smaller graph-topology experiment, not a reproduction.

Provider sources: [Kaggle GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage),
[Kaggle notebooks](https://www.kaggle.com/docs/notebooks), and
[Colab resource policy](https://research.google.com/colaboratory/faq.html).

## First experiment

Default model: **1,857,024 parameters**, four transformer blocks, width 192, four
attention heads, context 128, tied embeddings, fp32, no dropout. The language
tokens are raw bytes, so reported perplexity is **byte perplexity**, not BPE word
or token perplexity. All stages use the same 274-way output vocabulary.

| Arm | First 100 steps | Next 900 steps | Boundary token embeddings |
| --- | --- | --- | --- |
| `scratch` | Target training text | Target training text | Preserved |
| `reserve_text` | Separate text reserve | Target training text | Preserved |
| `grid_nca` | Periodic grid NCA | Target training text | Restored to initial values |
| `graph_nca` | Rewired graph NCA | Target training text | Restored to initial values |
| `shuffled_graph` | Shuffled graph trajectories | Target training text | Restored to initial values |

Each arm trains for **4,096,000 total tokens per seed**. Three paired model seeds
make 15 runs and **61,440,000 total training tokens** across the suite. The synthetic
stage uses 409,600 sampled tokens per arm/seed from a cached 294,912-token corpus;
sampling with replacement is intentional and recorded. The source text is also
reused; this is an inexpensive pilot, not a large unique-token pre-training run.

AdamW, clipping, and the stage-specific warmup/cosine schedules are shared. Every
arm restarts Adam and its schedule at step 100, including scratch. Thus scratch
uses target text throughout but has a declared optimizer boundary. It does not
suffer a forced embedding reset. The separate-text arm preserves byte semantics.
Synthetic arms reset the tied input/output table because their symbols have no
byte meaning. Position embeddings and the backbone transfer. Grid vs graph vs
shuffle use identical transfer policies, isolating their data differences.

### What is the GNN here?

An 8x8 graph has one binary state per node and eight typed incoming edges. A shared
neural rule maps the node and neighbor states through a 9-to-16-to-2 MLP. The rule
is randomly sampled and frozen independently for each trajectory. This is a
discrete neural graph cellular automaton; it is **not a GNN trained on a separate
task** and the GNN is **not inside the language model**.

The grid reference uses the eight periodic Moore neighbors. The graph variant
swaps endpoints within edge types at a declared 25% attempted rewiring rate,
preserving both incoming and outgoing degree and forbidding self/duplicate
neighbors. Actual changed-edge fraction and the topology checksum are logged.
Both variants execute the same message-passing code with the same sampled rules
and initial states. This isolates topology, not implementation efficiency. It
does **not** benchmark a tuned grid convolution against a sparse GPU GNN kernel.

Each rollout has 16 frames. Binary 2x2 patches map bijectively to 16 synthetic
tokens, with two frame markers; these IDs are separate from the 256 byte IDs.
Sampling never joins windows across distinct trajectories. Graph patches retain
the original node order, so serialization favors the grid and can disadvantage
rewiring. There is one cached graph and one data seed: independent graphs and
data seeds belong in a confirmatory follow-up.

Rules are sampled without selection; gzip ratios, cell-change rates, patch
coverage, and final fixed-point fractions reveal collapsed or trivial dynamics.
These random rules are not validated physical laws. A graph effect could be
mediated by changed sequence complexity; a follow-up should match complexity
bands as well as update rules and degree.
This binary-state pilot differs from the paper's alphabet, complexity selection,
tokenizer, model, and budget. A negative result here does not reject all NCAs.
Do not tune generator parameters against the held-out test loss.

### Data and evaluation

The default [Tiny Shakespeare corpus](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
is downloaded once and SHA-256 recorded. Byte ranges split into 20% separate text
reserve, 60% target train, 10% validation, 10% test. Ranges do not overlap, but
Shakespeare naturally repeats phrases: no claim of semantic deduplication is made.
This is a same-domain reserve baseline, not general web pre-training.

Validation uses a fixed non-overlapping subset, identical across arms and seeds.
The test set is evaluated only at the final step. Checksums cover all cached
arrays, graph edges, checkpoints, and experiment source. The smoke configuration
only checks execution; it has a much smaller architecture and six updates.

## Run on Kaggle

1. Import **`kaggle_gnn_nca_pilot.ipynb`** into a private Kaggle notebook.
2. Enable an available **GPU** accelerator and **Internet** for the corpus download.
   The code uses **one GPU**, even if Kaggle allocates two T4s. It has no paid API.
3. Run all cells. The notebook contains the source, configuration, and tests; no
   GitHub repository or additional uploaded code is required. It runs validity
   checks, a GPU smoke pass, the three-seed pilot, and checkpoint verification.
4. Use **Save Version / Run All** and retain notebook outputs. Download the results
   archive from the final cell; saving a local notebook does not preserve an
   unsaved remote runtime's files.

Runtime is intentionally not promised. The default is fp32 and deterministic
algorithms, favoring auditability over peak GPU throughput. Inspect the actual
timings before increasing the budget. A failed or interrupted run keeps completed
arm artifacts but has no automatic resume; choose a fresh output directory for a
new run. No cloud job has been launched merely by creating this project.

## Run locally

Install the dependencies in an isolated environment. A CPU works for checks:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe experiment.py all --config configs/smoke.json --data data/smoke --out runs/smoke --device cpu
.venv\Scripts\python.exe verify.py --data data/smoke --out runs/smoke
```

For the full pilot on a GPU:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python experiment.py all --device cuda
python verify.py --data data/pilot --out runs/pilot --device cuda
python plot_results.py --out runs/pilot
```

Use `--text /path/to/corpus.txt` to avoid downloading. Reuse a data directory only
with its exact original configuration. New runs require a fresh `--out` directory
to prevent accidental replacement or mixing of results. `prepare`, `run`, and
`report` can also be invoked separately.

## What would count as faster?

The predeclared target is **2.5 validation nats per byte**. A run records the first
evaluation reaching it, without interpolation. If either member of a paired
comparison never reaches the target, its speedup remains **unresolved**, not zero
or a fabricated estimate. Text baselines can reach the target in their first
stage. A synthetic arm becomes eligible after restoring its byte embeddings.

Report all four views:

- **Target-text tokens to target:** sample efficiency; excludes synthetic exposure.
- **Total training tokens to target:** includes synthetic and separate-text steps.
- **Training seconds to target:** synchronized GPU time for batches, forward,
  backward, and optimizer updates; includes pre-training.
- **Pipeline seconds to target:** adds cold synthetic generation, serialization,
  shared text preprocessing, data loading, model setup, and evaluation overhead.
  A shared three-update disposable warmup avoids charging lazy kernel/optimizer
  startup only to the first arm; its cost is conservatively added to every arm.
  Full synthetic preparation is conservatively charged per run, even though the
  suite reuses its cache. Excludes network download, package/environment startup,
  test evaluation and final checkpoint writing after training, and cloud queue
  time. It is a defined pipeline metric, not total cloud billing time.

For each pair, `speedup = baseline cost / graph cost`; values above one favor the
graph. Inspect graph vs **scratch** for practical transfer, graph vs **grid NCA**
for added topology benefit, and graph vs **shuffled graph** for useful order.
A faster curve from only the final language stage is insufficient if the extra
pre-training and generation cost erase the gain.

`report.json` contains paired differences, censored outcomes, and exploratory
bootstrap intervals. Three seeds provide a pilot signal, not strong statistical
proof; one seed has no uncertainty interval. Do not average only successful
seeds into a universal speed claim. `6PD` is labeled an approximate transformer
training FLOP proxy; it excludes attention corrections and generator FLOPs.
Equal final tokens do not imply equal complete pipeline compute.

Retain `summary.csv`, `curves.svg`, `report.json`, `environment.json`, the data
manifest, `metrics.jsonl`, `model.pt`, and `verification.json`. Only claim a pilot
speedup if paired GPU times favor graph at the same declared loss, the effect
holds across seeds, and the final test losses support comparable model quality.
Then repeat with at least five independent seeds/graphs, a larger corpus, and
an untouched confirmation test set before making a broader research claim.
