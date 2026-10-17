# Version 2: acceleration through NCA pre-pre-training

The primary question is whether grid-NCA pre-pre-training produces a transformer
initialization that learns natural language **faster during a subsequent,
otherwise identical language-training stage**. A synthetic-stage diagnostic is a
prerequisite; it is not itself evidence of language transfer.

This implementation is separate from historical `experiment.py`. Pilot 1 and its
artifacts retain their original identities and limitations.

## Default stages and comparisons

Configuration: `configs/transfer_v2.json`.

| Arm | Synthetic pre-pre-training | Language pre-training |
| --- | --- | --- |
| `scratch` | None | 2000 identical language updates |
| `grid_nca` | 1000 grid-NCA updates | 2000 identical language updates |
| `shuffled_nca` | 1000 shuffled-grid updates | 2000 identical language updates |

Three paired model seeds: 17, 29, 43. The graph-topology extension is deliberately
deferred until the grid reference shows useful transfer. The grid update is
already expressed as relational message passing, with an independently tested
equivalent spatial convolution. Rewiring is not silently mixed into this test.

Effective batch: 2 sequences per microbatch, 2 gradient-accumulation microbatches,
context 1024. Each update consumes 4096 input tokens. Each synthetic arm/seed gets
4,096,000 synthetic tokens and every arm/seed gets **8,192,000 language tokens**.
There are six PPT stages and nine language stages: 98,304,000 model-input tokens
in total if the gate permits all language runs. Masked conditioning tokens are
included in exposure counts; supervised synthetic-target counts are also logged.

All language arms start at **language step zero**, with identical:

- Initial language embedding/output weights (paired by seed, explicitly tied).
- Language data-window order, effective batch, context, validation subset and N.
- Fresh AdamW state, LR schedule, weight decay and clipping settings.
- Backbone architecture and original pre-PPT backbone initialization per seed.

The two PPT arms transfer only their selected backbone state. Synthetic symbol
embeddings, Adam moments and scheduler state are not transferred. Scratch begins
with the paired untrained backbone and receives no preceding target-text updates.
Checksums of language heads, batch orders and schedules are checked across arms.

## Model and stage-specific optimization

The model uses six layers, width 192, four attention heads, RMSNorm, rotary
position encoding and SwiGLU, without dropout. It is a small Llama-style backbone,
not the paper's full-sized model or a bit-exact Hugging Face reproduction.
Backbone initialization is isolated from embedding initialization so changing
vocabulary size cannot alter the paired original backbone.

The synthetic vocabulary is 10,002: 10,000 categorical patch IDs and two markers.
The language vocabulary is GPT-2's 50,257 BPE IDs. Parameter counts differ across
stages because embeddings differ: **4,577,088 during PPT** and **12,306,048 during
language training**. Language arms are identical in size. Counts are also
recorded by the on-device calibration and run summaries.

- PPT: Adam, LR 1e-4, no weight decay, no gradient clipping. A finite global-norm
  check still stops non-finite training.
- Language: AdamW, LR 5e-4, weight decay 1e-4, gradient clipping at 1.0.
- Both: independent 10% warmup and cosine decay to 10% of their stage LR.
- Precision: fp32, deterministic PyTorch algorithms, TF32 disabled. This is an
  auditability choice, not a maximum-throughput GPU-kernel comparison.

## Categorical NCA generation

Based on the [author's generator and trainer](https://github.com/danihyunlee/nca-pre-pretraining/tree/bdd1c71e04eb2e5a2c91481faf278e95846b4431),
retained with its MIT license in the source reference directory:

- Periodic 12x12 grid with 10 categorical states, represented one-hot.
- Shared 3x3 perception with four channels, then a 16-channel pointwise layer,
  ReLU and ten next-state logits. Biases are zero; kernels use a fan-in-normal
  distribution with truncated-normal variance correction.
- Frozen randomly sampled neural rule per trajectory, low-temperature categorical
  sampling (1e-4, matching the pinned launcher).
- Ten burn-in steps before recording. Profile ten frames for rule selection.
- Each 2x2 patch maps bijectively to a base-10 integer. Frame markers bracket the
  patch sequence. A 1024-token context spans about 27 grid frames.
- Rule-profile filtering uses **0.5 < gzip ratio < 1.0**, with marker-free
  little-endian int32 patch IDs, gzip level 9. Bounds are not silently relaxed.
- Actual cached trajectories use fresh initial conditions after screening;
  their measured compression ratios can differ from the profile band.
- 4096 retained training rules and 128 retained held-out rules, using separate
  generation streams. Model validation uses the first 32 held-out rules, fixed
  before training. Rule identities, frozen rule weights, token files, scores,
  acceptance rates, collapse rates, configuration and hashes are retained.

The cache is reused across model seeds, so this is three model seeds on one
synthetic dataset, not independent data/topology replication. Synthetic sampling
uses replacement from the cached trajectories. The shuffle control preserves
each trajectory's full patch histogram and exact marker positions, while
destroying patch order. Because context is a prefix, its prefix histogram need
not equal the unshuffled prefix histogram; native training-prefix controls are
fitted separately. A more tightly prefix-matched shuffle is a future ablation.

Explicit choices relative to reference quirks: spatial axes alone are padded;
initial categorical states are uniform as described in the paper; no accidental
channel padding is copied. This port is tested algebraically, but not claimed
bit-identical to JAX/Flax random initialization or the reference's PRNG streams.
See `METHODOLOGY_AUDIT.md` for the paper/code discrepancies.

## Verify synthetic learning before language training

Training examples are trajectory-aligned. The first complete grid is conditioning
context, excluded from the loss. Frame-marker prediction remains in the training
objective, matching the effective reference dataset behavior. Evaluation also
reports **patch-only** NLL and patch accuracy so marker prediction cannot satisfy
the quality check.

At step zero and every 100 updates, record synthetic train loss, held-out-rule
validation loss, patch-only NLL/accuracy and timing. Select the PPT checkpoint with
minimum held-out-rule patch NLL. Save it before transfer, and retain the final
PPT state too when it differs. The entire declared PPT training cost is charged,
even if the selected checkpoint is earlier.

Controls are fitted on synthetic training prefixes only:

- Smoothed patch unigram probabilities.
- A mixture of those probabilities and copying the patch at the same position
  in the previous grid. The mixture weight is selected on training data from a
  fixed 100-point grid, not on held-out rules.

For every grid-PPT model seed, the operational admission gate requires:

1. At least **5% relative reduction** in held-out-rule patch NLL from initialization.
2. Patch NLL at least **0.01 nats below both fitted controls**.

If any grid seed fails, the default workflow saves `ppt_gate.json`, selected
checkpoints, diagnostics and verification, then finishes with status
`stopped_at_synthetic_gate`. **No language models are trained.** This means the
configuration failed the predeclared prerequisite; it does not refute NCA
transfer in general. Do not silently relax the gate to obtain language results.
It is an engineering prerequisite added here, not a claim that the paper used
this exact admission rule. Passing it does not establish useful language transfer.

Only explicitly labeled execution-check fixtures can bypass the gate. They
cannot be used as scientific acceleration evidence.

## Fresh language data and reserved confirmation

The default language data are **WikiText-2 raw**, official train/validation
splits, from the [Salesforce dataset](https://huggingface.co/datasets/Salesforce/wikitext)
at revision `b08601e04326c79dfdd32d625aee71d232d685c3`. They do not reuse the
Shakespeare test set inspected in pilot 1. Parquet files are downloaded from
pinned URLs, joined with newline separators, encoded with GPT-2 BPE and hashed.
The tokenizer's actual merge table and special-token mapping are fingerprinted.
These downloads require Internet in the Kaggle notebook.

Language training draws random windows with replacement. It is an affordable
development corpus, not the paper's full OpenWebText corpus. Evaluation uses
32,768 fixed, non-overlapping target tokens from validation, with full context
blocks and no cross-split concatenation. Every arm sees identical sampled
training-window starts within a seed; hashes verify the match.

By default `evaluate_test=false`: the official confirmation test split is
**neither downloaded nor evaluated**. Do development decisions on validation
results. To schedule a confirmation evaluation, create a new predeclared config
with `evaluate_test=true` before preparation; then the final model of each arm is
evaluated once on its fixed test subset. Do not switch the flag after examining
test results or reuse an incompatible data directory.

WikiText content has its own CC-BY-SA/GFDL terms. Downloaded corpora and weights
are local run artifacts, excluded from publication inventories.

## Measure learning acceleration

Primary plots and metrics use **language-stage tokens/time starting at zero**.
For each seed, the predeclared reference target is the scratch arm's final
validation loss at N language updates. The method is fixed before running; the
numerical target is derived solely from that paired reference validation run.

For each synthetic arm, locate the first recorded evaluation reaching that
target, without interpolation. Report language-token and language-time speedups
as scratch cost divided by the synthetic-initialized arm's language cost.
Unreached targets remain censored (`null`). Never average only successful seeds:
an all-seed aggregate speedup is withheld if any paired seed misses the target.

Separately report total input tokens and measured cumulative cost, adding PPT
and relevant preparation to language training. A language-stage speedup can be
real while failing to repay PPT cost; report both findings explicitly.

Language seconds include training and fixed validation, starting after the
language model/head/optimizer are constructed. Cumulative seconds also charge
that setup, text preparation, relevant synthetic generation/cache writes,
shared manifest work and PPT stage costs. Synthetic preparation is conservatively
charged in full per arm/seed although the suite reuses its cache. They exclude
optional disposable calibration, package/runtime startup, cloud queueing and
work after the last validation record such as final checkpoint save, verification,
plots and ZIP creation. Neither metric is exact cloud-billing wall time.

The notebook calibrates actual update/evaluation timing on disposable models
first, then prints an estimate for the configured suite. That estimate excludes
several costs and is not a runtime guarantee. Calibration weights never transfer
into any research model.

## Running and completion

Import **`kaggle_nca_transfer_v2.ipynb`** into a private Kaggle notebook. Enable an
available GPU and Internet, then run all cells. It embeds the exact source and
configuration, executes validity checks and a tiny GPU execution fixture, prepares
real data, calibrates, runs PPT diagnostics and conditionally runs the matched
language benchmark. It uses one GPU.

The last cell exports **`nca_transfer_v2_results_<timestamp>.zip`** and explicitly
prints either `complete` or `stopped_at_synthetic_gate`. Both are completed
workflows; the second contains synthetic diagnostics rather than language
acceleration measurements. A Python exception is a failed run, not either status.
Save a Kaggle notebook version with outputs and download the ZIP.

CLI examples:

```bash
python -m transfer_v2.run all --config configs/transfer_v2.json --data data/transfer2 --out runs/transfer2 --device cuda
```

Separate `prepare`, `calibrate`, `ppt`, `language`, `report`, and `verify` commands
are available. The language command requires a passed PPT gate and unchanged
source/config/data identity. Never reuse existing run directories or silently
mix interrupted runs. Step-level resume is not implemented; retained PPT outputs
can feed a separate `language` command if that phase has not started.

CPU validity checks (not transfer measurements):

```bash
python scripts/run_logged.py --label v2-tests -- python -m unittest discover -s tests -p test_transfer_v2.py -v
python scripts/run_logged.py --label v2-offline -- python scripts/smoke_transfer_v2.py
```

The offline workflow checks both full completion and a deliberately failed
synthetic gate, reloading stage checkpoints and checking the language controls.
It uses generated byte fixtures, no WikiText download and no reserved test data.

New code is locally tested on CPU. GPU performance, gate outcomes and language
acceleration must be measured by the new Kaggle run; no positive result is baked
into the implementation.
