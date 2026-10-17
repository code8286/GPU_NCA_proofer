# Version 3: graph-NCA pre-pre-training transfer

The primary question is whether **graph-NCA pre-pre-training yields an
initialization that learns language faster than grid-NCA pre-pre-training**.
Both treatments enter an identical language stage. V2 showed similar gains
from grid and shuffled-grid PPT, so a graph arm needs its own shuffled control.
This is an exploratory topology treatment, not a claim of faster GPU kernels.

Use [the new Kaggle notebook](../kaggle_nca_transfer_v3.ipynb).
Configuration: `configs/transfer_v3.json`. V1, v2 and their returned evidence
retain their original identities. The versioned `transfer_v3/` package preserves
the v2 transformer implementation and language optimization settings.

## Five initializations, one language procedure

| Arm | Pre-pre-training | Subsequent language stage |
| --- | --- | --- |
| `scratch` | None | 2000 identical updates |
| `grid_nca` | 1000 grid updates | 2000 identical updates |
| `graph_nca` | 1000 graph updates | 2000 identical updates |
| `shuffled_nca` | 1000 shuffled-grid updates | 2000 identical updates |
| `shuffled_graph` | 1000 shuffled-graph updates | 2000 identical updates |

Effective batch/context is unchanged: 2 sequences x 2 accumulation x 1024
tokens = 4096 input tokens/update. Each PPT stage consumes 4.096M input tokens;
every language stage consumes 8.192M. Full completion runs 12 PPT and 15 language
stages, totaling 172.032M model-input tokens. Conditioning tokens count as
exposure; supervised synthetic-target counts are logged separately.

The model remains six layers, width 192, four heads, RMSNorm/RoPE/SwiGLU, no
dropout. PPT vocabulary 10,002 gives 4,577,088 parameters; language GPT-2
vocabulary 50,257 gives 12,306,048. PPT uses Adam at 1e-4 without weight decay or
clipping; language uses fresh AdamW at 5e-4, weight decay 1e-4 and clip 1. Both
stages have their own 10% warmup/cosine schedule. FP32, deterministic algorithms
and TF32 disabled match v2. Synthetic embeddings and optimizer moments never
transfer; the full selected backbone does.

Save the actual common initial backbone and language embedding/head for each
seed in `initialization_<seed>.pt`. Every PPT starts from that saved backbone;
all five language arms load the saved common head. Scratch loads the untrained
saved backbone. Other arms load their selected PPT core. Checksums verify these
relationships using stored tensors, without recreating initialization from a
seed under a different runtime.

## Graph construction and paired data

The categorical NCA family matches the v2 reference port: 12x12 nodes, ten
one-hot states, nine ordered perception slots -> four perception channels ->
16 ReLU hidden units -> ten next-state logits. Frozen random rule weights,
temperature 1e-4, spatial-only periodic padding for grid, ten burn-in updates,
2x2 base-ten patch tokens and frame markers are unchanged. Refer to
[v2's protocol](TRANSFER_V2_PROTOCOL.md) and the pinned author reference for
the original generator assumptions and documented port caveats.

Grid and graph both call the **same slot-labeled message function**. Grid donors
are the periodic 3x3 neighborhood. Graph construction swaps two donor indices
within the same non-self slot, rejecting self/duplicate neighbors. Stop once
at least 25% of the eight non-self donor entries differ from the grid. Retain
the actual changed fraction, graph array, topology seed and hash. Every node
keeps its fixed self slot and eight distinct other donors; each slot is a
permutation, preserving degree on both sides. This is a directed relational
message-passing GNN, not an isotropic adjacency-averaging GCN or a trained
generator. The graph is fixed during its replicate's train and validation.

Node labels and 2x2 serialization remain fixed. A graph patch need not correspond
to a local graph neighborhood; tokenization is not graph-isomorphism invariant.
Any result concerns this declared topology/serialization combination, rather
than a general property of all graph NCAs.

For each model seed 17, 29, 43, generate a separate paired rule cache and graph
topology. These are three independent **model/data/topology replicates**, each
paired across treatments; they do not separately estimate those three variance
components. Each replicate has 4096 retained train rules and 128 validation
rules, with the first 32 validation rules fixed for model evaluation.

Sample one candidate rule bank. Roll out both topologies with the same initial
categorical states and sampling draws. Retain a rule only when **both** ten-frame
profiles pass **0.5 < marker-free gzip ratio < 1.0**. Do not relax thresholds.
Generate fresh paired training/validation trajectories afterward, again sharing
initial states/draws within the pair. Screening and actual rollout initial
conditions differ. Fresh trajectories can fall outside the screening band.
Save one bank of frozen rule weights shared by the two topology treatments.
This conditions the rule population on joint eligibility; it is a different
grid dataset from v2, which is why all five controls are rerun within v3.

Shuffled controls preserve their parent's full trajectory histogram and exact
marker positions, destroying patch order. Prefix histograms need not match
because contexts use a prefix. Retained per-trajectory frequency structure and
formatting remain potential causes of transfer. A gate pass against global
frequency/copy baselines alone does not isolate local dynamics.

## Gate, verification and data policy

Before any language training, **every grid AND graph seed** must show at least
5% relative held-out patch-NLL improvement from initialization and at least
0.01 nats lower patch NLL than both train-fitted unigram and copy-mixture
controls. Shuffled diagnostics are retained but do not determine admission.
One structured seed failing stops the whole suite at `stopped_at_synthetic_gate`.
Only labeled execution fixtures may bypass this requirement.

PPT uses trajectory-aligned sequences, masking the first grid as conditioning.
Markers remain supervised; patch-only validation prevents markers alone passing
the gate. Evaluate every 100 updates and choose minimum validation patch NLL,
including step zero. Save selected and, when different, final PPT states. Charge
the entire declared K budget regardless of selection.

WikiText-2 raw/GPT-2 BPE, pinned dataset revision, sampling windows, fresh language
optimizer, schedule and fixed 32,768-token validation prefix match v2. Default
`evaluate_test=false`: the official test is neither downloaded nor evaluated.
V2 development validation has already been examined. No confirmation result is
claimed; freeze any future confirmation protocol before inspecting its data.

Verification reloads all **27 selected stage checkpoints plus three saved
initializations** after full completion, or 12 PPT checkpoints plus initializations
after a default gate stop. Extra final PPT checkpoints are also checked. It
checks source/data/model identity, actual rule-weight split disjointness, graph
invariants, shuffled histograms/markers, replayed losses, equal language budgets,
fresh optimizer records and common head/core/data-order/schedule hashes.

## Primary result and cost accounting

`report.json` contains `treatment_comparisons`. Its first comparison is
**graph vs grid**, targeting each seed's grid endpoint after N language updates.
Also report graph vs shuffled graph and grid vs shuffled grid. For every pair,
report final NLL difference, first recorded reference crossing and language-token/
time speedup. A treatment that misses its reference remains censored; do not
average successful seeds alone. Evaluations are discrete, without interpolation.

Keep the v2-style comparisons to the paired scratch endpoint as additional
context in `comparisons`/`aggregate`. Improvement over scratch alone is
insufficient to identify a graph-topology or dynamics advantage. Plot language
exposure and time from stage zero; report cumulative cost separately.

Cumulative time charges text preparation, language setup, the full PPT stage,
language training/validation and conservatively the **entire paired screening/
cache preparation for that replicate** to each synthetic treatment. Generation
is joint, so this cannot measure a standalone graph-vs-grid generation speedup.
It excludes common initialization snapshot creation, package startup, queueing,
disposable calibration and final language
checkpoint/verification/plot/archive work. Record generation diagnostics without
calling them optimized-kernel benchmarks.

## Run and completion

Upload `kaggle_nca_transfer_v3.ipynb`, enable one available GPU and Internet, then
Run All. The notebook embeds source/tests/config, runs a tiny GPU execution
fixture, prepares real data, calibrates, trains, verifies and exports
`nca_transfer_v3_results_<timestamp>.zip`.

Based on the returned v2 T4 timings, budget roughly **2.5–3 hours** for the default
suite, with preparation/verification/archive work additional and variable.
The notebook prints an estimate measured on your actual GPU before PPT. This
estimate is not a runtime guarantee. One GPU is used.

The final cell prints `WORKFLOW FINISHED: complete` or
`WORKFLOW FINISHED: stopped_at_synthetic_gate` and a ZIP download link. A gate
stop is useful diagnostic output, without language results. An exception is a
failed run. Save a Kaggle version with outputs and download the ZIP. No step-level
resume is implemented; do not rerun into existing output directories.

CLI: `python -m transfer_v3.run all --config configs/transfer_v3.json --data data/transfer3 --out runs/transfer3 --device cuda`.
Separate `prepare`, `calibrate`, `ppt`, `language`, `report`, `verify` are available.
Run local checks through `scripts/run_logged.py`: unit discovery and
`scripts/smoke_transfer_v3.py`. CPU fixtures demonstrate software execution only.
