# Stage 6: controlled transformer trainability

Declared before model fitting or assessment. This is a CPU development diagnostic
following stage 5's failed random-rule readiness. Preserve all v1-v5 sources,
plans, checkpoints and results. This stage never authorizes language training.

## Question and scope

Can the existing transformer implementation learn a simple known transition rule
well enough to pass BOTH unchanged readiness advantages? This follows the stage 5
transition-table measurement control with an actual learned transformer.

Use the unchanged v4 LM, tied embeddings, six layers, width 192, four attention
heads, RoPE, RMSNorm and SwiGLU. Preserve the 2x2 patch encoder, first-frame target
mask, marker handling, synthetic minibatch/update function, Adam, warmup/cosine
schedule and full 1000-update budget. Training predicts patches AND subsequent
markers; readiness scores patches only, exactly as in v4.

The task is intentionally easier: a binary 4x4 lattice, eight frames, 18-token
vocabulary and 47-token context. The known cell rule flips every binary cell each
frame; each patch token becomes its bitwise complement. It has no spatial
interaction, graph rewiring, random-rule filtering or burn-in. Alphabet, lattice
and context are simplified together to establish a basic capability check;
success cannot isolate which difference matters for v4. It does not establish
held-out-rule generalization, graph learning, language acceleration or cost recovery.

## Fixed data and training

Config: `configs/trainability_v6.json`. Generate 512 training, 64 development and
128 assessment initial conditions using seed 20261022. Canonical complement-pair
classes are sampled without replacement; no phase-equivalent trajectory crosses
splits. Save all array/class hashes before training. Each split has coherent and
matched prefix-shuffled versions, preserving first conditioning frame, marker
positions and supervised patch histogram. There are 28 scored patches/trajectory.

Three paired model seeds (131, 149, 167) share this one task/data split. For each,
train coherent and prefix-shuffled arms from the exact same saved initial full
state. Use common minibatch index order, batch 2, accumulation 2, learning rate
1e-4, 10% warmup, cosine decay to 10%, zero weight decay and no gradient clipping,
while checking finite gradients. Each arm receives exactly 1000 updates and
188,000 input tokens. All six arms total 6000 updates / 1,128,000 input tokens.
Record actual prediction-target counts separately. Compute on CPU with two
PyTorch threads, FP32 and deterministic algorithms. Timings are implementation
measurements, not GPU benchmarks or cloud-billing estimates.

Log development loss every 100 steps. Do not select a checkpoint, early stop,
retune settings or extend the budget after viewing progress. The primary endpoint
is the final declared update. No assessment scores are computed until every arm
has completed its full budget. Assessment arrays may be generated/hashed during
freeze; target likelihoods must remain unread until that final assessment.

## Fixed scoring and decision

Fit the stage 5 causal frequency prior/concentration only on training data, using
the same concentration grid and first 256 training initial conditions. On all
128 held-out assessment initial conditions, evaluate each seed's initial,
coherent-trained and shuffle-trained checkpoints on BOTH data versions: 18 cells,
each with 3584 patch targets. Retain per-initial-condition NLL, accuracy and counts.
No calibration/assessment checkpoint selection is permitted.

Each of the three coherent-trained final checkpoints must satisfy BOTH:

1. Frequency NLL minus coherent NLL >= **0.01 nats/patch**.
2. Same checkpoint's shuffled NLL minus coherent NLL >= **0.01 nats/patch**.

On the same training/assessment task, a train-only smoothed previous-frame
transition table must recover all 16 complement transitions and pass both margins.
The causal-frequency predictor must fail (zero frequency/self and ordering
advantages). Table failure invalidates the measurement. Report initialization
improvement, shuffle-trained model gaps and difference in ordering gaps as
descriptive diagnostics; they are not extra admission criteria. Do not treat the
shared-data model seeds or individual initial conditions as independent NCA
cohorts, and do not manufacture a language-transfer p-value.

Freeze protocol/config/source hashes, arrays and actual starting checkpoints
before training. Reject altered plans, source, data, summaries and final weights.
Preserve optimizer states and actual checkpoint hashes. Independent verification
must reload every checkpoint and reproduce all 18 assessment cells and decisions.
Use fresh output folders; never overwrite a frozen or failed run.

## Decision after this diagnostic

If it fails, retain the negative result and inspect capacity/pipeline with a new
declared one-factor development change. Do not retune this run. If it passes,
the next task is a separately declared bridge to an interacting categorical rule,
changing one task/representation factor at a time. This simple self-local task
does not license another language suite on the unchanged random-rule models.
Any future transfer suite still requires its own synthetic readiness, fresh
paired cohorts and an identical language stage; the official language test stays
untouched. Serializer/topology and shorter-PPT cost studies remain separate.
