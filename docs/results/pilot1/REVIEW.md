# Review of Kaggle output 1

**Protocol qualification, 17 October 2026:** these numbers audit the original
fixed-total-budget pilot. That pilot gives scratch 1000 text updates and the NCA
arms 100 synthetic + 900 text updates, with different scheduler positions on the
language-exposure axis. It does not cleanly isolate pre-pre-training transfer
into an identical language stage. See [the methodology audit](../../../METHODOLOGY_AUDIT.md)
for the paper/code comparison and retrospective language-token alignment. The
original numerical checks and negative findings for this budget remain valid.

## Finding

The run completed successfully, but this pilot does **not** show that graph-NCA pre-training accelerates language-model training.
Graph NCA took **1.432x as long** to reach the declared 2.5 validation-nats/byte target as target-text-only training (paired geometric ratio).
Its final held-out NLL was higher by **0.053230 nats/byte**, corresponding to **5.47% higher geometric-mean byte perplexity**.
All three seeds favored text-only over graph NCA on both time to target and final held-out loss.

## Checks completed

- All 15 pilot runs reached step 1000: five arms and three paired seeds.
- Every run used 1,857,024 parameters and 4,096,000 total training tokens; total pilot exposure was 61,440,000 tokens.
- Returned experiment source exactly matches the trusted local source and its recorded SHA-256.
- ZIP CRCs, all dataset checksums, source text, split ranges, and checkpoint SHA-256 hashes passed.
- Initialization hashes and initial validation loss match within each paired seed.
- Rewired graph preserves incoming/outgoing degree and has no self/duplicate edges; shuffled data preserve per-row frequencies and marker positions.
- All 12 reported pairwise comparisons were independently recomputed from the raw logs.
- Kaggle's saved GPU verification reports zero error for 15 reloaded checkpoints.
- All 15 checkpoints were independently re-evaluated locally on CPU; maximum CPU/GPU loss difference was **8.2e-08**, below the declared 0.0001 tolerance.

## Results

Means across three seeds. Lower loss, perplexity, time and tokens are better.

| Arm | Test NLL, nats/byte | Byte perplexity, geometric mean | Pipeline seconds to target, arithmetic mean | Total tokens to target, arithmetic mean |
| --- | ---: | ---: | ---: | ---: |
| scratch | 1.8828 | 6.572 | 6.117 | 819,200 |
| reserve_text | 1.8790 | 6.547 | 6.117 | 819,200 |
| grid_nca | 1.9289 | 6.882 | 8.818 | 1,160,533 |
| graph_nca | 1.9361 | 6.931 | 8.775 | 1,160,533 |
| shuffled_graph | 1.9363 | 6.933 | 9.276 | 1,228,800 |

The target is validation NLL <= 2.5; time/token estimates are the first recorded evaluation reaching it, without interpolation.
Evaluations occur every 50 updates (204,800 tokens), so small differences in threshold crossing are coarse. Loss is evaluated on 65,536 fixed held-out bytes for validation and likewise for test.
The text-only baseline trains on target text in both stages; the separate-text baseline uses a disjoint same-domain reserve before target training.

## Interpretation

1. **No practical speedup against text-only.** The graph pipeline speedup is 0.698x (baseline time / graph time). Equivalently graph takes about 43% longer at this target. Its final test loss is worse in every seed.
2. **No demonstrated graph-topology benefit over grid NCA.** Both cross the target at the same recorded update in every paired seed. The 1.005x aggregate timing ratio is about 0.5%, with mixed signs and an exploratory interval spanning one. Graph's mean test NLL is slightly worse.
3. **No convincing evidence that temporal trajectory structure helps.** Graph vs shuffled graph has virtually identical mean test loss (delta -0.000215 nats/byte). A roughly 6% aggregate threshold timing advantage is driven mainly by one seed crossing one evaluation earlier. This is not evidence of a general structural transfer gain.
4. **Some token savings disappear when all training is counted.** Graph's target-text token speedups against scratch are 1.00x, 1.00x and 1.33x, but total-token speedups are only 0.667x, 0.667x and 0.800x. Synthetic exposure is extra work and must remain in the accounting.
5. **These results apply to this small pilot.** They do not disprove NCA transfer at other scales or with other rules. Only one synthetic dataset/topology is shared across the three model seeds. Three seeds and seed-bootstrap intervals remain exploratory.

## Runtime and generator diagnostics

Hardware recorded: **Tesla T4**, PyTorch 2.11.0+cu128, fp32. Summed training time was **6.25 minutes**; summed last-logged per-arm pipeline costs were **7.08 minutes**. These are not the complete notebook wall time.
The earlier 30–90 minute planning estimate was overly conservative for this actual GPU run.
Graph synthetic preparation was recorded as 0.151 seconds, versus 0.327 seconds for grid, but each generator was timed once in a fixed order. Kernel warmup and order can explain this difference; it does not establish a GNN generation-throughput advantage.
About 36.3% of graph trajectories were at a fixed point on their final transition (grid: 35.7%). This is a useful warning about trivial dynamics, not a proven explanation of transfer performance.

## Next experiment

Preserve this run as a negative pilot. Before another full suite, inspect trajectories and pre-register a generator study with nontrivial dynamics and matched complexity bands. Treat embedding reset/transfer policy as a separate ablation: synthetic arms restore byte embeddings while text arms preserve them. Any tuned follow-up needs untouched confirmation data; do not use this test set to select a winning configuration.
The current results compare graph topology in a frozen random-rule NCA data generator. They do not compare optimized GPU graph kernels against convolution kernels or put a GNN inside the language model.

## Files

- `audit.json`: independent audit and CPU verification measurements.
- `comparison.csv`: aggregate metrics recalculated from the raw logs.
- Original Kaggle metrics, checkpoints and GPU verification remain in the research workspace; compact metrics and logs are included here.
- Original ZIP preserved unchanged locally; SHA-256 is recorded in `audit.json`.
