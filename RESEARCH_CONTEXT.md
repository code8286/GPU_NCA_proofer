# Research context: GNNs in NCA pre-pre-training

Updated: 17 October 2026.

The research question is whether **graph-NCA data improves the transformer
initialization acquired during pre-pre-training**, before ordinary language
pre-training. Grid NCA and graph NCA are synthetic sources for that first stage.
Neither is a replacement for the subsequent language-model training stage.

Primary references to retain:

- Paper, version 1: https://arxiv.org/html/2603.10055v1
- Author repository: https://github.com/danihyunlee/nca-pre-pretraining/tree/main
- Pinned source revision inspected: `bdd1c71e04eb2e5a2c91481faf278e95846b4431`
  (12 March 2026).
- Local source snapshot and checksums:
  `sources/nca-pre-pretraining-reference/reference.json`, with files beneath
  `sources/nca-pre-pretraining-reference/bdd1c71e04eb2e5a2c91481faf278e95846b4431/`.

The correct primary comparison is:

| Initialization entering language pre-training | Synthetic pre-pre-training | Language pre-training |
| --- | --- | --- |
| Random / no pre-pre-training control | None | Same N text updates |
| Grid-NCA initialized transformer | K grid-NCA updates | Same N text updates |
| Graph-NCA initialized transformer | K graph-NCA updates | Same N text updates |
| Shuffled-NCA initialized transformer | K shuffled-data updates | Same N text updates |

Use identical language data order, optimizer initialization, learning-rate
schedule, batch/context, evaluation and stage length. Compare graph vs grid as
pre-pre-training treatments. A text-only model remains a valid **no
pre-pre-training control**, as in the paper; it must not get an unacknowledged
head start on the common language-stage clock.

Report transfer efficiency on language-stage tokens/time first. Separately
report cumulative synthetic + language cost and generation cost. The paper also
reports a combined token-cost metric in Appendix D; these questions must remain
distinct rather than removing cost accounting altogether.

Pilot 1 is preserved locally in `Kaggle Outputs/gnn_nca_results_1.zip`; compact
public evidence is in `docs/results/pilot1/`. All 15 runs and
checkpoints passed independent verification. Its design was a **matched total
budget replacement pilot**: scratch received 1000 text updates; synthetic arms
received 100 synthetic + 900 text. Its negative result is valid for that design
but is not a clean replication of the paper's transfer experiment. Its scratch
scheduler also restarted after 100 text updates, unlike the synthetic arms'
language-stage schedule.

See `METHODOLOGY_AUDIT.md` for the specific mismatches. Existing aligned logs show
no sustained graph benefit, so correcting the framing must not be described as
discovering a positive result. At 900 target-text updates, mean validation NLLs
are 1.7542 (scratch), 1.8002 (grid), and 1.8170 (graph), with the scratch schedule
still confounded. No matched-text final scratch test checkpoint was saved.

Do not overwrite original run files or silently relabel the old experiment.
Establish useful held-out-rule NCA learning, reproduce grid-NCA transfer at an
affordable scale, then introduce the graph modification. Avoid interpreting a
one-time generation timing difference as an optimized graph-kernel speedup.

## Implemented next run

The independent-stage implementation is now `transfer_v2/`, configured by
`configs/transfer_v2.json`, with notebook `kaggle_nca_transfer_v2.ipynb`.
See `docs/TRANSFER_V2_PROTOCOL.md`. It restores categorical NCA/rule filtering/
burn-in and uses a small Llama-style model, independent 1000-update PPT and
2000-update language budgets, grid/shuffle/no-PPT controls, stage-boundary
checkpoints and a held-out-rule prediction gate. Language heads, data order,
optimizers and schedules match across arms. The language confirmation split is
reserved by default. The returned v2 GPU development run is now measured; see
[its results review](docs/results/transfer_v2_run1/REVIEW.md). All three grid seeds
passed the synthetic gate and reached their paired scratch endpoint after
1600–1800 language updates versus 2000, with a geometric language-token speedup
of 1.178x. Shuffled NCA achieved the same aggregate token speedup. Mean final
validation NLLs were 5.266091 (scratch), 5.233321 (grid), and 5.231528 (shuffled).
This supports a small synthetic-initialization transfer effect in this setup;
it does not isolate local dynamics or establish a graph benefit. The recorded
cumulative cost of grid NCA was about 13% higher at the reference loss.
The official test remains unused. Graph NCA has not been run under v2.

## Next stage: graph transfer v3

`transfer_v3/`, `configs/transfer_v3.json` and
`kaggle_nca_transfer_v3.ipynb` implement the next experiment; see
[the v3 protocol](docs/TRANSFER_V3_PROTOCOL.md). It compares no-PPT, grid, graph,
shuffled grid and shuffled graph before the same 2000-update language stage.
All synthetic arms receive 1000 PPT updates. Each paired seed gets an independent
rule cache/topology. Grid and graph share frozen rule weights and paired random
draws; selection requires both profiles to pass the same complexity band.
The graph changes donor connectivity while preserving degrees/ordered slots.

The primary comparison is graph-PPT versus grid-PPT; graph versus its shuffled
control is needed to assess ordered-trajectory structure. Comparisons to no-PPT
and cumulative cost remain additional results. Actual initial core/head weights
are saved, so cross-platform checks can verify those tensors directly. Both grid
and graph must pass the synthetic gate before any language run. This is new
software with CPU checks; no v3 GPU acceleration result exists yet. Preserve
the measured v2 result separately and leave official-test confirmation unused.
