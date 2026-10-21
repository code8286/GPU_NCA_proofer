# Transfer v4 run 1: audit and research decision

Reviewed 19 October 2026. User-run Kaggle output:
`nca_transfer_v4_results_20261019_073853_931720.zip`.

**Conclusion:** this configuration does not demonstrate graph-specific learning
acceleration. Synthetic PPT again improves language learning relative to scratch,
but coherent graph trajectories do not outperform grid or the stronger matched
frequency/formatting control at the declared target. The extra PPT cost is not
recovered within the measured language horizon. A new post-hoc causal frequency
diagnostic also shows that the synthetic admission gate is too weak to establish
dynamics-specific predictive competence.

## Evidence and audit scope

The original ZIP and extracted evidence are preserved. The archive has 454
members, 3,159,151,102 compressed bytes and 3,432,166,529 expanded bytes. Imported
source files are treated as data; the audit executes trusted canonical code and
loads checkpoints with `weights_only=True`. It writes CPU results outside the
imported run and checks that the original GPU verification remains unchanged.

Independent CPU replay passed **all 78 selected development stage checkpoints
and six actual saved initializations**, with maximum metric discrepancy
**2.78275788e-07**. See [audit.json](audit.json),
[CPU verification](cpu_verification.json), [independent metric audit](metric_audit.json)
and [copy provenance](PROVENANCE.json). The 170 copied evidence files retain their
original bytes. The [frozen primary analysis](primary_analysis.json) is separate
from `treatment_comparisons.csv`, whose reference-arm endpoints are secondary.

Original archive SHA-256:
`d49a157b38c2e069b077d579c2f532b7b8d8914d190db8f284b98cd5e02d55c5`.
The imported GPU verification remains unchanged at
`4ce6eaadbc79232cb535ce47f13c6422c8495e34a87f23bd243657f7a59a64cc`.
These are new CPU consistency checks of imported GPU measurements, not new
training or a rerun of the GPU timing. All 78 local software regression tests
also passed; that test count is separate from the checkpoint count.

CPU replay covers the saved selected stage states. Intermediate language-curve
values are audited as recorded logs; model states for every intermediate
evaluation were not saved. The comparison targets are recomputed from those
complete raw logs, rather than reconstructed by retraining or interpolation.

The six declared cohort seeds are 59, 71, 83, 97, 109 and 127. All 36 synthetic
models passed their diagnostic gates and selected step 1000. Only the 12
structured models were admission requirements. Every one of the 42 language
models received the same 2000-update language budget, with 8.192M input tokens;
each synthetic model additionally received 1000 PPT updates and 4.096M input
tokens. Scratch received no extra text or synthetic stage. Actual saved initial
backbones/heads, selected-PPT transfer, fresh language optimizers, language batch
orders and schedules match within each cohort. No official test split was loaded.

The audit checks the exact configuration, 14 runner source files, archived
prospective plan/protocol, cache checksums, topology degree/slot treatment,
joint profile-filter eligibility, train/validation rule identities and weights,
regenerated old/new shuffles, initial conditioning and supervised-prefix counts,
fitted controls, checkpoint selection/gates, reconstructed batch orders,
evaluation cadence and raw-curve comparisons. Whole binary schedule hashes need
not regenerate across Windows/Linux math libraries: sampled learning-rate values
are checked numerically, while within-run schedule identity is checked exactly.

The dataset, topology and initialization seed change together; these variance
sources are not separately estimated. These are fresh synthetic cohorts on an
already-inspected small language task, not untouched-domain confirmation.

The transferable backbone has 2,656,704 parameters; the GPT-2 vocabulary makes
the language model 12,306,048 parameters in total. The reinitialized language
embedding/head is shared within each cohort. This small-model result does not
establish behavior for larger backbones, other corpora, graph families or
serialization schemes.

## Frozen primary result

Both comparisons use the **same per-cohort scratch final NLL at 2000 updates**.
The first recorded crossing is used, without interpolation. Lower update counts
favor an initialization. Scratch first reaches its endpoint at update 2000.

| Cohort | Grid | Graph | Prefix-matched graph shuffle |
| --- | ---: | ---: | ---: |
| 59 | 1700 | 1700 | 1700 |
| 71 | 1700 | 1700 | 1700 |
| 83 | 1700 | 1700 | 1700 |
| 97 | 1800 | 1800 | 1800 |
| 109 | 1700 | 1700 | 1800 |
| 127 | 1700 | 1800 | 1700 |

| Graph comparison | Wins / ties / losses | Geometric token-speed ratio | Frozen one-sided sign p | Criterion met |
| --- | --- | ---: | ---: | --- |
| Versus grid | 0 / 5 / 1 | 0.990519x | 1.000000 | No |
| Versus prefix-graph shuffle | 1 / 4 / 1 | 1.000000x | 0.984375 | No |

Ratios are reference tokens divided by graph tokens; values above one favor
graph. There are no censored crossings in these comparisons. The declared
conservative sign test counts ties as non-wins and uses alpha 0.025 per test.
Both tests had to pass. Neither passes; no alternative target, seed exclusion
or changed significance rule is substituted after seeing the outcome.

This is a failure to demonstrate the declared advantage, not a proof that all
graph PPT mechanisms are equivalent or impossible. Evaluation every 100 updates
cannot resolve a crossing difference smaller than that grid. That limitation
does not justify interpolating new primary wins from the observed curves.

## Final language quality and cost

| Initialization | Mean final validation NLL | Token speedup versus scratch | Cumulative-time speedup versus scratch |
| --- | ---: | ---: | ---: |
| Scratch | 5.273956 | 1.000000x | 1.000000x |
| Grid | 5.230608 | 1.165316x | 0.864328x |
| Graph | 5.234337 | 1.154268x | 0.857989x |
| Original shuffled grid | 5.236298 | 1.143913x | 0.851626x |
| Original shuffled graph | 5.234172 | 1.165990x | 0.864376x |
| Prefix-matched shuffled grid | 5.235063 | 1.166591x | 0.864891x |
| Prefix-matched shuffled graph | 5.235680 | 1.154268x | 0.858035x |

Graph's mean NLL is 0.003729 nats/token worse than grid (about 0.37% higher
perplexity), with graph worse in five of six final-loss pairs. Graph is
0.001344 nats better than prefix-graph on average (about 0.13% lower perplexity),
with mixed paired signs. This small final-quality difference does not establish
the predeclared token-acceleration result.

The [paired effect figure](paired_graph_transfer.png) shows NLL differences
from update 500 onward for legibility. The archived [full learning curves](gpu/learning_curves.png)
include stage zero; all target crossings use the complete recorded curves.

Graph reaches the scratch target using 10-15% fewer language updates per cohort;
the geometric token ratio corresponds to about 13.4% fewer language tokens.
The recorded language-stage time ratio is 1.154195x, while cumulative time is
0.857989x: approximately **16.6% more cumulative time** at that loss. These are
different outcomes and must remain separate. Shuffled controls also transfer;
the downstream improvement is not isolated to coherent NCA dynamics.

Cumulative timing conservatively charges the whole paired generation/cache
preparation to each synthetic arm, as declared before the run. It excludes
initialization snapshots, disposable calibration, queueing, final verification,
plotting and export. It is neither an isolated graph-generation benchmark nor
the cloud bill. See `timing_summary.json` for the sum of recorded stage timers.

## New post-hoc diagnostic: causal adaptation to frequencies

The declared gate compared PPT prediction to a global training-fitted unigram
and a previous-frame-copy mixture. It did not include a predictor that infers
rule/trajectory-specific frequencies from the observed prefix.

The new diagnostic uses a Dirichlet-smoothed online unigram predictor. Its prior
probabilities are fitted on all training prefixes; concentration is chosen from
six declared candidates using only the first 256 training rules. It begins with
the true first-frame counts, then updates counts after each observed patch.
Markers are excluded, matching patch-only evaluation. Validation uses the same
32 held-out rules and 29,952 patch targets as the synthetic model diagnostic.

The calculation uses the closed-form joint probability of the sequential
predictor, not a multinomial count-mass including a combinatorial coefficient.
Although that identity is evaluated from final counts, individual predictions
depend only on earlier observed patches. A regression checks agreement with
explicit causal sequential prediction. Because the parent and prefix control
share initial and target counts, their joint baseline loss is identical.

| Parent task | Mean PPT held-out patch NLL | Causal frequency baseline NLL | PPT minus baseline |
| --- | ---: | ---: | ---: |
| Grid | 8.160504 | 7.572958 | +0.587546 |
| Graph | 8.101986 | 7.804035 | +0.297951 |

The simple predictor beats every structured PPT model, all 12 pairs. This is a
post-hoc diagnostic, not a revised v4 gate or a confirmatory statistical test.
It shows that beating the original controls was insufficient to establish
dynamics-specific readiness for transfer. It does **not** prove that the models
learned no ordered features, that NCA lacks predictable dynamics, or that
frequency learning uniquely caused the downstream benefit. Compare each model
to its own task's baseline; native grid/graph synthetic losses have different
task distributions and should not be ranked as if they were one dataset.

## Recommended next steps

1. **Do not repeat this graph configuration with more seeds or loosen its gate.**
   Preserve the negative primary result. Keep grid and both matched controls;
   v2/v3 remain separate evidence and are not pooled into this study.
2. **Use existing checkpoints before buying another language-training run.**
   Cross-evaluate grid/graph PPT checkpoints on their original held-out trajectories
   and their matched prefix shuffles; also cross-evaluate prefix-trained models.
   This tests an order-sensitivity difference while fixing conditioning and
   target counts. It needs no new model training. Freeze that diagnostic before
   inspecting its outcome, and label it development work on inspected rules.
3. **Strengthen readiness for the next declared study.** Require improvement over
   the causal trajectory-frequency predictor and a coherent-versus-shuffled
   predictive advantage before spending on language transfer. First validate
   the measurement on a simple known transition-rule positive control. Audit
   representation/capacity with one change at a time; current evidence does not
   identify serialization or rewiring as the cause of failure.
4. **Then choose the learning-speed objective explicitly.** For economical PPT,
   test shorter synthetic stages in a new development protocol, retaining the
   same full language stage and matched shuffled controls. Current synthetic
   warmup helps language exposure but is too costly at K=1000. For a GNN-specific
   claim, any graph-local serializer/topology change must be a separate matched
   ablation that first clears the stronger synthetic check. Do not change both
   together and attribute the outcome to graph structure.
5. **Confirm only after that development choice is frozen.** Use fresh paired
   cohorts and untouched confirmation data, preserve all controls and costs,
   and predeclare targets/censoring. The reserved official test remains untouched
   by this audit. No new GPU run or paid job has been launched.

The immediate next objective is checkpoint-level evidence for order-dependent
prediction beyond causal frequency adaptation, rather than another full GPU
language suite with the same admitted models.
