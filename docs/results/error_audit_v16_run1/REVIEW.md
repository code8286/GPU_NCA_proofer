# Stage 16: cache-versus-held-out error audit

Completed local CPU retained-output analysis, 31 October 2026. No new fitting
or model inference. All five arms and both fixed update-600 variants are retained.

**The small-cache fit does not generalize, and formatting alone cannot account
for its held-out failure.** Valid held-out vectors still have low semantic node
accuracy. Gold-prefix teacher forcing also leaves digit prediction weak. Source
and one-hop errors show that failure is not confined to longer paths. These
observations do not identify a unique causal mechanism or establish graph
equivalence, learning acceleration or cost recovery.

## Frozen scope and provenance

Plan `279bd3b7ea385de061d9ec8c02c5a87b329c8c2d4b1bdd15a9a1c225aa357702` and the
[frozen protocol](FROZEN_PROTOCOL.md) precede this new analysis. All 52 stage 15
evidence copies, its manifest and inherited scientific sources are hash-bound.
The SAME selected cache16 and full ID-development128 examples are compared
between small16 and the IMPORTED historical512 reference. No checkpoint or
example is selected by its new error profile. These are inspected development
data from one historical cohort, with two correlated tasks per graph.

Implementation disclosure: the inherited deterministic generator also recreates
assessment/OOD rows in memory before only ID rows are selected. The protocol's
blanket "No OOD data" wording is too broad; see the preserved-source
[regeneration note](../../ERROR_AUDIT_V16_REGENERATION_NOTE.md). No assessment/OOD
prediction, scoring, model input or selection occurs; no new forward pass runs.

See [full results](results.json), [100 group/cell rows](cells.csv) and
[50 paired comparison rows](paired.csv). Each arm/variant/probe includes pooled
and all four task/size cells; pooled summaries below do not replace them.

## Cache fitting versus held-out semantic and token errors

Cache entries are small16 / historical512 on the same 16 examples:

| Arm | Exact answers | Teacher-forced digit accuracy | Generated node-micro accuracy |
| --- | --- | --- | --- |
| scratch | 16/16 / 0/16 | 100.00% / 43.06% | 100.00% / 31.94% |
| grid_nca | 16/16 / 0/16 | 100.00% / 39.58% | 100.00% / 36.11% |
| graph_nca | 16/16 / 0/16 | 100.00% / 38.19% | 100.00% / 36.11% |
| prefix_grid | 16/16 / 0/16 | 100.00% / 37.50% | 100.00% / 37.50% |
| prefix_graph | 16/16 / 0/16 | 100.00% / 36.81% | 100.00% / 29.17% |

Every cache node, source and observed hop bin is correct in small16; digit and
other teacher token accuracy are both 100%. Every held-out exact score remains
0/128 for BOTH variants. Held-out details:

| Arm | Variant | Valid vectors | Node-micro, all outputs | Node-micro, valid vectors only (correct/nodes) | Teacher digit | Teacher other |
| --- | --- | --- | --- | --- | --- | --- |
| scratch | small16 | 85/128 | 12.50% | 18.56% (144/776) | 20.31% | 93.61% |
| scratch | reference512 | 127/128 | 21.44% | 21.63% (247/1142) | 27.60% | 99.93% |
| grid_nca | small16 | 68/128 | 9.98% | 18.08% (115/636) | 20.14% | 94.74% |
| grid_nca | reference512 | 128/128 | 18.92% | 18.92% (218/1152) | 24.22% | 100.00% |
| graph_nca | small16 | 66/128 | 8.77% | 16.83% (101/600) | 19.18% | 91.83% |
| graph_nca | reference512 | 128/128 | 21.09% | 21.09% (243/1152) | 27.95% | 99.93% |
| prefix_grid | small16 | 80/128 | 12.41% | 19.81% (143/722) | 20.57% | 93.04% |
| prefix_grid | reference512 | 126/128 | 19.62% | 19.89% (226/1136) | 25.35% | 99.93% |
| prefix_graph | small16 | 69/128 | 9.55% | 17.35% (110/634) | 19.36% | 95.17% |
| prefix_graph | reference512 | 126/128 | 21.18% | 21.52% (244/1134) | 27.26% | 99.93% |

Invalid vectors score zero at every node in unconditional rates. Conditional
rates describe different model-selected valid subsets, so their differences do
not isolate a formatting effect. Teacher scores are TOKEN-MICRO aggregates of
imported counters, supplying gold earlier answer tokens; they are not free-
generation node accuracies. Digit-bearing GPT-2 tokens can include punctuation.
Stage 15's reported token-cell means and node-example means use different
denominators; both those original metrics and this audit's explicit micro/macro
counters are preserved. No neural logit or teacher forward pass is rerun here.

## Source and hop errors

Unconditional correct nodes / all nodes in each role, including invalid outputs
as wrong. Hop bins use UNWEIGHTED paths even for weighted Dijkstra answers:

| Arm | Variant | Source | One hop | Multi hop | Unreachable |
| --- | --- | --- | --- | --- | --- |
| scratch | small16 | 10/128 | 41/388 | 93/636 | 0/0 |
| scratch | reference512 | 14/128 | 86/388 | 147/636 | 0/0 |
| grid_nca | small16 | 5/128 | 44/388 | 66/636 | 0/0 |
| grid_nca | reference512 | 12/128 | 85/388 | 121/636 | 0/0 |
| graph_nca | small16 | 8/128 | 33/388 | 60/636 | 0/0 |
| graph_nca | reference512 | 17/128 | 74/388 | 152/636 | 0/0 |
| prefix_grid | small16 | 11/128 | 41/388 | 91/636 | 0/0 |
| prefix_grid | reference512 | 13/128 | 76/388 | 137/636 | 0/0 |
| prefix_graph | small16 | 5/128 | 33/388 | 72/636 | 0/0 |
| prefix_graph | reference512 | 19/128 | 96/388 | 129/636 | 0/0 |

These ID graphs have no unreachable nodes. Their 0/0 cells are unobserved, with
null accuracy in the results, not zero-percent performance. Source correctness
is a simple prerequisite (its answer should be zero), but the diagnosis does not
infer a unique source-index, tokenizer, optimization or serialization defect.
Valid-vector-conditional role denominators and each task/size cell are in the CSV.

## Signed and first-error distributions

Under/equal/over compare generated values to truth only among VALID vectors.
Negative/reachable counts cannot be interpreted as an unconditional rate.
First wrong is an OUTPUT NODE INDEX, not graph distance; its histogram excludes
exact and invalid vectors. All counts, including zero categories, remain in JSON.

| Arm | Variant | Under / equal / over | Valid nodes | Negative on reachable | First-wrong index:count (valid erroneous vectors) |
| --- | --- | --- | --- | --- | --- |
| scratch | small16 | 323 / 144 / 309 | 776 | 0 | 0:59, 1:15, 2:11 (85) |
| scratch | reference512 | 484 / 247 / 411 | 1142 | 0 | 0:94, 1:20, 2:9, 3:3, 5:1 (127) |
| grid_nca | small16 | 248 / 115 / 273 | 636 | 0 | 0:46, 1:17, 2:5 (68) |
| grid_nca | reference512 | 510 / 218 / 424 | 1152 | 0 | 0:105, 1:17, 2:3, 3:2, 5:1 (128) |
| graph_nca | small16 | 257 / 101 / 242 | 600 | 0 | 0:52, 1:9, 2:4, 3:1 (66) |
| graph_nca | reference512 | 492 / 243 / 417 | 1152 | 0 | 0:92, 1:24, 2:10, 3:2 (128) |
| prefix_grid | small16 | 283 / 143 / 296 | 722 | 0 | 0:58, 1:13, 2:8, 3:1 (80) |
| prefix_grid | reference512 | 471 / 226 / 439 | 1136 | 0 | 0:103, 1:15, 2:8 (126) |
| prefix_graph | small16 | 245 / 110 / 279 | 634 | 0 | 0:49, 1:14, 2:5, 3:1 (69) |
| prefix_graph | reference512 | 500 / 244 / 390 | 1134 | 0 | 0:87, 1:28, 2:6, 3:3, 4:1, 5:1 (126) |

Many valid erroneous vectors already fail at output index zero. This is a
descriptive ordering fact; it does not demonstrate that later failures are
caused by the first error, exposure bias or an inability to propagate messages.
Teacher-forced digit failures also occur with correct earlier answers supplied.

## Paired held-out changes

Small16 minus historical512, on the SAME 128 examples/1152 nodes per arm.
Gained means only small16 is correct/valid; lost means only reference512 is.
Both/neither categories and identical cache comparisons remain in the full CSV.

| Arm | Format gained / lost | Node correct gained / lost |
| --- | --- | --- |
| scratch | 1 / 43 | 95 / 198 |
| grid_nca | 0 / 60 | 71 / 174 |
| graph_nca | 0 / 62 | 60 / 202 |
| prefix_grid | 0 / 46 | 84 / 167 |
| prefix_graph | 2 / 59 | 77 / 211 |

This is a descriptive paired endpoint comparison, not an independent-cohort
test or causal separation of cache content, diversity, repetition and lengths.
Do not turn overlapping arm ranges into graph no-effect/equivalence evidence.

## Verification, cost and next decision

A separate invocation rescores all 1440 retained outputs with a different
strict decimal parser and all-pairs hop algorithm. All 100 diagnostic rows and
50 paired rows, serialized CSVs and input/source identities reproduce exactly.
Aggregation remains shared code; teacher counters and the stage 15 checkpoint
replay are imported evidence, not independent neural verification. New model
forwards, generations, optimizer updates, PPT/language runs and OOD predictions:
ZERO. Analysis/replay wall times are retained in the append-only log; original
fit/replay and initialization costs remain in stages 13–15.

Run `python -m error_audit_v16.run verify` through `scripts/run_logged.py` to
reproduce from distributable evidence without private weights, tokenizer cache,
raw caches or network. [Usage](../../ERROR_AUDIT_V16_USAGE.md) distinguishes
this semantic replay from the original neural replay. No remote/paid job,
official language test, email, push or external publication is performed.

Preserve all endpoints and gates. The next economical DEVELOPMENT proposal is
an intermediate 64-example cache (16 graphs per size, both tasks), at the same
600 updates and from the same five original pre-adaptation starts, using the
same outcome-blind nested ranking. Retain both 16 and 512 references on identical
evaluation examples. Change cache size alone; do not also extend training,
alter formatting/objective or choose a checkpoint. This probes a fitting/diversity
tradeoff; it does not promise a remedy or isolate repetition causally. Freeze its
data, budget and criteria separately BEFORE any fit. No such fit runs here.

The original categorical/random-rule readiness failure, separate readiness
bridge and OOD/full-language gates remain unchanged. A real graph-specific
confirmation still requires eligible synthetic starts, fresh paired rule/data/
topology cohorts, identical full language/adaptation procedures, scratch/grid/
graph/both prefix controls, ID-only selection, frozen OOD criteria/censoring/
multiplicity and total costs. This audit does not satisfy those prerequisites.
