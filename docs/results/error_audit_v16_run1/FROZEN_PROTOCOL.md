# Stage 16: frozen cache-versus-held-out error audit

Local CPU development analysis of retained stage 15 and imported stage 13/14
outputs. Freeze this protocol, source hashes and input hashes before computing
the new diagnostic tables. This is a descriptive audit of already inspected
development evidence, not a new confirmatory experiment or fit.

## Fixed scope

Retain scratch, grid_nca, graph_nca, prefix_grid and prefix_graph, both the
small16 and historical512 variants, and only the fixed update-600 endpoints.
Use the SAME 16 cache examples and full 128 ID-development examples in each
variant. Regenerate their graph/answer identities using stage 15's frozen
selection. No checkpoint selection, extra updates, new model forward passes,
OOD data, assessment predictions, topology/serializer changes or new targets.
Bind all 52 stage 15 evidence copies and its provenance manifest. Their replay
receipts are imported evidence, not a new neural replay in this stage.

## Measurements and denominators

Report pooled and all four task/size cells, without dropping invalid outputs.
There are 100 rows: five arms x two variants x two probes x five groups.

1. Strict vector validity, exact answers, coverage, example-macro node accuracy
   and node-micro accuracy. Invalid outputs count zero on every node. Also
   report valid-vector-conditional node accuracy with its node denominator;
   never substitute it for the unconditional score.
2. Source, one-hop, multi-hop and unreachable node counters, unconditional and
   valid-vector-conditional accuracies. Hop bins use unweighted shortest paths
   for BOTH tasks; Dijkstra truth still uses original weighted distances.
   Include empty bins as null accuracy, never perfect or zero observed accuracy.
3. Signed under/equal/over counts among valid generated nodes; negative answers
   on reachable nodes. Count first wrong output-node index among VALID erroneous
   vectors only. Preserve a histogram and its denominator. Invalid/exact vectors
   have no first wrong index. These are output order positions, not graph hops.
4. Teacher-forced digit-bearing and other token correct/target/NLL counters,
   summed over retained per-case counters, and token-micro rates. Check token
   partitions and exact case coverage. Gold earlier answers are supplied in
   teacher forcing; its digit tokens can span punctuation, and cannot establish
   semantic per-node correctness or free-generation competence.
5. Paired small16-versus-reference512 counts on identical examples: format
   gained/lost/both/neither and node correctness gained/lost/both/neither, with
   invalid outputs wrong. Report per cell and pooled; do not infer independent
   samples from the two tasks on the same graph. Cache-versus-development
   differences are unpaired distribution comparisons.

Independently check generated correctness, hop partitions and first wrong
indices using a separate strict parser and Floyd-Warshall hop computation in
verification. Recompute the complete result in a separate invocation and bind
its SHA256 receipt. Preserve failures and their resolution through the logger.
Public replay needs no weights, tokenizer cache, network or private data. It
reaggregates imported teacher counters; it cannot independently reproduce neural
logits, teacher NLLs or training from scratch.

## Decision and costs

There is no new pass threshold, hypothesis test, confidence interval or model
ranking. Report every arm/cell, denominators and the full signed/first-error
distribution regardless of outcome. Retain the stage 15 cache-only positive
control and failed full-ID endpoint. All original synthetic readiness, separate
categorical/random-rule bridge and OOD eligibility gates remain unchanged.

New optimizer updates, model generations and model forwards are ZERO. Record
analysis/replay wall time separately in logs. Imported fit/replay costs remain
in their original studies and are not zero-cost initialization. A later cache-
diversity OR training-budget treatment needs its own separately frozen protocol,
same starts and all controls; this audit cannot choose a winning endpoint or
prove which treatment will solve generalization. No remote/paid job, email,
official-language-test access, push or external publication is authorized here.
