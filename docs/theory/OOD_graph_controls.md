# Mathematical basis of the stage 12 functional controls

These are informal arguments supporting the implemented measurement controls.
They add no claim of a new Lean check or LaTeX compilation. Read the
[experimental protocol](../OOD_V12_PROTOCOL.md) for operational scope.

## A source-edge removal must change the answer

Let G be a simple undirected graph with strictly positive edge weights. Let s
be a non-isolated source. Choose an incident edge (s,v) with minimum weight w
among all edges incident to s. The direct path has length w. Every alternative
path from s to v starts with another source edge of weight at least w and
contains at least one further strictly positive edge. Thus every alternative
has length strictly greater than w. This remains true when several incident
edges tie for the minimum: simplicity excludes a second parallel direct edge.

Consequently, deleting (s,v) strictly increases distance(s,v), or makes v
unreachable. At least one component of the full distance vector changes. Taking
all weights to be one yields the BFS case. Dijkstra uses the sampled positive
integer weights. Zero/negative weights or parallel/self edges are excluded by
the implementation; isolated sources are excluded at sampling.

The implemented deletion changes the number of edges and may change degrees
and connectivity. It tests whether the predictor follows the changed input;
it does not isolate a degree-preserving topology effect or prove a mechanism
specific to graph pre-pre-training.

## Relabeling and serialization controls

A bijective relabeling transported consistently to nodes, edges and source
preserves path lengths. The output vector follows the declared input-node
order, and the renaming view does not reorder those positions. Its distance
vector is therefore identical. Permuting the edge-list order also preserves
the graph and its answer.

Equality of two model outputs under these transformations is insufficient:
both outputs can be wrong. Report correctness of both predictions alongside
plain consistency. Conversely, an output change after edge deletion is
insufficient unless the native and changed predictions are both correct.

## Why the cached-native control scores 75% yet fails the paired test

For each native/task instance there are four views. Three have the same correct
answer: native, renaming and edge ordering. The cut has a different answer by
the first argument. A predictor that caches the native oracle answer and repeats
it scores exactly three of four examples correctly, but never gets both native
and cut correct. This construction validates that paired correctness detects
failure to use the changed input even when pooled exact match appears high.

A source-only predictor emits zero at s and -1 at every other node. The native
source has a neighbor, so at least one non-source native distance is finite.
The source-only native prediction is always wrong, hence its native-and-cut
paired correctness is zero even when a cut happens to isolate the source.

## Conservative structural exclusion

Initially, corresponding nodes of isomorphic unweighted graphs have equal
degrees. At each refinement step, their previous colors and multisets of
neighbor colors agree, so their new deterministic hash colors agree. By
induction, sorted node-color multisets after n rounds agree. Consequently every
isomorphic graph receives the same invariant key. The exclusion rejects any
key already reserved by an earlier split, including intervention graphs.

Non-isomorphic graphs can share a refinement key or hash, which causes extra
rejection. This does not permit an isomorphic graph to pass the exclusion.
This is not a complete isomorphism canonicalizer, a cryptographic impossibility
claim, subgraph-disjointness or a guarantee of independent sampled examples.
Weights, source and display labels are intentionally ignored in the exclusion.
Repeated structures within a split, especially paths, retain explicit counts.
