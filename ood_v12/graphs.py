"""Independent graph oracles, topology fingerprints, and matched interventions."""
from collections import Counter, deque
import hashlib
import heapq
import json
import numpy as np


def distances(n, edges, source, weighted=False):
    adjacency = [[] for _ in range(n)]
    if not 0 <= source < n:
        raise ValueError('Invalid source')
    seen = set()
    for u, v, w in edges:
        if not (0 <= u < v < n and isinstance(w, int) and 1 <= w <= 9) or (u, v) in seen:
            raise ValueError('Expected unique undirected positive-integer edges')
        seen.add((u, v))
        adjacency[u].append((v, w if weighted else 1))
        adjacency[v].append((u, w if weighted else 1))
    result = [float('inf')]*n
    result[source] = 0
    if weighted:
        queue = [(0, source)]
        while queue:
            cost, u = heapq.heappop(queue)
            if cost != result[u]:
                continue
            for v, weight in adjacency[u]:
                if cost+weight < result[v]:
                    result[v] = cost+weight
                    heapq.heappush(queue, (cost+weight, v))
    else:
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, _ in adjacency[u]:
                if result[v] == float('inf'):
                    result[v] = result[u]+1
                    queue.append(v)
    return [-1 if x == float('inf') else int(x) for x in result]


def floyd(n, edges, source, weighted=False):
    """Different all-pairs implementation, used to audit labels and interventions."""
    d = np.full((n, n), 10000, dtype=np.int64)
    np.fill_diagonal(d, 0)
    for u, v, w in edges:
        d[u, v] = d[v, u] = w if weighted else 1
    for k in range(n):
        d = np.minimum(d, d[:, k, None]+d[None, k, :])
    return [-1 if x >= 10000 else int(x) for x in d[source]]


def topology_key(n, edges):
    """Unweighted WL invariant: isomorphs collide; other collisions are conservative.

    This is NOT a complete graph-isomorphism canonicalizer. A collision rejects
    extra graphs across splits; it cannot allow an isomorphic graph to leak.
    Weights/source/printed labels do not affect this conservative exclusion.
    """
    adj = [set() for _ in range(n)]
    for u, v, _ in edges:
        adj[u].add(v)
        adj[v].add(u)
    colors = [str(len(a)) for a in adj]
    for _ in range(n):
        colors = [hashlib.sha256(json.dumps([colors[i], sorted(colors[j] for j in adj[i])]).encode()).hexdigest() for i in range(n)]
    return hashlib.sha256(json.dumps([n, sorted(colors)]).encode()).hexdigest()


def sample_graph(n, family, rng):
    def er(nodes):
        return [(u, v) for i, u in enumerate(nodes) for v in nodes[i+1:] if rng.random() < min(.7, 2.5/(len(nodes)-1))]
    for _ in range(20000):
        if family == 'cyclic_er':
            pairs = er(list(range(n)))
        elif family == 'branching_tree':
            # Random recursive tree, then independent random node relabeling.
            order = rng.permutation(n).tolist()
            pairs = [tuple(sorted((order[i], order[int(rng.integers(i))]))) for i in range(1, n)]
        elif family == 'path':
            order = rng.permutation(n).tolist()
            pairs = [tuple(sorted((u, v))) for u, v in zip(order[:-1], order[1:])]
        elif family == 'two_components':
            order = rng.permutation(n).tolist()
            pairs = []
            for nodes in (order[:n//2], order[n//2:]):
                pairs += [tuple(sorted((nodes[i], nodes[int(rng.integers(i))]))) for i in range(1, len(nodes))]
        else:
            raise ValueError('Unknown graph family')
        if not pairs or len(pairs) > 3*n:
            continue
        edges = [(int(u), int(v), int(rng.integers(1, 10))) for u, v in sorted(pairs)]
        reach = distances(n, edges, 0)
        degree = Counter(x for u, v, _ in edges for x in (u, v))
        if family == 'cyclic_er' and (-1 in reach or len(edges) < n+1):
            continue
        if family == 'branching_tree' and max(degree.values()) < 3:
            continue
        if family == 'two_components':
            groups = set()
            for u in range(n):
                groups.add(tuple(i for i, d in enumerate(distances(n, edges, u)) if d >= 0))
            if len(groups) != 2 or any(len(g) != size for g, size in zip(sorted(groups, key=len), sorted([n//2, n-n//2]))):
                continue
        return edges
    raise RuntimeError('Declared graph sampling exhausted; no distribution relaxation')


def intervene(n, edges, source, weighted):
    # Cheapest source edge is strictly shortest to its neighbor under positive
    # weights. Removing it changes at least that distance, even with path ties.
    incident = [(w if weighted else 1, u, v, w) for u, v, w in edges if source in (u, v)]
    _, u, v, w = min(incident)
    changed = [e for e in edges if e != (u, v, w)]
    assert distances(n, edges, source, weighted) != distances(n, changed, source, weighted)
    return changed
