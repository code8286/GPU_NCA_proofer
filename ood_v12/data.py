"""Small CLRS-inspired distance tasks; split-level structural exclusions."""
import hashlib
import json
import string
import numpy as np
from .graphs import distances, sample_graph, topology_key, intervene

LABELS = string.ascii_uppercase
TASKS = ('bfs_distances', 'dijkstra_distances')
VIEWS = ('native', 'renamed', 'edge_order', 'edge_cut')


def prompt(n, edges, source, labels, weighted):
    names = ','.join(labels)
    links = ','.join(labels[u]+labels[v]+(':'+str(w) if weighted else '') for u, v, w in edges)
    task = 'dijkstra_distances' if weighted else 'bfs_distances'
    return (f'{task}\nnodes={names};undirected_edges={links};source={labels[source]}\n'
            'Return integer distances in nodes order; -1 means unreachable.\ndistances=')


def answer(values):
    return '['+','.join(map(str, values))+']'


def make_examples(n, edges, source, rng, metadata, views=VIEWS):
    labels = rng.choice(list(LABELS), n, replace=False).tolist()
    renamed = [labels[i] for i in rng.permutation(n)]
    if renamed == labels:
        renamed = labels[1:]+labels[:1]
    reordered = [edges[i] for i in rng.permutation(len(edges))]
    if reordered == edges and len(edges) > 1:
        reordered = edges[::-1]
    result = []
    base = hashlib.sha256(json.dumps([n, edges, source, metadata], sort_keys=True).encode()).hexdigest()
    for task in TASKS:
        weighted = task == 'dijkstra_distances'
        cut = intervene(n, edges, source, weighted)
        for view in views:
            use_edges = cut if view == 'edge_cut' else reordered if view == 'edge_order' else edges
            use_labels = renamed if view == 'renamed' else labels
            truth = distances(n, use_edges, source, weighted)
            result.append({'id': f'{base}:{task}:{view}', 'base_id': base, 'task': task, 'view': view,
                           'prompt': prompt(n, use_edges, source, use_labels, weighted), 'references': [answer(truth)],
                           'truth': truth, 'graph': {'n': n, 'edges': use_edges, 'source': source, 'labels': use_labels},
                           'auxiliary': metadata | {'length': n, 'use_hints': False,
                               'topology_key': topology_key(n, edges), 'benchmark_kind': 'clrs_inspired_distances_v12'}})
    return result


def cells(c, split):
    if split in ('train', 'development'):
        count = c[split+'_graphs_per_cell']
        return [('id', 'cyclic_er', n, count) for n in c['id_sizes']]
    count = c['assessment_graphs_per_cell']
    result = [('id', 'cyclic_er', n, count) for n in c['id_sizes']]
    result += [('size', 'cyclic_er', n, count) for n in c['ood_sizes']]
    result += [('family', f, n, count) for n in c['id_sizes'] for f in c['families'][1:]]
    result += [('joint', f, n, count) for n in c['ood_sizes'] for f in c['families'][1:]]
    return result


def generate(c):
    rng = np.random.default_rng(c['data_seed'])
    prior_keys, arrays, info = set(), {}, {}
    for split in ('train', 'development', 'assessment'):
        rows, current_keys = [], set()
        for regime, family, n, count in cells(c, split):
            accepted = attempts = 0
            while accepted < count:
                attempts += 1
                if attempts > 100000:
                    raise RuntimeError('Split exclusion exhausted; no identity-policy relaxation')
                edges = sample_graph(n, family, rng)
                key = topology_key(n, edges)
                # Multiple rooted/labeled/weighted versions may occur inside a
                # split, particularly paths. Report topology-cluster counts;
                # never call these independent topology replicates.
                source = int(rng.integers(n))
                if not any(source in (u, v) for u, v, _ in edges):
                    continue
                metadata = {'split': split, 'regime': regime, 'family': family, 'cell': f'{regime}_{family}_{n}', 'ordinal': accepted}
                examples = make_examples(n, edges, source, rng, metadata)
                keys = {topology_key(n, r['graph']['edges']) for r in examples}
                if keys & prior_keys:
                    continue
                rows.extend(examples if split == 'assessment' else [r for r in examples if r['view'] == 'native'])
                current_keys.update(keys)
                accepted += 1
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate example identity')
        prior_keys.update(current_keys)
        arrays[split] = rows
        info[split] = {'examples': len(rows), 'base_graph_instances': len({r['base_id'] for r in rows}),
                       'base_topology_clusters': len({r['auxiliary']['topology_key'] for r in rows}),
                       'reserved_topology_clusters_including_cuts': len(current_keys)}
    return arrays, info


def public_input(row):
    """Only prompt/identity reach a predictor; truth/graph metadata stay in scorer."""
    return {'id': row['id'], 'prompt': row['prompt']}


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(',', ':'))+'\n')
