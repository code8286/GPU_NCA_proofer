"""Explicit denominators for retained generated answers and teacher counters."""
from collections import Counter
from statistics import mean
import math
from ood_v12.scoring import parse_answer
from ood_v12.graphs import distances

ROLES = ('source', 'one_hop', 'multi_hop', 'unreachable')


def indexed(values, expected):
    table = {v['id']: v for v in values}
    if len(table) != len(values) or set(table) != set(expected):
        raise ValueError('Complete, unique, exactly matched case coverage required')
    return table


def rate(a, b):
    return a / b if b else None


def annotate(rows, predictions, independent=False):
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('Nonempty unique dataset required')
    table = indexed(predictions, ids)
    out = []
    for row in rows:
        p = table[row['id']]; g = row['graph']; n = g['n']
        if independent:
            # Independent parser and all-pairs algorithm, not primary helpers.
            from scripts.check_id_audit_v14_review import decimal_vector
            from ood_v12.graphs import floyd
            parsed = decimal_vector(p['text'], n) if p.get('status', 'ok') == 'ok' else None
            hops = floyd(n, g['edges'], g['source'], False)
        else:
            parsed = parse_answer(p['text'], n) if p.get('status', 'ok') == 'ok' else None
            hops = distances(n, g['edges'], g['source'], False)
        roles = ['source' if h == 0 else 'one_hop' if h == 1 else 'multi_hop' if h >= 2 else 'unreachable' for h in hops]
        correct = [parsed is not None and a == b for a, b in zip(parsed or [None]*n, row['truth'])]
        wrong = [i for i, ok in enumerate(correct) if not ok]
        out.append({'id': row['id'], 'task': row['task'], 'n': n, 'base_id': row['base_id'],
                    'source': g['source'], 'truth': row['truth'], 'parsed': parsed,
                    'valid': parsed is not None, 'coverage': p.get('status', 'ok') == 'ok',
                    'correct': correct, 'roles': roles,
                    'first_wrong': wrong[0] if parsed is not None and wrong else None})
    return out


def aggregate(cases, teacher):
    table = indexed(teacher, [v['id'] for v in cases])
    c = Counter(); tf = Counter(); first = Counter(); macro = []
    for v in cases:
        valid = v['valid']; n = v['n']; ok = v['correct']
        c.update(examples=1, nodes=n, valid_examples=int(valid), covered_examples=int(v['coverage']),
                 exact_examples=int(all(ok)), correct_nodes=sum(ok), valid_nodes=n*int(valid))
        macro.append(sum(ok)/n)
        for i, role in enumerate(v['roles']):
            c[role+'_nodes'] += 1; c[role+'_correct'] += int(ok[i]); c[role+'_valid_nodes'] += int(valid)
            if valid:
                a, b = v['parsed'][i], v['truth'][i]
                c['under'] += a < b; c['equal'] += a == b; c['over'] += a > b
                c['negative_reachable'] += b >= 0 and a < 0
        if v['first_wrong'] is not None:
            first[str(v['first_wrong'])] += 1
        t = table[v['id']]
        if (t['targets'] != t['digit_targets']+t['other_targets'] or
                t['correct'] != t.get('digit_correct',0)+t.get('other_correct',0) or
                not math.isclose(t['nll_sum'],t.get('digit_nll_sum',0.)+t.get('other_nll_sum',0.),rel_tol=1e-12,abs_tol=1e-12)):
            raise ValueError('Teacher token partitions differ')
        for prefix in ('', 'digit_', 'other_'):
            count = t[prefix+'targets']; correct = t.get(prefix+'correct',0); loss = t.get(prefix+'nll_sum',0.)
            if not 0 <= correct <= count or loss < 0 or not math.isfinite(loss):
                raise ValueError('Invalid teacher counters')
            tf[prefix+'targets'] += count; tf[prefix+'correct'] += correct; tf[prefix+'nll_sum'] += loss
    result = dict(c)
    for k in ('under','equal','over','negative_reachable'):
        result[k] = c[k]
    result.update(exact_accuracy=c['exact_examples']/c['examples'], validity=c['valid_examples']/c['examples'],
                  coverage=c['covered_examples']/c['examples'], node_macro_accuracy=mean(macro),
                  node_micro_accuracy=c['correct_nodes']/c['nodes'],
                  valid_node_accuracy=rate(c['correct_nodes'],c['valid_nodes']),
                  first_wrong_counts=dict(sorted(first.items(),key=lambda v:int(v[0]))), first_wrong_vectors=sum(first.values()),
                  graphs=len({v['base_id'] for v in cases}))
    for role in ROLES:
        for suffix in ('nodes','correct','valid_nodes'):
            result[role+'_'+suffix] = c[role+'_'+suffix]
        result[role+'_accuracy'] = rate(c[role+'_correct'],c[role+'_nodes'])
        result[role+'_valid_accuracy'] = rate(c[role+'_correct'],c[role+'_valid_nodes'])
    for prefix in ('', 'digit_', 'other_'):
        tf[prefix+'accuracy'] = rate(tf[prefix+'correct'],tf[prefix+'targets'])
        tf[prefix+'nll'] = rate(tf[prefix+'nll_sum'],tf[prefix+'targets'])
    return result | {'teacher':dict(tf)}


def paired(small, reference):
    table = indexed(reference,[v['id'] for v in small]); c = Counter()
    for a in small:
        b = table[a['id']]
        if a['truth'] != b['truth'] or a['roles'] != b['roles']:
            raise ValueError('Paired example labels differ')
        category = lambda x,y: 'both' if x and y else 'gained' if x else 'lost' if y else 'neither'
        c['format_'+category(a['valid'],b['valid'])] += 1
        for x,y in zip(a['correct'],b['correct']):
            c['node_'+category(x,y)] += 1
    return {f'{kind}_{label}':c[f'{kind}_{label}'] for kind in ('format','node') for label in ('both','gained','lost','neither')}
