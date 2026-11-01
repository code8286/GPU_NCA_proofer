"""Strict free-generation output scoring, paired correctness and shift reporting."""
from collections import defaultdict
import re
from statistics import mean


def parse_answer(text, n):
    if not isinstance(text, str) or not re.fullmatch(r'\s*\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]\s*', text):
        return None
    values = [int(x) for x in re.findall(r'-?\d+', text)]
    return values if len(values) == n and all(x >= -1 for x in values) else None


def score(rows, predictions):
    if len(predictions) != len(rows) or len({p['id'] for p in predictions}) != len(predictions):
        raise ValueError('Every example needs exactly one prediction; no duplicate/missing rows')
    table = {p['id']: p for p in predictions}
    if set(table) != {r['id'] for r in rows}:
        raise ValueError('Prediction and dataset identities differ')
    cases, groups, pairs = [], defaultdict(list), defaultdict(dict)
    for row in rows:
        pred = table[row['id']]
        if not isinstance(pred.get('text'), str):
            raise ValueError('Prediction text must be a string')
        if pred.get('status', 'ok') != 'ok':
            parsed = None
        else:
            parsed = parse_answer(pred['text'], row['graph']['n'])
        correct = parsed == row['truth']
        node_accuracy = mean(a == b for a, b in zip(parsed, row['truth'])) if parsed is not None else 0.
        aux = row['auxiliary']
        cell = (row['task'], aux['regime'], aux['family'], aux['length'], row['view'])
        item = {'id': row['id'], 'base_id': row['base_id'], 'task': row['task'], 'view': row['view'],
                'cell': aux['cell'], 'valid_format': parsed is not None, 'exact_match': correct,
                'node_accuracy': node_accuracy, 'status': pred.get('status', 'ok'), 'parsed': parsed}
        cases.append(item)
        groups[cell].append(item)
        pairs[row['base_id'], row['task']][row['view']] = item
    aggregate = []
    for key, values in sorted(groups.items()):
        task, regime, family, n, view = key
        aggregate.append({'task': task, 'regime': regime, 'family': family, 'n': n, 'view': view, 'count': len(values),
                          'coverage': mean(v['status'] == 'ok' for v in values),
                          **{k: mean(v[k] for v in values) for k in ('valid_format', 'exact_match', 'node_accuracy')}})
    paired = []
    for (base, task), views in sorted(pairs.items()):
        if set(views) != {'native', 'renamed', 'edge_order', 'edge_cut'}:
            continue
        native = views['native']
        paired.append({'base_id': base, 'task': task, 'cell': native['cell'],
                       'native_and_cut_correct': bool(native['exact_match'] and views['edge_cut']['exact_match']),
                       'native_and_renamed_correct': bool(native['exact_match'] and views['renamed']['exact_match']),
                       'native_and_edge_order_correct': bool(native['exact_match'] and views['edge_order']['exact_match']),
                       'rename_consistent': native['parsed'] is not None and native['parsed'] == views['renamed']['parsed']})
    return {'cases': cases, 'cells': aggregate, 'paired': paired,
            'overall_exact_match': mean(r['exact_match'] for r in cases),
            'overall_valid_format': mean(r['valid_format'] for r in cases),
            'coverage': mean(r['status'] == 'ok' for r in cases),
            'comparison_eligible': all(r['status'] == 'ok' for r in cases),
            'scope': 'Development diagnostic; cell-level scores primary, pooled average descriptive; no acceleration claim'}


def accuracy_crossing(curve, target):
    """First observed ID-development crossing; no interpolation or OOD selection."""
    if not 0 < target <= 1:
        raise ValueError('Predeclare an accuracy target in (0,1]')
    previous = -1
    for row in curve:
        if row['step'] <= previous or row.get('selection_split') != 'id_development':
            raise ValueError('Strictly ordered ID-development curve required')
        previous = row['step']
    for row in curve:
        if row['exact_match'] >= target:
            return {'reached': True, 'step': row['step'], 'tokens': row['tokens'], 'seconds': row['seconds']}
    return {'reached': False, 'step': None, 'tokens': None, 'seconds': None}


def require_paired_transfer(records):
    """Fail closed before interpreting future matched downstream probe results."""
    arms = {'scratch', 'grid_nca', 'graph_nca', 'prefix_grid', 'prefix_graph'}
    if len(records) != 5 or {r['arm'] for r in records} != arms:
        raise ValueError('Retain scratch, grid, graph and both matched controls')
    same = ('seed', 'model_config_sha256', 'language_updates', 'language_order_sha256',
            'language_schedule_sha256', 'language_head_initial_sha256', 'probe_updates',
            'probe_order_sha256', 'probe_schedule_sha256', 'probe_head_initial_sha256', 'ood_plan_sha256')
    if any(len({r[k] for r in records}) != 1 for k in same):
        raise ValueError('Both downstream stages must match, including heads/orders/schedules')
    for row in records:
        if row['arm'] != 'scratch' and (not row['synthetic_readiness_verified'] or not row['readiness_evidence_sha256']):
            raise ValueError('New confirmatory transfer requires verified synthetic readiness evidence')
    return True


def score_official(records, predictions):
    """Strict final-string scoring for official prompt/references JSON exports.

    This adapter preserves official text; it does not reinterpret predecessor
    vectors as distances or silently convert this local diagnostic to CLRS.
    """
    import hashlib
    import json
    converted = []
    for row in records:
        if not isinstance(row.get('prompt'), str) or not row.get('references') or not all(isinstance(x, str) for x in row['references']):
            raise ValueError('Expected official prompt/references records')
        identity = row.get('id') or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        converted.append((identity, row['references']))
    index = {p['id']: p for p in predictions}
    if len(index) != len(predictions) or len({r[0] for r in converted}) != len(converted) or set(index) != {r[0] for r in converted}:
        raise ValueError('Official export prediction identities must match exactly')
    rows = [{'id': key, 'exact_match': index[key].get('status', 'ok') == 'ok' and index[key]['text'].strip() in [s.strip() for s in refs]} for key, refs in converted]
    return {'cases': rows, 'exact_match': mean(r['exact_match'] for r in rows),
            'scope': 'Official-export final-reference string match; no local distance reinterpretation'}
