"""Matched probes, teacher-forced token diagnostics and generated node errors."""
from collections import Counter, defaultdict
import hashlib
from statistics import mean
import torch
from torch.nn import functional as F
from adaptation_v13.training import encode, collate
from ood_v12.scoring import score
from ood_v12.graphs import distances


def select_training(rows, c):
    selected = set()
    for size in (8,10):
        bases = {r['base_id'] for r in rows if r['graph']['n'] == size}
        rank = lambda b: (hashlib.sha256((c['selection_salt']+':'+b).encode()).hexdigest(), b)
        if len(bases) < c['training_graphs_per_size']:
            raise ValueError('Insufficient original training graphs')
        selected.update(sorted(bases,key=rank)[:c['training_graphs_per_size']])
    result = sorted([r for r in rows if r['base_id'] in selected],key=lambda r:(r['graph']['n'],r['base_id'],r['task']))
    if len(result) != 4*c['training_graphs_per_size'] or len({r['id'] for r in result}) != len(result):
        raise ValueError('Retain both tasks exactly once per graph')
    return result


def covariates(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['task'],r['graph']['n']].append(r)
    return [{'task':task,'n':n,'examples':len(values),'graphs':len({r['base_id'] for r in values}),
             'mean_edges':mean(len(r['graph']['edges']) for r in values),
             'mean_source_degree':mean(sum(r['graph']['source'] in (u,v) for u,v,_ in r['graph']['edges']) for r in values),
             'mean_target_distance':mean(mean(r['truth']) for r in values)} for (task,n),values in sorted(groups.items())]


def digit_bearing(tokenizer, token):
    if token == tokenizer.eot_token:
        return False
    return any(48 <= b <= 57 for b in tokenizer.decode_single_token_bytes(token))


@torch.no_grad()
def teacher_forced(model, tokenizer, rows, batch=8):
    model.eval()
    counts = defaultdict(Counter)
    cases = []
    for start in range(0,len(rows),batch):
        subset = rows[start:start+batch]
        examples = [encode(r,tokenizer,model.context) for r in subset]
        x,y = collate(examples)
        hidden = model.core(model.embedding(x))
        mask = y != -100
        logits = F.linear(hidden[mask],model.embedding.weight)
        truth = y[mask]
        losses = F.cross_entropy(logits,truth,reduction='none').tolist()
        predicted = logits.argmax(-1).tolist()
        offset = 0
        for row, example in zip(subset,examples):
            totals = Counter()
            targets = [t for t in example['target'] if t != -100]
            for target in targets:
                role = 'digit' if digit_bearing(tokenizer,target) else 'other'
                correct = int(predicted[offset] == target)
                totals['targets'] += 1; totals['correct'] += correct; totals['nll_sum'] += losses[offset]
                totals[role+'_targets'] += 1; totals[role+'_correct'] += correct; totals[role+'_nll_sum'] += losses[offset]
                offset += 1
            counts[row['task'],row['graph']['n']].update(totals)
            cases.append({'id':row['id'],**dict(totals)})
        if offset != len(losses):
            raise ValueError('Teacher-forced answer mask alignment differs')
    cells = []
    for (task,n), value in sorted(counts.items()):
        cell = {'task':task,'n':n,**dict(value)}
        for role in ('','digit_','other_'):
            count = value[role+'targets']
            cell[role+'accuracy'] = value[role+'correct']/count if count else None
            cell[role+'nll'] = value[role+'nll_sum']/count if count else None
        cells.append(cell)
    return {'cases':cases,'cells':cells}


def generated_errors(rows, predictions):
    value = score(rows,predictions)
    index = {r['id']:r for r in rows}
    cells = defaultdict(Counter)
    errors = []
    for case in value['cases']:
        row = index[case['id']]
        g = row['graph']
        hops = distances(g['n'],g['edges'],g['source'],False)
        counts = Counter()
        wrong = []
        for i, (truth,hop) in enumerate(zip(row['truth'],hops)):
            role = 'source' if hop == 0 else 'one_hop' if hop == 1 else 'multi_hop' if hop >= 2 else 'unreachable'
            predicted = None if case['parsed'] is None else case['parsed'][i]
            correct = predicted == truth
            counts[role+'_nodes'] += 1; counts[role+'_correct'] += int(correct)
            if not correct:
                wrong.append(i)
            if predicted is not None:
                counts['valid_nodes'] += 1
                counts['under'] += predicted < truth
                counts['equal'] += correct
                counts['over'] += predicted > truth
                counts['negative_reachable'] += truth >= 0 and predicted < 0
        counts['invalid_examples'] += case['parsed'] is None
        first = wrong[0] if wrong and case['parsed'] is not None else None
        errors.append({'id':row['id'],'first_wrong_node':first,
                       'normalized_first_wrong':first/(g['n']-1) if first is not None else None,
                       **dict(counts)})
        cells[row['task'],g['n']].update(counts)
    output = []
    for cell in value['cells']:
        totals = cells[cell['task'],cell['n']]
        output.append(cell | dict(totals) | {r+'_accuracy':totals[r+'_correct']/totals[r+'_nodes'] if totals[r+'_nodes'] else None
                                           for r in ('source','one_hop','multi_hop','unreachable')})
    return {'cases':value['cases'],'errors':errors,'cells':output,
            'exact_match':value['overall_exact_match'],'valid_format':value['overall_valid_format'],
            'node_accuracy':mean(r['node_accuracy'] for r in value['cases']),'coverage':value['coverage']}


def interpret(train, development, c):
    def competent(v):
        return (len(v['cells']) == 4 and all(x['exact_match'] >= c['exact_target'] for x in v['cells'])
                and v['valid_format'] >= c['format_target'] and v['coverage'] == 1)
    t,d = competent(train),competent(development)
    gap = train['exact_match']-development['exact_match']
    label = ('training_probe_competence_not_established' if not t else 'both_probes_competent' if d
             else 'held_out_gap_after_training_probe_competence' if gap >= c['large_gap']
             else 'cell_or_format_failure_without_large_pooled_gap')
    return {'training_probe_competent':t,'development_competent':d,'train_minus_development_exact':gap,'interpretation':label}
