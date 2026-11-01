"""Freeze, validate controls and cross-evaluate retained models; zero fitting."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
import time
import torch
from transfer_v4.common import dump, sha_file
from transfer_v4.model import LM
from .data import read_jsonl, write_jsonl, public_input, answer
from .graphs import distances, floyd, topology_key
from .inference import greedy
from .scoring import score, score_official
from .plan import ROOT, read, freeze, verify_plan, tokenizer as get_tokenizer


def compact(value):
    return {k:v for k,v in value.items() if k not in ('cases', 'paired')} | {
        'cases': len(value['cases']), 'paired_cases': len(value['paired']),
        'native_and_cut_correct': mean(r['native_and_cut_correct'] for r in value['paired']) if value['paired'] else None,
        'native_and_renamed_correct': mean(r['native_and_renamed_correct'] for r in value['paired']) if value['paired'] else None,
        'native_and_edge_order_correct': mean(r['native_and_edge_order_correct'] for r in value['paired']) if value['paired'] else None}


def control_results(out):
    plan = verify_plan(out)
    rows = read_jsonl(out/'data/assessment.jsonl')
    split_keys, totals = {}, Counter()
    for split in ('train', 'development', 'assessment'):
        data = read_jsonl(out/f'data/{split}.jsonl')
        keys = set()
        for row in data:
            g = row['graph']
            truth = floyd(g['n'], g['edges'], g['source'], row['task'] == 'dijkstra_distances')
            if truth != row['truth'] or row['references'] != [answer(truth)]:
                raise ValueError('Independent all-pairs oracle disagrees')
            keys.add(topology_key(g['n'], g['edges']))
        if any(keys & old for old in split_keys.values()):
            raise ValueError('Native or intervention topology leaked between splits')
        split_keys[split] = keys
        totals[split] = len(data)
    index = {(r['base_id'], r['task'], r['view']): r for r in rows}
    oracle, cached, source = [], [], []
    for r in rows:
        native = index[r['base_id'], r['task'], 'native']
        if r['view'] in ('renamed', 'edge_order') and native['truth'] != r['truth']:
            raise ValueError('Invariance-control truth changed')
        if r['view'] == 'edge_cut' and native['truth'] == r['truth']:
            raise ValueError('Counterfactual has no causal output change')
        oracle.append({'id': r['id'], 'text': r['references'][0]})
        cached.append({'id': r['id'], 'text': native['references'][0]})
        values = [-1]*r['graph']['n']
        values[r['graph']['source']] = 0
        source.append({'id': r['id'], 'text': answer(values)})
    results = {k: compact(score(rows, values)) for k, values in [('oracle', oracle), ('cached_native_answer', cached), ('source_only', source)]}
    if results['oracle']['overall_exact_match'] != 1 or results['oracle']['native_and_cut_correct'] != 1 or results['cached_native_answer']['native_and_cut_correct'] != 0 or results['source_only']['native_and_cut_correct'] != 0:
        raise ValueError('Positive/causal negative measurement controls failed')
    return {'plan_sha256': sha_file(out/'plan.json'), 'passed': True, 'independent_oracle_examples': dict(totals),
            'actual_graph_topology_clusters': {k:len(v) for k,v in split_keys.items()}, 'controls': results,
            'new_optimizer_updates': 0, 'scope': plan['scope'], 'language_training_allowed': False}


def controls(out):
    if (out/'controls.json').exists():
        raise FileExistsError('Preserve first control result; use verify')
    result = control_results(out)
    dump(out/'controls.json', result)
    dump(out/'controls_receipt.json', {'sha256': sha_file(out/'controls.json')})
    print('PASS: independent graph oracles, structural split/cut exclusions, all OOD oracle scores, invariance and causal negative controls.')


def checkpoint_predictions(out, record):
    p = verify_plan(out)
    if not read(out/'controls.json')['passed'] or sha_file(out/'controls.json') != read(out/'controls_receipt.json')['sha256']:
        raise ValueError('Validated measurement controls required first')
    tokenizer = get_tokenizer()
    if tokenizer.n_vocab != p['tokenizer']['vocab']:
        raise ValueError('Tokenizer vocabulary changed')
    saved = torch.load(ROOT/record['checkpoint'], map_location='cpu', weights_only=True)
    if saved['stage'] != 'language' or saved['arm'] != record['arm'] or saved['seed'] != record['seed'] or saved['language_step'] != record['language_updates']:
        raise ValueError('Retained final language endpoint required')
    if saved['vocab'] != tokenizer.n_vocab or saved['config']['model']['context'] != p['config']['context_limit']:
        raise ValueError('No tokenizer/context substitution or extrapolation')
    model = LM(saved['config']['model'], saved['vocab'], saved['seed'], saved['seed']+100000)
    model.load_state_dict(saved['model'], strict=True)
    rows = read_jsonl(out/'data/legacy_pilot.jsonl')
    return greedy(model, tokenizer, [public_input(r) for r in rows], p['config']['generation_tokens'])


def legacy(out):
    p = verify_plan(out)
    if (out/'legacy_results.json').exists():
        raise FileExistsError('Preserve first historical prediction; use verify')
    rows, values, evidence = read_jsonl(out/'data/legacy_pilot.jsonl'), [], {}
    for record in p['legacy_checkpoints']:
        start = time.perf_counter()
        predictions = checkpoint_predictions(out, record)
        name = f"predictions/{record['arm']}_{record['seed']}.jsonl"
        write_jsonl(out/name, predictions)
        evidence[name] = sha_file(out/name)
        result = score(rows, predictions)
        stop = dict(Counter(v['stop_reason'] for v in predictions))
        values.append({'arm': record['arm'], 'seed': record['seed'], 'checkpoint_sha256': record['checkpoint_sha256'],
                       'score': compact(result), 'stop_reasons': stop, 'prediction_seconds': time.perf_counter()-start})
        print(record['arm'], 'exact=', result['overall_exact_match'], 'valid=', result['overall_valid_format'], 'coverage=', result['coverage'], stop, flush=True)
    dump(out/'legacy_results.json', {'plan_sha256': sha_file(out/'plan.json'), 'models': values, 'prediction_files_sha256': evidence,
          'new_optimizer_updates': 0, 'scope': 'One already-inspected v4 cohort, zero-shot on new task format; no task-adaptation training; exploratory calibration, not a graph-acceleration test',
          'original_random_rule_readiness': 'still_failed', 'graph_acceleration_claim': None,
          'reserved_confirmation_generated': False, 'language_training_allowed': False})
    dump(out/'legacy_results_receipt.json', {'sha256': sha_file(out/'legacy_results.json')})


def verify(out):
    p = verify_plan(out)
    if control_results(out) != read(out/'controls.json'):
        raise ValueError('Measurement controls do not reproduce')
    result = read(out/'legacy_results.json')
    if sha_file(out/'legacy_results.json') != read(out/'legacy_results_receipt.json')['sha256']:
        raise ValueError('Legacy result identity changed')
    rows = read_jsonl(out/'data/legacy_pilot.jsonl')
    for record in p['legacy_checkpoints']:
        name = f"predictions/{record['arm']}_{record['seed']}.jsonl"
        saved = read_jsonl(out/name)
        if sha_file(out/name) != result['prediction_files_sha256'][name] or checkpoint_predictions(out, record) != saved:
            raise ValueError('Free-generation checkpoint replay differs')
        score_row = next(r for r in result['models'] if r['arm'] == record['arm'])
        if compact(score(rows, saved)) != score_row['score'] or dict(Counter(r['stop_reason'] for r in saved)) != score_row['stop_reasons']:
            raise ValueError('Replayed score/stop accounting differs')
    dump(out/'verification.json', {'passed': True, 'legacy_models': len(p['legacy_checkpoints']), 'legacy_predictions_replayed': len(rows)*len(p['legacy_checkpoints']),
          'controls_sha256': sha_file(out/'controls.json'), 'results_sha256': sha_file(out/'legacy_results.json'),
          'scope': 'Separate same-code checkpoint generation replay and independent graph-label oracle checks; no retraining or independent neural implementation'})
    print('PASS: all five retained checkpoints reproduce 160 free-generated predictions and scores exactly; independent label/control audit unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'controls', 'legacy', 'verify', 'score', 'score-official'))
    parser.add_argument('--out', type=Path, default=ROOT/'runs/ood_v12_run1')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ood_v12.json')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--report', type=Path)
    a = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if a.command == 'freeze':
        freeze(a.out, read(a.config))
    elif a.command in ('score', 'score-official'):
        if not a.dataset or not a.predictions or not a.report or a.report.exists():
            parser.error('Provide dataset/predictions and a NEW report path')
        result = (score_official if a.command == 'score-official' else score)(read_jsonl(a.dataset), read_jsonl(a.predictions))
        dump(a.report, result)
    else:
        globals()[a.command](a.out)


if __name__ == '__main__':
    main()
