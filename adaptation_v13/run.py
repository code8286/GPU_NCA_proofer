"""Frozen five-arm ID calibration; no language/PPT fitting or automatic OOD run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from transfer_v4.common import dump, sha_file, state_sha
from transfer_v4.model import LM
from ood_v12.plan import ROOT, read, verify_plan as verify_ood, tokenizer
from ood_v12.data import read_jsonl, write_jsonl, public_input
from ood_v12.graphs import floyd, topology_key
from ood_v12.inference import greedy
from ood_v12.scoring import score
from .training import encode, collate, answer_loss, order, learning_rate, nll, decision

CONFIG = ROOT/'configs/adaptation_v13.json'
PROTOCOL = ROOT/'docs/ADAPTATION_V13_PROTOCOL.md'
PRIOR = ROOT/'runs/ood_v12_run1'


def sources():
    paths = list((ROOT/'adaptation_v13').glob('*.py'))
    paths += list((ROOT/'ood_v12').glob('*.py'))
    paths += [ROOT/'transfer_v4/model.py', ROOT/'transfer_v4/common.py']
    return {p.relative_to(ROOT).as_posix(): sha_file(p) for p in sorted(paths)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dataset_audit(arrays):
    previous = set()
    for split in ('train', 'development'):
        rows = arrays[split]
        keys = set()
        for row in rows:
            g = row['graph']
            if (row['view'] != 'native' or row['auxiliary']['regime'] != 'id'
                    or g['n'] not in (8, 10) or row['auxiliary']['split'] != split):
                raise ValueError('ID-only native examples required')
            if row['truth'] != floyd(g['n'], g['edges'], g['source'], row['task'] == 'dijkstra_distances'):
                raise ValueError('Independent graph oracle differs')
            keys.add(topology_key(g['n'], g['edges']))
        if keys & previous or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Split overlap or duplicate example')
        previous |= keys


def freeze(out):
    c = read(CONFIG)
    prior = verify_ood(PRIOR)
    controls = read(PRIOR/'controls.json')
    if not controls['passed'] or sha_file(PRIOR/'controls.json') != read(PRIOR/'controls_receipt.json')['sha256']:
        raise ValueError('Prior independent split/cut/oracle controls required')
    tok = tokenizer()
    arrays = {s: read_jsonl(PRIOR/f'data/{s}.jsonl') for s in ('train', 'development')}
    dataset_audit(arrays)
    encoded = {s: [encode(r, tok, 1024) for r in rows] for s, rows in arrays.items()}
    if len(arrays['train']) != 512 or len(arrays['development']) != 128:
        raise ValueError('Preserve full predeclared ID data')
    if any(len(tok.encode(r['prompt']))+c['generation_tokens'] > 1024 for r in arrays['development']):
        raise ValueError('Complete generation budget must fit')
    indices = order(512, c['updates'], c['batch'], c['order_seed'])
    out.mkdir(parents=True, exist_ok=False)
    for s in arrays:
        (out/'data').mkdir(exist_ok=True)
        shutil.copyfile(PRIOR/f'data/{s}.jsonl', out/f'data/{s}.jsonl')
    np.save(out/'order.npy', indices)
    (out/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    exposure = {'input_tokens': 0, 'supervised_targets': 0, 'padded_positions': 0}
    for batch in indices:
        rows = [encoded['train'][i] for i in batch]
        exposure['input_tokens'] += sum(len(r['input']) for r in rows)
        exposure['supervised_targets'] += sum(sum(v != -100 for v in r['target']) for r in rows)
        exposure['padded_positions'] += len(rows)*max(len(r['input']) for r in rows)
    p = {'schema_version': 13, 'created_utc': datetime.now(timezone.utc).isoformat(),
         'config': c, 'source_sha256': sources(), 'protocol_sha256': sha_file(PROTOCOL),
         'prior_plan_sha256': sha_file(PRIOR/'plan.json'), 'prior_controls_sha256': sha_file(PRIOR/'controls.json'),
         'historical_checkpoints': prior['legacy_checkpoints'],
         'input_sha256': {f'data/{s}.jsonl': sha_file(out/f'data/{s}.jsonl') for s in arrays} | {'order.npy': sha_file(out/'order.npy')},
         'encoded_sha256': {s: digest(v) for s, v in encoded.items()},
         'sampled_order_sha256': digest([[encoded['train'][i]['id'] for i in b] for b in indices]),
         'schedule_sha256': digest([learning_rate(s, c) for s in range(1, c['updates']+1)]),
         'expected_exposure_per_arm': exposure, 'tokenizer_vocab': tok.n_vocab,
         'environment': {'torch': str(torch.__version__), 'numpy': np.__version__},
         'original_random_rule_readiness': 'still_failed', 'eligible_for_graph_confirmation': False,
         'scope': 'Historical single-cohort ID interface/trainability calibration, not new PPT or language training'}
    dump(out/'plan.json', p)
    dump(out/'plan_receipt.json', {'plan_sha256': sha_file(out/'plan.json')})
    print('FROZEN', sha_file(out/'plan.json'), '5 arms x', c['updates'], 'updates;', exposure, flush=True)


def verify_plan(out):
    p = read(out/'plan.json')
    if (sha_file(out/'plan.json') != read(out/'plan_receipt.json')['plan_sha256']
            or p['config'] != read(CONFIG) or p['source_sha256'] != sources()
            or p['protocol_sha256'] != sha_file(PROTOCOL)
            or p['protocol_sha256'] != sha_file(out/'FROZEN_PROTOCOL.md')):
        raise ValueError('Frozen plan/source/protocol changed')
    prior = verify_ood(PRIOR)
    if (p['prior_plan_sha256'] != sha_file(PRIOR/'plan.json') or p['historical_checkpoints'] != prior['legacy_checkpoints']
            or p['prior_controls_sha256'] != sha_file(PRIOR/'controls.json')):
        raise ValueError('Historical inputs changed')
    for name, checksum in p['input_sha256'].items():
        if sha_file(out/name) != checksum:
            raise ValueError('Frozen data/order changed: '+name)
    return p


def load_inputs(out, p):
    tok = tokenizer()
    arrays = {s: read_jsonl(out/f'data/{s}.jsonl') for s in ('train', 'development')}
    encoded = {s: [encode(r, tok, 1024) for r in rows] for s, rows in arrays.items()}
    if {s: digest(v) for s, v in encoded.items()} != p['encoded_sha256'] or tok.n_vocab != p['tokenizer_vocab']:
        raise ValueError('Tokenization changed')
    return tok, arrays, encoded


def model_from(saved):
    model = LM(saved['config']['model'], saved['vocab'], saved['seed'], saved['seed']+100000)
    model.load_state_dict(saved['model'], strict=True)
    return model


def evaluate(model, tok, arrays, encoded, c):
    predictions = greedy(model, tok, [public_input(r) for r in arrays['development']], c['generation_tokens'])
    value = score(arrays['development'], predictions)
    return predictions, {'exact_match': value['overall_exact_match'], 'valid_format': value['overall_valid_format'],
                         'coverage': value['coverage'], 'node_accuracy': sum(r['node_accuracy'] for r in value['cases'])/len(value['cases']),
                         'cells': value['cells'], 'train_nll': nll(model, encoded['train'][:64], c['batch']),
                         'development_nll': nll(model, encoded['development'], c['batch'])}


def fit(out):
    p = verify_plan(out)
    c = p['config']
    if (out/'fit_started.json').exists():
        raise FileExistsError('Preserve first fitting attempt; no automatic overwrite/recovery')
    dump(out/'fit_started.json', {'plan_sha256': sha_file(out/'plan.json'), 'started_utc': datetime.now(timezone.utc).isoformat()})
    tok, arrays, encoded = load_inputs(out, p)
    indices = np.load(out/'order.npy', allow_pickle=False)
    if not np.array_equal(indices, order(len(arrays['train']), c['updates'], c['batch'], c['order_seed'])):
        raise ValueError('Frozen batch schedule differs')
    summaries = []
    for record in p['historical_checkpoints']:
        name = record['arm']
        folder = out/'models'/name
        folder.mkdir(parents=True, exist_ok=False)
        saved = torch.load(ROOT/record['checkpoint'], map_location='cpu', weights_only=True)
        if saved['language_step'] != 2000 or saved['stage'] != 'language' or saved['arm'] != name:
            raise ValueError('Wrong inherited stage endpoint')
        model = model_from(saved)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], betas=(.9,.999), eps=1e-8, weight_decay=c['weight_decay'])
        initial_sha = state_sha(model.state_dict())
        curve, trace, files = [], [], {}
        train_seconds = eval_seconds = 0.
        counts = dict.fromkeys(('input_tokens', 'supervised_targets', 'padded_positions'), 0)
        for step in range(c['updates']+1):
            if step:
                tick = time.perf_counter()
                model.train()
                batch = [encoded['train'][i] for i in indices[step-1]]
                x, y = collate(batch)
                lr = learning_rate(step, c)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                loss = answer_loss(model, x, y)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss; retain failed run')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                counts['input_tokens'] += sum(len(r['input']) for r in batch)
                counts['supervised_targets'] += int((y != -100).sum())
                counts['padded_positions'] += x.numel()
                train_seconds += time.perf_counter()-tick
                trace.append({'step': step, 'lr': lr, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                              'sampled_ids': [r['id'] for r in batch], **counts})
            if step in c['evaluation_updates']:
                checkpoint = folder/f'step_{step}.pt'
                torch.save({'config': saved['config'], 'vocab': saved['vocab'], 'seed': saved['seed'], 'arm': name,
                            'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'adaptation_step': step,
                            'plan_sha256': sha_file(out/'plan.json'), 'inherited_checkpoint_sha256': record['checkpoint_sha256'],
                            'initial_model_state_sha256': initial_sha, **counts}, checkpoint)
                files[checkpoint.relative_to(out).as_posix()] = sha_file(checkpoint)
                tick = time.perf_counter()
                predictions, metrics = evaluate(model, tok, arrays, encoded, c)
                eval_seconds += time.perf_counter()-tick
                predfile = folder/f'predictions_{step}.jsonl'
                write_jsonl(predfile, predictions)
                files[predfile.relative_to(out).as_posix()] = sha_file(predfile)
                curve.append({'step': step, 'selection_split': 'id_development', 'tokens': counts['input_tokens'],
                              'targets': counts['supervised_targets'], 'seconds': train_seconds, 'evaluation_seconds': eval_seconds, **metrics})
                dump(folder/'progress.json', {'arm': name, 'step': step, 'exact_match': metrics['exact_match'], 'valid_format': metrics['valid_format'],
                                             'train_seconds': train_seconds, 'evaluation_seconds': eval_seconds})
                print(name, step, 'ID exact', metrics['exact_match'], 'valid', metrics['valid_format'],
                      'train/dev NLL', metrics['train_nll']['answer_eos_nll'], metrics['development_nll']['answer_eos_nll'], flush=True)
        if counts != p['expected_exposure_per_arm']:
            raise ValueError('Actual exposure differs from frozen order')
        tracefile = folder/'updates.jsonl'
        write_jsonl(tracefile, trace)
        files[tracefile.relative_to(out).as_posix()] = sha_file(tracefile)
        summary = {'arm': name, 'curve': curve, 'decision': decision(curve, c), 'files_sha256': files,
                   'counts': counts, 'updates': c['updates'], 'sampled_order_sha256': digest([r['sampled_ids'] for r in trace]),
                   'schedule_sha256': digest([r['lr'] for r in trace]), 'initial_model_state_sha256': initial_sha,
                   'train_seconds': train_seconds, 'evaluation_seconds': eval_seconds}
        dump(folder/'summary.json', summary)
        summaries.append(summary)
        del model, optimizer, saved
    passed = all(s['decision']['passed'] for s in summaries)
    result = {'plan_sha256': sha_file(out/'plan.json'), 'models': summaries, 'id_calibration_passed': passed,
              'new_scientific_optimizer_updates': len(summaries)*c['updates'], 'original_random_rule_readiness': 'still_failed',
              'graph_acceleration_claim': None, 'new_language_updates': 0,
              'ood_decision': 'eligible_for_separate_fixed_endpoint_development_evaluation' if passed else 'skipped_failed_ID_calibration',
              'new_ood_predictions': 0}
    dump(out/'results.json', result)
    dump(out/'results_receipt.json', {'sha256': sha_file(out/'results.json')})
    verify_plan(out)
    print('COMPLETE: ID calibration', passed, ';', result['ood_decision'], flush=True)


def verify(out):
    p = verify_plan(out)
    result = read(out/'results.json')
    if sha_file(out/'results.json') != read(out/'results_receipt.json')['sha256']:
        raise ValueError('Result identity changed')
    tok, arrays, encoded = load_inputs(out, p)
    dataset_audit(arrays)
    c = p['config']
    expected_order = [[encoded['train'][i]['id'] for i in b] for b in np.load(out/'order.npy', allow_pickle=False)]
    total = checkpoints = 0
    if [m['arm'] for m in result['models']] != [r['arm'] for r in p['historical_checkpoints']]:
        raise ValueError('All five arms must be retained in frozen order')
    for record, summary in zip(p['historical_checkpoints'], result['models']):
        for name, checksum in summary['files_sha256'].items():
            if sha_file(out/name) != checksum:
                raise ValueError('Artifact changed: '+name)
        folder = out/'models'/summary['arm']
        trace = read_jsonl(folder/'updates.jsonl')
        if (len(trace) != c['updates'] or [r['sampled_ids'] for r in trace] != expected_order
                or summary['sampled_order_sha256'] != p['sampled_order_sha256']
                or summary['schedule_sha256'] != p['schedule_sha256']
                or summary['counts'] != p['expected_exposure_per_arm']):
            raise ValueError('Actual common stage/budget differs')
        for i, row in enumerate(trace, 1):
            if row['step'] != i or row['lr'] != learning_rate(i, c):
                raise ValueError('Update/schedule sequence differs')
        inherited = torch.load(ROOT/record['checkpoint'], map_location='cpu', weights_only=True)
        if state_sha(inherited['model']) != summary['initial_model_state_sha256']:
            raise ValueError('Inherited full model changed at adaptation origin')
        del inherited
        if [r['step'] for r in summary['curve']] != c['evaluation_updates']:
            raise ValueError('Fixed evaluation checkpoint omitted')
        for row in summary['curve']:
            step = row['step']
            saved = torch.load(folder/f'step_{step}.pt', map_location='cpu', weights_only=True)
            if saved['adaptation_step'] != step or saved['plan_sha256'] != sha_file(out/'plan.json') or saved['inherited_checkpoint_sha256'] != record['checkpoint_sha256']:
                raise ValueError('Checkpoint receipt/step differs')
            if step:
                steps = {int(s['step']) for s in saved['optimizer']['state'].values()}
                if steps != {step} or len(saved['optimizer']['state']) != len(saved['model']):
                    raise ValueError('Every model parameter must receive all adaptation updates')
                expected = trace[step-1]
                if any(saved[k] != expected[k] for k in p['expected_exposure_per_arm']):
                    raise ValueError('Checkpoint exposure differs')
            elif state_sha(saved['model']) != summary['initial_model_state_sha256'] or saved['optimizer']['state']:
                raise ValueError('Step zero must be unchanged full model and fresh optimizer')
            model = model_from(saved)
            preds, metrics = evaluate(model, tok, arrays, encoded, c)
            if preds != read_jsonl(folder/f'predictions_{step}.jsonl') or any(metrics[k] != row[k] for k in metrics):
                raise ValueError('Separate checkpoint generation/NLL replay differs')
            total += len(preds)
            checkpoints += 1
            del model, saved
        if decision(summary['curve'], c) != summary['decision']:
            raise ValueError('Fixed decision differs')
    passed = all(m['decision']['passed'] for m in result['models'])
    if result['id_calibration_passed'] != passed or result['new_scientific_optimizer_updates'] != 3000 or result['new_ood_predictions'] != 0:
        raise ValueError('Calibration/update/OOD accounting differs')
    dump(out/'verification.json', {'passed': True, 'checkpoint_replays': checkpoints, 'generated_answers_replayed': total,
                                  'results_sha256': sha_file(out/'results.json'), 'plan_sha256': sha_file(out/'plan.json'),
                                  'scope': 'Separate same-code checkpoint generation/NLL replay and independent graph labels; no retraining'})
    print('VERIFIED', checkpoints, 'checkpoints;', total, 'exact generated-answer replays', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'fit', 'verify'))
    parser.add_argument('--out', type=Path, default=ROOT/'runs/adaptation_v13_run1')
    a = parser.parse_args()
    c = read(CONFIG)
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    globals()[a.command](a.out)


if __name__ == '__main__':
    main()
