"""Journaled recovery of a Windows status-file interruption, without changing the frozen kernel."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
from pathlib import Path
import shutil
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import dump, sha_file
from confirmation_v11 import run
from confirmation_v11.plan import read, verify_plan

OUT = ROOT/'runs/confirmation_v11_run1'
GUARD = '                    if completed_or_archive(folder, rows, out, plan):\n                        continue\n'


def derived_source():
    original = inspect.getsource(run.train)
    needle = '                    folder.mkdir(exist_ok=False)'
    if original.count(needle) != 1:
        raise ValueError('Expected exactly one unchanged model-folder entry')
    derived = original.replace(needle, GUARD+needle)
    if derived.replace(GUARD, '') != original:
        raise ValueError('Recovery must only add the completed-model guard')
    return original, derived


def retry_dump(path, value):
    # Windows readers can temporarily prevent atomic replacement. Retries only
    # repeat the same serialization; they never repeat an optimizer update.
    for attempt in range(100):
        try:
            return dump(path, value)
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def completed(folder, plan):
    row = read(folder/'summary.json')
    if sha_file(folder/'final.pt') != row['checkpoint_sha256']:
        raise ValueError('Retained final checkpoint changed')
    saved = torch.load(folder/'final.pt', map_location='cpu', weights_only=True)
    initial = torch.load(folder.parent/f"initialization_{row['seed']}.pt", map_location='cpu', weights_only=True)
    model = run.LM(plan['training_config']['model'], 18, row['seed'], row['seed']+50000)
    model.load_state_dict(saved['model'], strict=True)
    assert saved['config'] == plan['training_config'] and saved['settings'] == plan['settings']
    assert saved['plan_sha256'] == sha_file(folder.parent/'plan.json')
    for k in ('seed', 'level', 'objective', 'arm', 'factor', 'steps'):
        assert saved[k] == row[k]
    assert row['steps'] == 1000 and row['input_tokens'] == 188000 and row['prediction_targets'] == 168000
    assert row['initial_core_sha256'] == saved['core_initial_sha256'] == initial['core_sha256']
    assert row['initial_head_sha256'] == saved['head_initial_sha256'] == initial['head_sha256']
    assert model.core_hash() == row['final_core_sha256'] and model.head_hash() == row['final_head_sha256']
    assert saved['optimizer']['state'] and all(int(v['step']) == 1000 for v in saved['optimizer']['state'].values())
    assert row['actual_order'] == plan['expected_orders'][f"{row['level']}_{row['seed']}"]
    return row | {'summary_sha256': sha_file(folder/'summary.json')}


def freeze(out):
    plan = verify_plan(out)
    if (out/'execution_recovery.json').exists() or (out/'training_complete.json').exists():
        raise FileExistsError('Preserve existing recovery/completed training')
    original, derived = derived_source()
    events = [read_line for line in (ROOT/'logs/checks.jsonl').read_text().splitlines() if (read_line := __import__('json').loads(line))['label'] == 'confirmation-v11-twenty-four-fresh-fixed-budget-models']
    if len(events) != 1 or events[0]['exit_code'] != 1 or 'status.json.tmp' not in events[0]['stderr'] or 'PermissionError' not in events[0]['stderr']:
        raise ValueError('Expected retained status replacement failure only')
    kept, partial = {}, {}
    for folder in sorted(out.glob('*_*_*_*')):
        if not folder.is_dir():
            continue
        if (folder/'final.pt').exists() and (folder/'summary.json').exists():
            completed(folder, plan)
            kept[folder.name] = {p.name: sha_file(p) for p in folder.iterdir() if p.is_file()}
        else:
            metrics = [__import__('json').loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
            partial[folder.name] = {'files_sha256': {p.name: sha_file(p) for p in folder.iterdir() if p.is_file()},
                                    'metrics': metrics, 'completed_updates': metrics[-1]['step'],
                                    'archive': 'execution_interruptions/'+folder.name}
    assert len(kept) == 21 and set(partial) == {'4096_standard_prefix_211'}
    assert partial['4096_standard_prefix_211']['completed_updates'] == 800
    # Preserve the failed atomic-write temporary before another status write.
    if (out/'status.json.tmp').exists():
        (out/'execution_interruptions').mkdir(exist_ok=True)
        shutil.copyfile(out/'status.json.tmp', out/'execution_interruptions/status_at_failure.json.tmp')
    record = {'created_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha_file(out/'plan.json'),
              'driver_sha256': sha_file(Path(__file__)), 'original_train_sha256': hashlib.sha256(original.encode()).hexdigest(),
              'derived_train_sha256': hashlib.sha256(derived.encode()).hexdigest(), 'retained_models': kept,
              'interrupted_models': partial, 'failed_event': events[0], 'extra_aborted_updates': 800,
              'extra_aborted_input_tokens': 150400, 'extra_aborted_prediction_targets': 134400,
              'remaining_endpoints': 3, 'numerical_changes': False, 'language_allowed': False}
    dump(out/'execution_recovery.json', record)
    dump(out/'execution_recovery_receipt.json', {'sha256': sha_file(out/'execution_recovery.json')})
    print('Frozen execution-only recovery:', sha_file(out/'execution_recovery.json'), '21 endpoints preserved, 3 pending; 800 aborted updates charged separately.')


def verify_record(out):
    verify_plan(out)
    record = read(out/'execution_recovery.json')
    original, derived = derived_source()
    assert sha_file(out/'execution_recovery.json') == read(out/'execution_recovery_receipt.json')['sha256']
    assert sha_file(Path(__file__)) == record['driver_sha256']
    assert hashlib.sha256(original.encode()).hexdigest() == record['original_train_sha256']
    assert hashlib.sha256(derived.encode()).hexdigest() == record['derived_train_sha256']
    assert sha_file(out/'plan.json') == record['plan_sha256']
    for name, files in record['retained_models'].items():
        assert all(sha_file(out/name/p) == h for p,h in files.items())
    return record, derived


def completed_or_archive(folder, rows, out, plan):
    record = read(out/'execution_recovery.json')
    if folder.name in record['retained_models']:
        rows.append(completed(folder, plan))
        return True
    if folder.name in record['interrupted_models']:
        entry = record['interrupted_models'][folder.name]
        destination = out/entry['archive']
        # Verify both resolved move targets are confined to this exact run.
        if not folder.resolve().is_relative_to(out.resolve()) or not destination.resolve().is_relative_to((out/'execution_interruptions').resolve()):
            raise ValueError('Unsafe interrupted-folder move')
        assert folder.exists() and not destination.exists()
        assert all(sha_file(folder/p) == h for p,h in entry['files_sha256'].items())
        destination.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(destination)
    return False


def train(out):
    _, derived = verify_record(out)
    namespace = run.__dict__.copy() | {'completed_or_archive': completed_or_archive, 'dump': retry_dump}
    exec(compile(derived, '<frozen-train-with-execution-recovery-guard>', 'exec'), namespace)
    namespace['train'](out)


def verify(out):
    record, _ = verify_record(out)
    plan = verify_plan(out)
    rows = run.verify_training(out, plan)
    for entry in record['interrupted_models'].values():
        assert all(sha_file(out/entry['archive']/p) == h for p,h in entry['files_sha256'].items())
    assert len(rows) == 24
    dump(out/'execution_recovery_verification.json', {'passed': True, 'retained_models_unchanged': 21,
          'completed_final_endpoints': 24, 'extra_aborted_updates': 800, 'recovery_sha256': sha_file(out/'execution_recovery.json'),
          'training_sha256': sha_file(out/'training_complete.json'), 'language_allowed': False})
    print('PASS: 21 retained endpoints unchanged; interrupted evidence archived; all 24 frozen endpoints complete; only execution guard/status retries added.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'train', 'verify'))
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    globals()[a.command](OUT)


if __name__ == '__main__':
    main()
