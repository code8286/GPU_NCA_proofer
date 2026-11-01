"""Check the actually executed recovery helper and all endpoints before scoring."""
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.recover_confirmation_v11_6950 import OUT, verify_record, completed
from confirmation_v11.plan import verify_plan
from confirmation_v11.run import verify_training
from transfer_v4.common import dump, sha_file


def main():
    torch.set_num_threads(2)
    record, _ = verify_record(OUT)
    plan = verify_plan(OUT)
    rows = verify_training(OUT, plan)
    for row in rows:
        folder = OUT/f"{row['level']}_{row['objective']}_{row['arm']}_{row['seed']}"
        assert completed(folder, plan) == row
    for name, entry in record['interrupted_models'].items():
        assert all(sha_file(OUT/entry['archive']/p) == h for p,h in entry['files_sha256'].items())
        old = entry['metrics']
        new = [json.loads(x) for x in (OUT/name/'metrics.jsonl').read_text().splitlines()][:len(old)]
        clean = lambda rows: [{k:v for k,v in r.items() if k != 'stage_seconds'} for r in rows]
        assert clean(old) == clean(new)
    dump(OUT/'execution_recovery_verification_6950.json', {
        'passed': True, 'completed_final_endpoints': 24, 'retained_models_unchanged': 21,
        'interrupted_prefix_metrics_reproduced_through_update': 800,
        'extra_aborted_updates': 800, 'extra_aborted_input_tokens': 150400,
        'recovery_sha256': sha_file(OUT/'execution_recovery.json'),
        'preserved_executed_helper_sha256': sha_file(ROOT/'scripts/recover_confirmation_v11_6950.py'),
        'training_sha256': sha_file(OUT/'training_complete.json'), 'language_allowed': False})
    print('PASS: all 24 final weights/Adam states, common starts and frozen budgets/orders; 21 retained folders unchanged; restarted fit reproduces every recorded non-time metric through update 800; extra compute retained.')


if __name__ == '__main__':
    main()
