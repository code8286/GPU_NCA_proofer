"""Reconstruct nested inputs and phase exclusions without scoring checkpoints."""
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diversity_v10.plan import read, verify_plan
from diversity_v10.data import generate
from interaction_v7.data import canonical


def main():
    out = ROOT/'runs/diversity_v10_run1'
    plan = verify_plan(out)
    ids = read(out/'data/original_identities.json')
    original = np.load(out/'data/original_train.npy', allow_pickle=False)
    prefix = np.load(out/'data/original_prefix_train.npy', allow_pickle=False)
    coherent, shuffled, expanded = generate(ids, original, prefix, plan['training_config'], plan['settings'])
    np.testing.assert_array_equal(coherent, np.load(out/'data/train.npy', allow_pickle=False))
    np.testing.assert_array_equal(shuffled, np.load(out/'data/prefix_train.npy', allow_pickle=False))
    assert expanded == read(out/'data/expanded_identities.json')
    classes = set(expanded['joint_orbit_ids'])
    assert len(classes) == 4096 and len(expanded['initial_ids']) == 4096
    assert not classes & set(ids['joint_orbit_ids']['development']+ids['joint_orbit_ids']['assessment'])
    mask = np.array(read(out/'queries/eligibility.json')['eligible'], dtype=bool)
    changed_classes = set(canonical(np.array(ids['initial_ids']['assessment'])[mask] ^ 8).tolist())
    assert int(mask.sum()) == 122 and not classes & changed_classes
    assert plan['original_training_config']['splits']['train'] == 512
    for row in plan['standard_controls']:
        assert row['source_summary']['input_tokens'] == 188000 and row['source_summary']['prediction_targets'] == 168000
        assert row['source_summary']['train_order_sha256'] == plan['expected_orders'][f"512_{row['seed']}"]['sha256']
    print('PASS: full nested data/shuffle reconstruction; 4096 unique training phases; unchanged 512 rows; development/assessment and all 122 eligible interventions excluded; imported budgets and reconstructed row orders agree.')


if __name__ == '__main__':
    main()
