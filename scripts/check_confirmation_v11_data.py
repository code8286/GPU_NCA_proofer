"""Reconstruct fresh-phase data and controls before any scientific fitting."""
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from confirmation_v11.plan import read, verify_plan, reference
from confirmation_v11.data import generate
from transfer_v4.controls import prefix_shuffled


def main():
    out = ROOT/'runs/confirmation_v11_run1'
    plan = verify_plan(out)
    _, _, forbidden = reference()
    arrays, ids, native, changed = generate(forbidden, plan['training_config'])
    assert ids == read(out/'data/split_identities.json')
    for name, values in arrays.items():
        np.testing.assert_array_equal(values, np.load(out/f'data/{name}.npy', allow_pickle=False))
    for name, values in (('native', native), ('changed', changed)):
        np.testing.assert_array_equal(values, np.load(out/f'queries/spatial_{name}.npy', allow_pickle=False))
    groups = [set(ids['joint_orbit_ids'][k]) for k in ('train', 'development', 'assessment')]
    interventions = set(ids['changed_assessment_orbits'])
    assert [len(x) for x in groups] == [4096, 64, 128]
    assert all(not x & set(forbidden) for x in groups+[interventions])
    assert groups[0].isdisjoint(groups[1] | groups[2] | interventions)
    assert groups[1].isdisjoint(groups[2] | interventions)
    # Each pair is reserved before selecting the next original, so no pair can
    # accidentally reuse a previously reserved original or changed phase.
    assert len(groups[2] | interventions) == 256
    assert read(out/'queries/eligibility.json')['eligible'] == [True]*128
    np.testing.assert_array_equal(arrays['prefix_train'][:512], prefix_shuffled(arrays['train'][:512], plan['training_config'], plan['training_config']['data_seed']+1))
    old, c = plan['parent_training_config'], plan['training_config']
    assert {k:v for k,v in old.items() if k not in ('seeds', 'data_seed')} == {k:v for k,v in c.items() if k not in ('seeds', 'data_seed')}
    print(f'PASS: exact input/shuffle/query reconstruction; {len(forbidden)} historical joint classes excluded; fresh 4096/64/128 split, all 128 reserved intervention pairs; nested cache identity and unchanged training settings.')


if __name__ == '__main__':
    main()
