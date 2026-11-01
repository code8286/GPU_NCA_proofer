"""Deterministic fresh joint-phase splits, including reserved intervention pairs."""
import numpy as np
from interaction_v7.data import canonical
from diversity_v10.data import trajectories, draw_order, exposure_mask
from transfer_v4.controls import prefix_shuffled, validate_prefix_control


def select_ids(forbidden, seed, sizes):
    seen = set(int(k) for k in forbidden)
    candidates = np.random.default_rng(seed).permutation(65536)
    classes = canonical(candidates)
    changed_classes = canonical(candidates ^ 8)
    selected = {}
    # Reserve both members of every assessment intervention pair first. This is
    # outcome-free eligibility by construction, not post-scoring case removal.
    for split in ('assessment', 'development', 'train'):
        rows = []
        for identity, key, changed_key in zip(candidates, classes, changed_classes):
            keys = {int(key), int(changed_key)} if split == 'assessment' else {int(key)}
            if seen.isdisjoint(keys):
                rows.append(int(identity))
                seen.update(keys)
                if len(rows) == sizes[split]:
                    break
        if len(rows) != sizes[split]:
            raise ValueError('Not enough fresh joint phases for '+split)
        selected[split] = rows
    return {'initial_ids': selected,
            'joint_orbit_ids': {k: canonical(np.array(v)).tolist() for k, v in selected.items()},
            'changed_assessment_orbits': canonical(np.array(selected['assessment']) ^ 8).tolist(),
            'historical_forbidden_orbits': sorted(set(int(k) for k in forbidden))}


def generate(forbidden, c):
    ids = select_ids(forbidden, c['data_seed'], c['splits'])
    arrays = {}
    for i, split in enumerate(('train', 'development', 'assessment')):
        tokens = trajectories(ids['initial_ids'][split])
        shuffled = prefix_shuffled(tokens, c, c['data_seed']+1+i)
        validate_prefix_control(tokens, shuffled, c)
        arrays[split], arrays['prefix_'+split] = tokens, shuffled
    native = arrays['assessment'][:, :8].copy()
    changed = trajectories(np.array(ids['initial_ids']['assessment']) ^ 8)[:, :8].copy()
    np.testing.assert_array_equal(native[:, 1], changed[:, 1])
    np.testing.assert_array_equal(native[:, 7] ^ changed[:, 7], np.ones(len(native), dtype=np.uint16))
    return arrays, ids, native, changed
