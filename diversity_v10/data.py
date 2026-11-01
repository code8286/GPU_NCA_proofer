"""Append spatial trajectories without changing any inspected held-out phase."""
import hashlib
import numpy as np
import torch
from interaction_v7.data import canonical, unpack, step
from transfer_v4.nca import encode_frames
from transfer_v4.controls import prefix_shuffled, validate_prefix_control


def expanded_ids(identities, total, seed):
    original = identities['initial_ids']['train']
    if total <= len(original):
        raise ValueError('Expansion must increase training availability')
    forbidden = set(sum(identities['joint_orbit_ids'].values(), []))
    forbidden.update(int(k) for k in canonical(np.array(identities['initial_ids']['assessment']) ^ 8))
    selected, seen = list(original), set(forbidden)
    candidates = np.random.default_rng(seed).permutation(65536)
    keys = canonical(candidates)
    for identity, key in zip(candidates, keys):
        if int(key) not in seen:
            selected.append(int(identity))
            seen.add(int(key))
            if len(selected) == total:
                return selected
    raise ValueError('Not enough eligible joint phases')


def trajectories(ids):
    frames = [unpack(ids)]
    for _ in range(7):
        frames.append(step(frames[-1], 'spatial'))
    cells = np.stack(frames, axis=1).reshape(len(ids), 8, 16)
    return encode_frames(torch.tensor(cells), 4, 2).numpy().astype(np.uint16)


def generate(identities, original, original_prefix, c, settings):
    ids = expanded_ids(identities, settings['training_conditions'], settings['expansion_seed'])
    tokens = trajectories(ids)
    shuffled = prefix_shuffled(tokens, c, c['data_seed']+1)
    np.testing.assert_array_equal(tokens[:len(original)], original)
    np.testing.assert_array_equal(shuffled[:len(original)], original_prefix)
    validate_prefix_control(tokens, shuffled, c)
    return tokens, shuffled, {'initial_ids': ids, 'joint_orbit_ids': canonical(np.array(ids)).tolist()}


def draw_order(seed, c, rows):
    rng, digest = np.random.default_rng(seed+101), hashlib.sha256()
    indices = []
    for _ in range(c['ppt']['steps']*c['accumulate']):
        values = rng.integers(0, rows, size=c['batch']).astype('<i8')
        digest.update(values.tobytes())
        indices.extend(values.tolist())
    return {'sha256': digest.hexdigest(), 'draws': len(indices), 'unique_rows': len(set(indices)),
            'observed_indices': sorted(set(indices))}


def exposure_mask(probe, level, objective, seed, orders, count):
    if objective is None or probe in ('development', 'assessment'):
        return np.zeros(count, dtype=bool)
    offset = 0 if probe == 'anchor' else 512
    seen = set(orders[f'{level}_{seed}']['observed_indices'])
    return np.array([offset+i in seen for i in range(count)], dtype=bool)
