"""Two matched binary involutions with split exclusion under both rule orbits."""
import numpy as np
import torch
from transfer_v4.nca import encode_frames
from transfer_v4.controls import prefix_shuffled, validate_prefix_control

RULES = ('complement', 'spatial')


def step(cells, rule):
    if rule == 'complement':
        return 1 ^ cells
    if rule == 'spatial':
        return 1 ^ cells ^ np.roll(cells, 1, axis=-1) ^ np.roll(cells, -1, axis=-1)
    raise ValueError('Unknown frozen rule')


def unpack(ids):
    return ((np.asarray(ids)[:, None] >> np.arange(16)) & 1).astype(np.int64).reshape(-1, 4, 4)


def pack(cells):
    return (cells.reshape(-1, 16).astype(np.int64)*(1 << np.arange(16))).sum(1)


def canonical(ids):
    ids = np.asarray(ids)
    transformed = pack(step(unpack(ids), 'spatial'))
    # Complement commutes with the spatial involution; these four states
    # therefore include both rules' complete phase-equivalence classes.
    return np.minimum.reduce([ids, ids ^ 65535, transformed, transformed ^ 65535])


def generate(c):
    if c['nca'] != {'grid': 4, 'states': 2} or c['frames'] != 8 or c['model']['context'] != 47:
        raise ValueError('Keep the binary lattice, frame count and context fixed')
    n = sum(c['splits'].values())
    if list(c['splits']) != ['train', 'development', 'assessment'] or min(c['splits'].values()) < 1 or n > 16512:
        raise ValueError('Invalid jointly phase-disjoint split sizes')
    candidates = np.random.default_rng(c['data_seed']).permutation(65536)
    keys = canonical(candidates)
    selected, selected_keys, seen = [], [], set()
    for identity, key in zip(candidates, keys):
        if int(key) not in seen:
            selected.append(int(identity))
            selected_keys.append(int(key))
            seen.add(int(key))
            if len(selected) == n:
                break
    arrays, identities, classes = {}, {}, {}
    initial = unpack(selected)
    for rule in RULES:
        frames = [initial]
        for _ in range(c['frames']-1):
            frames.append(step(frames[-1], rule))
        cells = np.stack(frames, axis=1).reshape(n, c['frames'], 16)
        tokens = encode_frames(torch.tensor(cells), 4, 2).numpy().astype(np.uint16)
        start = 0
        for i, (split, count) in enumerate(c['splits'].items()):
            name = f'{rule}_{split}'
            arrays[name] = tokens[start:start+count].copy()
            arrays[f'{rule}_prefix_{split}'] = prefix_shuffled(arrays[name], c, c['data_seed']+1+i)
            validate_prefix_control(arrays[name], arrays[f'{rule}_prefix_{split}'], c)
            identities[split] = selected[start:start+count]
            classes[split] = selected_keys[start:start+count]
            start += count
    return arrays, {'initial_ids': identities, 'joint_orbit_ids': classes}


def decode_patches(patches):
    bits = ((patches[..., None].astype(np.int64) >> np.arange(4)) & 1)
    out = np.empty((len(patches), 4, 4), dtype=np.int64)
    for p in range(4):
        r, col = (p//2)*2, (p%2)*2
        out[:, r, col], out[:, r, col+1] = bits[:, p, 0], bits[:, p, 1]
        out[:, r+1, col], out[:, r+1, col+1] = bits[:, p, 2], bits[:, p, 3]
    return out


def oracle_nll(tokens, rule, probability):
    # Predict each frame only from the actually observed preceding FULL frame.
    frames = tokens.reshape(len(tokens), 8, 6)
    prev = decode_patches(frames[:, :-1, 1:5].reshape(-1, 4))
    next_cells = step(prev, rule).reshape(-1, 1, 16)
    predicted = encode_frames(torch.tensor(next_cells), 4, 2).numpy().reshape(len(tokens), 7, 6)[:, :, 1:5]
    correct = predicted == frames[:, 1:, 1:5]
    losses = -np.log(np.where(correct, probability, (1-probability)/15))
    return losses.reshape(len(tokens), -1).mean(1).tolist()
