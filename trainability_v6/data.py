"""Complement-rule trajectories with disjoint initial-condition phase classes."""
import numpy as np
import torch
from transfer_v4.nca import encode_frames
from transfer_v4.controls import prefix_shuffled, validate_prefix_control


def generate(c):
    if c['rule'] != 'binary_cell_complement' or c['nca'] != {'grid': 4, 'states': 2}:
        raise ValueError('Declare a separate task for another rule or lattice')
    if c['model']['context'] != c['frames']*6-1 or c['frames'] < 3:
        raise ValueError('Use complete framed trajectories')
    sizes = c['splits']
    if list(sizes) != ['train', 'development', 'assessment'] or min(sizes.values()) < 1 or sum(sizes.values()) > 32768:
        raise ValueError('Invalid phase-disjoint split sizes')
    rng = np.random.default_rng(c['data_seed'])
    # Canonical representative min(id, id^65535) is exactly a number <32768.
    keys = rng.choice(32768, sum(sizes.values()), replace=False)
    ids = keys ^ (rng.integers(0, 2, len(keys))*65535)
    initial = ((ids[:, None] >> np.arange(16)) & 1).astype(np.int64)
    cells = (initial[:, None]+np.arange(c['frames'])[None, :, None]) % 2
    tokens = encode_frames(torch.tensor(cells), 4, 2).numpy().astype(np.uint16)
    arrays, identities, start = {}, {}, 0
    for i, (name, count) in enumerate(sizes.items()):
        arrays[name] = tokens[start:start+count].copy()
        arrays['prefix_'+name] = prefix_shuffled(arrays[name], c, c['data_seed']+1+i)
        validate_prefix_control(arrays[name], arrays['prefix_'+name], c)
        identities[name] = keys[start:start+count].tolist()
        start += count
    return arrays, identities
