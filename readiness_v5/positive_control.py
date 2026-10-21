"""Learned transition-table measurement control; no transformer training."""
import numpy as np
import torch
from transfer_v4.common import frame_length
from transfer_v4.nca import encode_frames
from transfer_v4.controls import prefix_shuffled, validate_prefix_control
from .frequency import counts, fit, per_rule_nll
from .gate import decision


def previous_and_truth(tokens, c):
    first, end = frame_length(c), c['model']['context']+1
    return tokens[:, np.arange(first, end)-first], tokens[:, first:end]


def table_scores(table, tokens, c):
    before, truth = previous_and_truth(tokens, c)
    v = c['nca']['states']**4
    return [float(-np.log(table[a[b < v], b[b < v]]).mean()) for a, b in zip(before, truth)]


def positive_control(settings):
    s = settings['positive_control']
    side, states, frames = s['grid'], s['states'], s['frames']
    if side != 4 or states != 2:
        raise ValueError('This measurement control is defined for a binary 4x4 lattice')
    c = {'nca': {'grid': side, 'states': states}, 'model': {'context': frames*((side//2)**2+2)-1}}
    rng = np.random.default_rng(s['seed'])
    n, train_n = s['train_trajectories']+s['validation_trajectories'], s['train_trajectories']
    # One member of each complement pair: no phase-equivalent trajectory crosses splits.
    keys = rng.choice(32768, n, replace=False)
    ids = keys ^ (rng.integers(0, 2, n)*65535)
    initial = ((ids[:, None] >> np.arange(16)) & 1).astype(np.int64)
    cells = (initial[:, None]+np.arange(frames)[None, :, None]) % 2
    tokens = encode_frames(torch.tensor(cells), side, states).numpy().astype(np.uint16)
    train, val = tokens[:train_n], tokens[train_n:]
    shuffled = prefix_shuffled(val, c, s['seed']+1)
    validate_prefix_control(val, shuffled, c)
    table = np.ones((16, 16), dtype=np.float64)
    a, b = previous_and_truth(train, c)
    mask = b < 16
    np.add.at(table, (a[mask], b[mask]), 1)
    table /= table.sum(1, keepdims=True)
    if not np.array_equal(table.argmax(1), np.arange(16) ^ 15):
        raise ValueError('Train-only transition table did not recover the known rule')
    prior, alpha, trials = fit(train, c, settings)
    ai, at = counts(val, c)
    bi, bt = counts(shuffled, c)
    frequency = float(np.mean(per_rule_nll(ai, at, prior, alpha)))
    shuffled_frequency = float(np.mean(per_rule_nll(bi, bt, prior, alpha)))
    native, changed = float(np.mean(table_scores(table, val, c))), float(np.mean(table_scores(table, shuffled, c)))
    learned = decision(native, changed, frequency, settings)
    negative = decision(frequency, shuffled_frequency, frequency, settings)
    return {'scope': 'Known self-local cell rule; held-out initial conditions, not unseen rules or transformer competence',
            'train_trajectories': train_n, 'held_out_trajectories': len(val),
            'complement_equivalence_classes_disjoint': not bool(set(keys[:train_n]) & set(keys[train_n:])),
            'transition_table_recovers_rule': True, 'concentration': alpha, 'concentration_fit': trials,
            'learned_coherent_nll': native, 'learned_shuffled_nll': changed,
            'frequency_nll': frequency, 'frequency_shuffled_nll': shuffled_frequency,
            'learned_gate': learned, 'frequency_negative_gate': negative,
            'passed': learned['passed'] and not negative['passed'] and abs(frequency-shuffled_frequency) < 1e-12}
