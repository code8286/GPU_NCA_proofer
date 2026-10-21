"""Causal, exchangeable patch-frequency likelihood with train-only calibration."""
import numpy as np
import torch
from transfer_v4.common import frame_length


def counts(tokens, c):
    v, first, end = c['nca']['states']**4, frame_length(c), c['model']['context']+1
    if tokens.ndim != 2 or tokens.shape[1] < end:
        raise ValueError('Incomplete trained prefix')
    initial = np.zeros((len(tokens), v), dtype=np.float64)
    target = np.zeros_like(initial)
    for i, row in enumerate(tokens):
        a, b = row[:first], row[first:end]
        initial[i] = np.bincount(a[a < v], minlength=v)
        target[i] = np.bincount(b[b < v], minlength=v)
    return initial, target


def per_rule_nll(initial, target, prior, alpha):
    if alpha <= 0 or np.any(np.asarray(prior) <= 0) or not np.isclose(np.sum(prior), 1):
        raise ValueError('Positive normalized prior and concentration required')
    a, b = torch.as_tensor(initial), torch.as_tensor(target)
    q = torch.as_tensor(prior)*alpha
    total = a.sum(1)+alpha
    logp = torch.lgamma(total)-torch.lgamma(total+b.sum(1))+(torch.lgamma(q+a+b)-torch.lgamma(q+a)).sum(1)
    if torch.any(b.sum(1) == 0):
        raise ValueError('No scored patch targets')
    return (-logp/b.sum(1)).tolist()


def fit(train, c, settings):
    v = c['nca']['states']**4
    y = train[:, frame_length(c):c['model']['context']+1]
    n = np.bincount(y[y < v], minlength=v).astype(float)+1
    prior = n/n.sum()
    initial, target = counts(train[:settings['frequency_fit_rules']], c)
    trials = [{'concentration': alpha, 'train_patch_nll': float(np.average(per_rule_nll(initial, target, prior, alpha), weights=target.sum(1)))}
              for alpha in settings['frequency_concentrations']]
    return prior, min(trials, key=lambda t:t['train_patch_nll'])['concentration'], trials
