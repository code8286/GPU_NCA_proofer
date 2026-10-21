"""Normalize a sixfold first-transition weight without changing masks or data."""
import numpy as np
import torch
from torch.nn import functional as F
from transfer_v4.common import frame_length, lr_at
from transfer_v4.training import batch


def weights(labels, c, factor):
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Positive finite first-transition weight required')
    position = torch.arange(1, labels.shape[1]+1, device=labels.device)
    first = frame_length(c)
    coefficient = torch.where((position >= first) & (position < 2*first), factor, 1.)
    return coefficient[None, :].expand_as(labels)*(labels != -100)


def loss(logits, labels, c, factor):
    w = weights(labels, c, factor)
    if not torch.any(w > 0):
        raise ValueError('No supervised targets')
    if factor == 1:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    values = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none', ignore_index=-100).reshape_as(labels)
    return (values*w).sum()/w.sum()


def update(model, optimizer, array, c, stage, local_step, rng, order_hash, factor):
    # Same v4 batch, schedule, Adam, accumulation and finite-norm policy;
    # the normalized objective above is the only numerical treatment change.
    lr = lr_at(local_step, stage)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    loss_sum, prediction_targets = 0., 0
    for _ in range(c['accumulate']):
        x, y, indices = batch(array, c, rng, 'cpu', True)
        order_hash.update(indices.astype('<i8').tobytes())
        value = loss(model(x), y, c, factor)
        if not torch.isfinite(value):
            raise FloatingPointError('Non-finite weighted objective')
        (value/c['accumulate']).backward()
        loss_sum += float(value.detach())/c['accumulate']
        prediction_targets += int((y != -100).sum())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf') if stage['clip'] is None else stage['clip'], error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise FloatingPointError('Non-finite gradient norm')
    optimizer.step()
    return loss_sum, prediction_targets, lr
