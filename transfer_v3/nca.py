"""Categorical spatial NCA port, frozen random rules and rule-disjoint datasets.

Reference: danihyunlee/nca-pre-pretraining at bdd1c71e04eb2e5a2c91481faf278e95846b4431.
This implements spatial-only padding and uniform initial states as described in
the paper, rather than inheriting the reference's channel-padding side effect.
"""
from __future__ import annotations

import gzip
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import frame_length, sync


def lattice(side):
    ids = np.arange(side*side).reshape(side, side)
    return np.stack([np.roll(ids, (-dy, -dx), (0, 1)).ravel()
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 1)


def sample_rules(count, states, generator, device):
    def lecun(shape, fan_in):
        # Flax-style fan-in variance with a normal truncated to [-2,2].
        u = torch.rand(shape, generator=generator, device=device)*.9544997361+.02275013195
        return (2*u-1).erfinv()*math.sqrt(2/fan_in)/.8796256610
    return {"perception": lecun((count, 9*states, 4), 9*states),
            "hidden": lecun((count, 4, 16), 4),
            "output": lecun((count, 16, states), 16)}


def message_logits(state, edges, rules, states):
    local = F.one_hot(state[:, edges], states).float().flatten(-2)
    perception = torch.einsum("bni,bic->bnc", local, rules["perception"])
    hidden = F.relu(torch.einsum("bnc,bch->bnh", perception, rules["hidden"]))
    return torch.einsum("bnh,bhs->bns", hidden, rules["output"])


def convolution_logits(state, rules, side, states):
    # Independent spatial-convolution formulation for the equivalence test.
    b = state.shape[0]
    x = F.one_hot(state, states).float().reshape(b, side, side, states).permute(0, 3, 1, 2)
    weight = rules["perception"].reshape(b, 9, states, 4).permute(0, 3, 2, 1).reshape(b*4, states, 3, 3)
    perceived = F.conv2d(F.pad(x.reshape(1, b*states, side, side), (1, 1, 1, 1), mode="circular"), weight, groups=b)
    p = perceived.reshape(b, 4, side*side).transpose(1, 2)
    hidden = F.relu(torch.einsum("bnc,bch->bnh", p, rules["hidden"]))
    return torch.einsum("bnh,bhs->bns", hidden, rules["output"])


def step(state, edges, rules, states, temperature, generator):
    logits = message_logits(state, edges, rules, states)
    if temperature > 0:
        u = torch.rand(logits.shape, device=logits.device, generator=generator).clamp(1e-7, 1-1e-7)
        logits = logits-temperature*torch.log(-torch.log(u))
    return logits.argmax(-1)


def rollout(rules, c, frames, generator, device, edges=None):
    n = c["nca"]
    edges = torch.as_tensor(lattice(n["grid"]) if edges is None else edges, dtype=torch.long, device=device)
    batch = rules["perception"].shape[0]
    state = torch.randint(n["states"], (batch, n["grid"]**2), generator=generator, device=device)
    for _ in range(n["burnin"]):
        state = step(state, edges, rules, n["states"], n["temperature"], generator)
    values = []
    for _ in range(frames):
        values.append(state)
        state = step(state, edges, rules, n["states"], n["temperature"], generator)
    return torch.stack(values, 1)


def encode_frames(frames, side, states):
    x = frames.reshape(*frames.shape[:-1], side, side)
    patches = (x[..., 0::2, 0::2]+states*x[..., 0::2, 1::2]
               +states**2*x[..., 1::2, 0::2]+states**3*x[..., 1::2, 1::2]).flatten(-2)
    start = torch.full((*patches.shape[:-1], 1), states**4, dtype=torch.long, device=frames.device)
    return torch.cat((start, patches, start+1), -1).flatten(1)


def complexity(tokens, c):
    patches = tokens.reshape(tokens.shape[0], -1, frame_length(c))[:, :, 1:-1]
    scores = []
    for row in patches:
        raw = np.asarray(row, dtype="<i4").tobytes(order="C")
        scores.append(len(gzip.compress(raw, compresslevel=9, mtime=0))/len(raw))
    return np.array(scores)



def shuffled(tokens, c, seed):
    result = tokens.copy()
    rng = np.random.default_rng(seed)
    cutoff = c["nca"]["states"]**4
    for row in result:
        mask = row < cutoff
        row[mask] = rng.permutation(row[mask])
    return result


def targets(tokens, c, patch_only=False):
    labels = tokens[:, 1:].clone()
    absolute = torch.arange(1, tokens.shape[1], device=tokens.device)
    mask = absolute < frame_length(c)
    labels[:, mask] = -100
    if patch_only:
        labels[labels >= c["nca"]["states"]**4] = -100
    return labels


def control_probabilities(train, c):
    cutoff, length = c["nca"]["states"]**4, c["model"]["context"]
    # Fit controls on the SAME trained prefix, never on held-out rules.
    prefix = train[:, :length+1].astype(np.int64)
    y = prefix[:, 1:]
    absolute = np.arange(1, length+1)
    mask = (absolute[None] >= frame_length(c)) & (y < cutoff)
    counts = np.bincount(y[mask], minlength=cutoff).astype(np.float64)+1
    probabilities = counts/counts.sum()
    previous = prefix[:, np.maximum(0, absolute-frame_length(c))]
    truth, before = y[mask], previous[mask]
    base = probabilities[truth]
    copy = (truth == before).astype(np.float64)
    mixture_grid = np.linspace(0, .99, 100)
    losses = [-np.log((1-q)*base+q*copy).mean() for q in mixture_grid]
    mixture = float(mixture_grid[int(np.argmin(losses))])
    return probabilities, mixture


def evaluate_controls(tokens, c, probabilities, mixture):
    length, f = c["model"]["context"], frame_length(c)
    prefix = tokens[:, :length+1].astype(np.int64)
    y = prefix[:, 1:]
    absolute = np.arange(1, length+1)
    mask = (absolute[None] >= f) & (y < len(probabilities))
    previous = prefix[:, np.maximum(0, absolute-f)]
    truth, before = y[mask], previous[mask]
    base = probabilities[truth]
    mixture_probability = (1-mixture)*base+mixture*(truth == before)
    return {"unigram_patch_nll": float(-np.log(base).mean()),
            "copy_mixture_patch_nll": float(-np.log(mixture_probability).mean()),
            "copy_mixture_weight_fitted_on_train": mixture,
            "unigram_patch_accuracy": float(np.mean(truth == probabilities.argmax())),
            "previous_frame_patch_accuracy": float(np.mean(truth == before)),
            "patch_targets": int(mask.sum())}
