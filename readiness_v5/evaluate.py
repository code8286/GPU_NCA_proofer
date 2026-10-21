"""Per-rule patch scores; model forwards/target mask use frozen v4 code."""
import numpy as np
import torch
from torch.nn import functional as F
from transfer_v4.nca import targets


@torch.no_grad()
def score(model, tokens, c):
    model.eval()
    length = c['model']['context']
    result = []
    for start in range(0, len(tokens), c['eval_batch']):
        z = torch.tensor(tokens[start:start+c['eval_batch'], :length+1].astype(np.int64))
        y = targets(z, c, patch_only=True)
        logits = model(z[:, :-1])
        loss = F.cross_entropy(logits.transpose(1, 2), y, reduction='none', ignore_index=-100)
        pred = logits.argmax(-1)
        for j in range(len(z)):
            mask = y[j] != -100
            n = int(mask.sum())
            result.append({'patch_targets': n, 'patch_nll': float(loss[j, mask].double().sum()/n),
                           'patch_accuracy': float((pred[j, mask] == y[j, mask]).double().mean())})
    return result


def mean_nll(rows):
    return sum(r['patch_nll']*r['patch_targets'] for r in rows)/sum(r['patch_targets'] for r in rows)
