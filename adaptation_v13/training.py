"""Answer-only SFT with complete prompt conditioning and original tied LM head."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def encode(row, tokenizer, context):
    p = tokenizer.encode(row['prompt'])
    a = tokenizer.encode(row['references'][0])+[tokenizer.eot_token]
    if not p or not a or len(p)+len(a)-1 > context:
        raise ValueError('Entire prompt and answer must fit the original context')
    ids = p+a
    return {'id': row['id'], 'input': ids[:-1],
            'target': [-100]*(len(p)-1)+a}


def collate(rows, device='cpu'):
    length = max(len(r['input']) for r in rows)
    x = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for i, r in enumerate(rows):
        x[i, :len(r['input'])] = torch.tensor(r['input'], device=device)
        y[i, :len(r['target'])] = torch.tensor(r['target'], device=device)
    return x, y


def answer_loss(model, x, y):
    if x.shape != y.shape or x.shape[1] > model.context or not (y != -100).any():
        raise ValueError('Require aligned nonempty answer targets within context')
    z = model.core(model.embedding(x))
    mask = y != -100
    return F.cross_entropy(F.linear(z[mask], model.embedding.weight), y[mask])


def order(count, updates, batch, seed):
    rng = np.random.default_rng(seed)
    indices = []
    while len(indices) < updates*batch:
        indices.extend(rng.permutation(count).tolist())
    return np.array(indices[:updates*batch], dtype=np.int64).reshape(updates, batch)


def learning_rate(step, c):
    # Public step is one-based; exactly warmup peak then endpoint .1 * LR.
    if not 1 <= step <= c['updates']:
        raise ValueError('Step outside frozen training budget')
    warm = c['warmup_updates']
    if step <= warm:
        return c['learning_rate']*step/warm
    progress = (step-warm)/(c['updates']-warm)
    return c['learning_rate']*(.1+.9*(1+math.cos(math.pi*progress))/2)


@torch.no_grad()
def nll(model, examples, batch):
    model.eval()
    total = targets = 0
    for start in range(0, len(examples), batch):
        x, y = collate(examples[start:start+batch])
        count = int((y != -100).sum())
        total += float(answer_loss(model, x, y))*count
        targets += count
    return {'answer_eos_nll': total/targets, 'supervised_targets': targets, 'examples': len(examples)}


def decision(curve, c):
    from ood_v12.scoring import accuracy_crossing
    final = curve[-1]
    if final['step'] != c['updates']:
        raise ValueError('Require fixed final endpoint')
    passed = (final['exact_match'] >= c['id_exact_match_target']
              and final['valid_format'] >= c['id_format_target']
              and final['coverage'] == 1
              and all(r['exact_match'] >= c['id_exact_match_target'] for r in final['cells'])
              and len(final['cells']) == 4)
    return {'passed': passed, 'first_pooled_crossing': accuracy_crossing(curve, c['id_exact_match_target'])}
