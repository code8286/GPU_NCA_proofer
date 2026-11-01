"""Greedy answers from trusted local LM checkpoints; no target conditioning."""
import re
import torch
from torch.nn import functional as F


@torch.no_grad()
def last_logits(model, x):
    # Same LM computation, projecting only the final hidden position instead of
    # allocating a vocabulary-sized tensor at every prompt position.
    hidden = model.core(model.embedding(x))
    return F.linear(hidden[:, -1], model.embedding.weight)


def impossible_prefix(text):
    # This is early rejection, not constrained decoding: logits are never masked.
    # Accepted answers contain one bracketed integer vector and whitespace only.
    t = text.lstrip()
    if not t:
        return False
    if not t.startswith('[') or re.search(r'[^\[\]\d,\s-]', t):
        return True
    if '[' in t[1:] or t.count(']') > 1:
        return True
    return ']' in t and bool(t.split(']', 1)[1].strip())


@torch.no_grad()
def greedy(model, tokenizer, inputs, max_new_tokens=128):
    """Predictor input is deliberately restricted to ID/prompt, not gold data."""
    model.eval()
    device = next(model.parameters()).device
    result = []
    for row in inputs:
        if set(row) != {'id', 'prompt'}:
            raise ValueError('Predictor must receive prompt and identity only')
        prefix = tokenizer.encode(row['prompt'])
        if not prefix or len(prefix)+max_new_tokens > model.context:
            result.append({'id': row['id'], 'text': '', 'status': 'context_overflow', 'generated_tokens': [], 'stop_reason': 'preflight_rejection'})
            continue
        generated, stop = [], 'token_limit'
        for _ in range(max_new_tokens):
            x = torch.tensor([prefix+generated], dtype=torch.long, device=device)
            token = int(last_logits(model, x).argmax(-1))
            if token == tokenizer.eot_token:
                stop = 'eos'
                break
            generated.append(token)
            text = tokenizer.decode(generated)
            if impossible_prefix(text):
                stop = 'irrecoverable_format_prefix'
                break
            if ']' in text:
                stop = 'closing_bracket'
                break
        result.append({'id': row['id'], 'text': tokenizer.decode(generated), 'status': 'ok',
                       'generated_tokens': generated, 'stop_reason': stop})
    return result
