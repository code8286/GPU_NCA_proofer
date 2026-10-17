"""Small RoPE/RMSNorm/SwiGLU causal transformer; no pretrained external weights."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import state_sha


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-6).to(x.dtype)*self.weight


def rope(x):
    # Interleaved real/imaginary coordinates; equivalent rotary formulation.
    t, d = x.shape[-2:]
    frequency = 10000.0**(-torch.arange(0, d, 2, device=x.device).float()/d)
    angle = torch.arange(t, device=x.device).float()[:, None]*frequency[None]
    a, b = x[..., 0::2], x[..., 1::2]
    co, si = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    return torch.stack((a*co-b*si, a*si+b*co), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, h = config["width"], config["heads"]
        hidden = ((int(8*d/3)+63)//64)*64
        self.heads = h
        self.attention_norm, self.ffn_norm = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
        self.gate, self.up, self.down = (nn.Linear(d, hidden, bias=False),
                                       nn.Linear(d, hidden, bias=False), nn.Linear(hidden, d, bias=False))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).reshape(b, t, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, dropout_p=0., is_causal=True)
        x = x+self.output(a.transpose(1, 2).reshape(b, t, d))
        z = self.ffn_norm(x)
        return x+self.down(F.silu(self.gate(z))*self.up(z))


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config["layers"])])
        self.norm = RMSNorm(config["width"])

    def forward(self, x):
        return self.norm(self.blocks(x))


class LM(nn.Module):
    def __init__(self, config, vocab, core_seed, head_seed):
        super().__init__()
        # Separate streams ensure vocabulary-size changes cannot change the
        # paired initial backbone. Construction happens on CPU, then moves.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(core_seed)
            self.core = Backbone(config)
            for module in self.core.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=.02)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.embedding = nn.Embedding(vocab, config["width"])
            nn.init.normal_(self.embedding.weight, std=.02)
        self.vocab = vocab
        self.context = config["context"]

    def forward(self, tokens):
        if tokens.shape[1] > self.context:
            raise ValueError("Sequence exceeds declared context")
        z = self.core(self.embedding(tokens))
        return F.linear(z, self.embedding.weight)  # Explicitly tied input/output table.

    def core_hash(self):
        return state_sha(self.core.state_dict())

    def head_hash(self):
        return state_sha(self.embedding.state_dict())


def language_model(c, vocab, seed, ppt_state=None):
    model = LM(c["model"], vocab, core_seed=seed, head_seed=seed+100_000)
    if ppt_state is not None:
        # Never transfer synthetic embeddings, optimizer moments, or scheduler.
        model.core.load_state_dict(ppt_state, strict=True)
    return model
