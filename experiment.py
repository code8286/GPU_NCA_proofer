"""A falsifiable, matched-token pilot of graph-NCA transfer into a byte LM."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("scratch", "reserve_text", "grid_nca", "graph_nca", "shuffled_graph")
VOCAB = 274  # bytes 0..255, binary 2x2 patches 256..271, frame markers 272..273
TEXT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    assert c["width"] % c["heads"] == 0
    assert 0 < c["pre_steps"] < c["total_steps"]
    assert c["grid"] % 2 == 0 and c["grid"] >= 4
    assert c["frames"] * ((c["grid"] // 2) ** 2 + 2) > c["context"]
    assert c["precision"] == "fp32", "Pilot uses consistent fp32 on CPU and GPU."
    assert len(set(c["seeds"])) == len(c["seeds"]) and c["seeds"]
    return c


def neighbors(side, rng=None, fraction=0.0):
    """Eight typed incoming edges; periodic Moore lattice or degree-matched rewiring.

    Each relation is a permutation: rewiring preserves incoming AND outgoing degree,
    with no self or duplicate neighbors. Slot labels retain lattice direction labels.
    """
    ids = np.arange(side * side).reshape(side, side)
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    table = np.stack([np.roll(ids, (-dy, -dx), (0, 1)).ravel() for dy, dx in offsets], 1)
    if fraction:
        assert rng is not None
        for r in range(8):
            for _ in range(round(side * side * fraction / 2)):
                for _attempt in range(100):
                    a, b = rng.choice(side * side, 2, replace=False)
                    va, vb = table[a, r], table[b, r]
                    if vb != a and va != b and vb not in table[a] and va not in table[b]:
                        table[a, r], table[b, r] = vb, va
                        break
    return table


def graph_step(x, edges, w1, b1, w2, b2):
    # x [batch,nodes]; shared relational message function, then a shared node MLP.
    # One independently sampled frozen neural rule per trajectory (batch member).
    local = torch.cat((x[..., None], x[:, edges]), -1).float() * 2 - 1
    hidden = F.relu(torch.einsum("bnr,brh->bnh", local, w1) + b1[:, None])
    return (torch.einsum("bnh,bhk->bnk", hidden, w2) + b2[:, None]).argmax(-1)


def patch_tokens(frames, side):
    # Spatial patches use a fixed node order even on a graph. This serialization
    # favors the lattice and is a deliberate, documented pilot limitation.
    x = frames.reshape(*frames.shape[:-1], side, side)
    p = x[..., 0::2, 0::2] + 2*x[..., 0::2, 1::2] + 4*x[..., 1::2, 0::2] + 8*x[..., 1::2, 1::2]
    p = p.flatten(-2) + 256
    start = torch.full((*p.shape[:-1], 1), 272, device=p.device)
    end = torch.full_like(start, 273)
    return torch.cat((start, p, end), -1).flatten(1)


@torch.no_grad()
def generate(c, topology, device):
    seed_all(c["data_seed"])
    rng = np.random.default_rng(c["data_seed"])
    edge_array = neighbors(c["grid"], rng, c["rewire_fraction"] if topology == "graph" else 0)
    edges = torch.as_tensor(edge_array, dtype=torch.long, device=device)
    chunks, changed, fixed, ratios = [], [], [], []
    sync(device)
    start = time.perf_counter()
    for offset in range(0, c["trajectories"], 64):
        b = min(64, c["trajectories"] - offset)
        h = c["rule_hidden"]
        x = torch.randint(2, (b, c["grid"]**2), device=device)
        w1 = torch.randn(b, 9, h, device=device) / 3
        b1 = torch.randn(b, h, device=device) * 0.2
        w2 = torch.randn(b, h, 2, device=device) / math.sqrt(h)
        b2 = torch.randn(b, 2, device=device) * 0.2
        frames = [x]
        for _ in range(c["frames"] - 1):
            y = graph_step(x, edges, w1, b1, w2, b2)
            changed.append(float((x != y).float().mean()))
            frames.append(y)
            x = y
        fixed.extend((frames[-1] == frames[-2]).all(-1).cpu().tolist())
        chunk = patch_tokens(torch.stack(frames, 1), c["grid"]).cpu().numpy().astype(np.uint16)
        chunks.append(chunk)
        for row in chunk:
            buf = row.astype("<u2").tobytes()
            ratios.append(len(gzip.compress(buf, mtime=0))/len(buf))
    sync(device)
    tokens = np.concatenate(chunks)
    raw = tokens.astype("<u2").tobytes()
    stats = {"generation_seconds": time.perf_counter()-start,
             "tokens": tokens.size, "gzip_ratio": len(gzip.compress(raw, mtime=0))/len(raw),
             "mean_cell_change": statistics.mean(changed),
             "final_fixed_point_fraction": statistics.mean(fixed),
             "per_trajectory_gzip_quartiles": np.quantile(ratios, [.25, .5, .75]).tolist(),
             "distinct_patch_tokens": int(np.unique(tokens[(tokens >= 256) & (tokens < 272)]).size),
             "changed_edges_fraction": float(np.mean(edge_array != neighbors(c["grid"]))),
             "topology_sha256": hashlib.sha256(edge_array.tobytes()).hexdigest()}
    return tokens, stats, edge_array


def shuffle_trajectories(tokens, seed):
    """Destroy patch dynamics, preserving every row's unigram counts AND markers."""
    rng = np.random.default_rng(seed)
    out = tokens.copy()
    for row in out:
        mask = row < 272
        row[mask] = rng.permutation(row[mask])
    return out


def prepare(args, c, device):
    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    # Fail closed: cached artifacts must match config and checksums at run time.
    source = Path(args.text) if args.text else root / "input.txt"
    if not source.exists():
        print("Downloading Tiny Shakespeare...", flush=True)
        urllib.request.urlretrieve(TEXT_URL, source)
    raw = source.read_bytes()
    text_start = time.perf_counter()
    n = len(raw)
    cuts = [0, int(.2*n), int(.8*n), int(.9*n), n]
    manifest = {"config": c, "source": str(source.resolve()),
                "source_url": TEXT_URL if not args.text else None,
                "source_sha256": digest(source), "splits": {}, "synthetic": {}, "files": {}}
    for i, name in enumerate(("reserve", "train", "val", "test")):
        a, b = cuts[i:i+2]
        if b-a < (c["context"]+1)*c["batch"]:
            raise ValueError("Text split too short for this configuration")
        np.save(root / f"{name}.npy", np.frombuffer(raw[a:b], dtype=np.uint8).astype(np.uint16))
        manifest["splits"][name] = {"start_byte": a, "end_byte": b, "tokens": b-a}
    manifest["text_preprocessing_seconds"] = time.perf_counter()-text_start
    for kind in ("grid", "graph"):
        tokens, stats, edges = generate(c, kind, device)
        t = time.perf_counter()
        np.save(root / f"{kind}.npy", tokens)
        np.save(root / f"{kind}_edges.npy", edges)
        stats["generation_seconds"] += time.perf_counter()-t
        manifest["synthetic"][kind] = stats
        if kind == "graph":
            t = time.perf_counter()
            shuffled = shuffle_trajectories(tokens, c["data_seed"]+1)
            np.save(root / "shuffled.npy", shuffled)
            manifest["synthetic"]["shuffled"] = {
                "generation_seconds": stats["generation_seconds"] + time.perf_counter()-t,
                "parent": "graph", "preserves": "per-trajectory patch histogram and marker positions"}
    for f in sorted(root.glob("*.npy")):
        manifest["files"][f.name] = digest(f)
    dump(root / "manifest.json", manifest)
    print(json.dumps(manifest["synthetic"], indent=2), flush=True)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["width"]
        self.heads = c["heads"]
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.norm2(x))


class LM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.token = nn.Embedding(VOCAB, c["width"])
        self.position = nn.Embedding(c["context"], c["width"])
        self.blocks = nn.Sequential(*[Block(c) for _ in range(c["layers"])])
        self.norm = nn.LayerNorm(c["width"])
        self.head = nn.Linear(c["width"], VOCAB, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Embedding, nn.Linear)):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.token(x) + self.position(torch.arange(x.shape[1], device=x.device))
        return self.head(self.norm(self.blocks(z)))


def sample_batch(array, c, rng, device):
    t, b = c["context"], c["batch"]
    if array.ndim == 1:
        starts = rng.integers(0, len(array)-t, size=b)
        chunk = array[starts[:, None] + np.arange(t+1)[None, :]]
    else:
        rows = rng.integers(0, array.shape[0], size=b)
        starts = rng.integers(0, array.shape[1]-t, size=b)
        chunk = array[rows[:, None], starts[:, None]+np.arange(t+1)[None, :]]
    chunk = torch.as_tensor(chunk.astype(np.int64), device=device)
    return chunk[:, :-1], chunk[:, 1:]


def eval_batches(array, c, device):
    # Non-overlapping targets; every arm uses exactly the same fixed subset.
    t, b = c["context"], c["batch"]
    count = min(c["eval_batches"]*b, (len(array)-1)//t)
    blocks = np.stack([array[i*t:i*t+t+1] for i in range(count)])
    return [torch.as_tensor(blocks[i:i+b].astype(np.int64), device=device) for i in range(0, count, b)]


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    total, n = 0., 0
    for z in batches:
        logits = model(z[:, :-1])
        total += float(F.cross_entropy(logits.reshape(-1, VOCAB), z[:, 1:].reshape(-1), reduction="sum"))
        n += z[:, 1:].numel()
    model.train()
    return total/n


def lr_at(step, length, c):
    warm = min(c["warmup"], max(1, length//10))
    if step < warm:
        return c["lr"]*(step+1)/warm
    phase = (step-warm)/max(1, length-warm-1)
    return c["lr"]*(.1 + .9*(1+math.cos(math.pi*phase))/2)


def warmup_training(c, array, device):
    """Pay lazy optimizer/kernel startup once, before any measured arm.

    The temporary model is discarded. Each real arm reseeds and initializes from
    scratch. The shared warmup cost is conservatively charged to every arm.
    """
    seed_all(0)
    model = LM(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"])
    rng = np.random.default_rng(0)
    for _ in range(3):
        x, y = sample_batch(array, c, rng, device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).backward()
        opt.step()
    sync(device)


def load_data(root, c):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["config"] != c:
        raise ValueError("Dataset/config mismatch; prepare a new data directory")
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    names = ("reserve", "train", "val", "test", "grid", "graph", "shuffled")
    return {n: np.load(root / f"{n}.npy") for n in names}, manifest


def run_one(c, arm, seed, arrays, manifest, device, out):
    sync(device)
    start = time.perf_counter()
    seed_all(seed)
    model = LM(c).to(device)
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    parameter_count = sum(p.numel() for p in model.parameters())
    # Synthetic symbols have no byte semantics. Reset synthetic embeddings only;
    # text baselines retain theirs. Graph/grid/shuffled use identical resets.
    embedding_init = model.token.weight.detach().clone()
    pre_name = {"scratch": "train", "reserve_text": "reserve", "grid_nca": "grid",
                "graph_nca": "graph", "shuffled_graph": "shuffled"}[arm]
    prep_seconds = (manifest["synthetic"].get(pre_name, {}).get("generation_seconds", 0.)
                    + manifest["text_preprocessing_seconds"] + manifest.get("load_seconds", 0.))
    validation = eval_batches(arrays["val"], c, device)
    pre_rng = np.random.default_rng(seed+100)
    target_rng = np.random.default_rng(seed+200)
    train_seconds, rows, target_step = 0., [], 0
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)

    def record(step, stage):
        loss = evaluate(model, validation)
        sync(device)
        pre_done = min(step, c["pre_steps"])
        # scratch's first-stage text tokens count as downstream target exposure.
        target_tokens = (max(0, step-c["pre_steps"])+(pre_done if arm == "scratch" else 0))*c["batch"]*c["context"]
        row = {"arm": arm, "seed": seed, "step": step, "stage": stage,
               "total_tokens": step*c["batch"]*c["context"], "target_text_tokens": target_tokens,
               "val_nll": loss, "val_perplexity": math.exp(loss),
               "train_seconds": train_seconds,
               "pipeline_seconds": prep_seconds+time.perf_counter()-start}
        rows.append(row)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False)+"\n")
        print(f"{arm} seed={seed} step={step} val={loss:.4f} total_tokens={row['total_tokens']}", flush=True)

    record(0, "initial")
    for stage, length, name, rng in (("pre", c["pre_steps"], pre_name, pre_rng),
                                     ("target", c["total_steps"]-c["pre_steps"], "train", target_rng)):
        if stage == "target" and arm in ("grid_nca", "graph_nca", "shuffled_graph"):
            with torch.no_grad():
                model.token.weight.copy_(embedding_init)
        # Every arm resets Adam and stage scheduler at the same boundary.
        optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        if stage == "target":
            record(c["pre_steps"], "target_start")
        for local_step in range(length):
            sync(device)
            tick = time.perf_counter()
            x, y = sample_batch(arrays[name], c, rng, device)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(local_step, length, c)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            sync(device)
            train_seconds += time.perf_counter()-tick
            target_step += 1
            if target_step % c["eval_every"] == 0 or local_step == length-1:
                record(target_step, stage)
    test_nll = evaluate(model, eval_batches(arrays["test"], c, device))
    torch.save({"model": model.state_dict(), "config": c, "arm": arm, "seed": seed,
                "initial_sha256": initial_hash, "manifest": manifest}, out / "model.pt")
    summary = {"arm": arm, "seed": seed, "parameters": parameter_count,
               "initial_sha256": initial_hash, "test_nll": test_nll,
               "test_perplexity": math.exp(test_nll), "final_val_nll": rows[-1]["val_nll"],
               "train_seconds": train_seconds, "preparation_seconds": prep_seconds,
               "val_evaluated_tokens": sum(z[:, 1:].numel() for z in validation),
               "embedding_transfer": "reset" if arm in ("grid_nca", "graph_nca", "shuffled_graph") else "preserved",
               "total_tokens": rows[-1]["total_tokens"],
               "approx_train_flops_6PD": 6*parameter_count*rows[-1]["total_tokens"],
               "model_sha256": digest(out / "model.pt")}
    dump(out / "summary.json", summary)
    return summary


def first_hit(rows, threshold, key):
    # Text baselines can reach the target during their first stage. Synthetic
    # stages are excluded until their byte embeddings have been restored.
    eligible = [r for r in rows if (r["stage"] in ("target_start", "target") or
                (r.get("arm") in ("scratch", "reserve_text") and r["stage"] == "pre"))
                and r["val_nll"] <= threshold]
    return min((r[key] for r in eligible), default=None)


def report(out, c):
    out = Path(out)
    summaries, curves = {}, {}
    for arm in ARMS:
        for seed in c["seeds"]:
            folder = out / f"{arm}_{seed}"
            if (folder / "summary.json").exists():
                summaries[arm, seed] = json.loads((folder / "summary.json").read_text())
                curves[arm, seed] = [json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()]
    comparisons = []
    for baseline in ("scratch", "grid_nca", "reserve_text", "shuffled_graph"):
        for seed in c["seeds"]:
            if (baseline, seed) not in summaries or ("graph_nca", seed) not in summaries:
                continue
            a, b = summaries[baseline, seed], summaries["graph_nca", seed]
            row = {"baseline": baseline, "seed": seed,
                   "graph_minus_baseline_test_nll": b["test_nll"]-a["test_nll"]}
            for key in ("total_tokens", "target_text_tokens", "train_seconds", "pipeline_seconds"):
                ah = first_hit(curves[baseline, seed], c["target_nll"], key)
                bh = first_hit(curves["graph_nca", seed], c["target_nll"], key)
                row[f"{key}_speedup"] = ah/bh if ah is not None and bh is not None and bh > 0 else None
            row["target_status"] = "both_reached" if row["pipeline_seconds_speedup"] is not None else "unresolved_not_both_reached"
            comparisons.append(row)
    aggregate = []
    for baseline in ("scratch", "grid_nca", "reserve_text", "shuffled_graph"):
        pairs = [r for r in comparisons if r["baseline"] == baseline]
        if not pairs:
            continue
        values = [r["graph_minus_baseline_test_nll"] for r in pairs]
        rng = np.random.default_rng(123)
        boot = np.mean(rng.choice(values, (10000, len(values)), replace=True), 1)
        speedups = [r["pipeline_seconds_speedup"] for r in pairs if r["pipeline_seconds_speedup"] is not None]
        speedup_ci = None
        if len(speedups) >= 2:
            samples = np.exp(np.mean(rng.choice(np.log(speedups), (10000, len(speedups)), replace=True), 1))
            speedup_ci = np.quantile(samples, [.025, .975]).tolist()
        aggregate.append({"baseline": baseline, "paired_seeds": len(values),
                          "mean_test_nll_delta": statistics.mean(values),
                          "bootstrap_95_ci_exploratory": np.quantile(boot, [.025, .975]).tolist() if len(values) >= 2 else None,
                          "pairs_both_reached_target": len(speedups),
                          "geomean_pipeline_speedup_reached_pairs_only": math.exp(statistics.mean(map(math.log, speedups))) if speedups else None,
                          "speedup_bootstrap_95_ci_exploratory": speedup_ci,
                          "speedup_caution": "Exclude no seed silently: if any seed is censored, reached-only speedup does not establish a suite-wide gain.",
                          "caution": "Three seeds are exploratory; one seed has no uncertainty estimate."})
    dump(out / "report.json", {"threshold_nll": c["target_nll"], "comparisons": comparisons,
                               "aggregate": aggregate, "claim": "No automatic proof: inspect paired results and measured time-to-target."})
    if summaries:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(next(iter(summaries.values()))))
            writer.writeheader()
            writer.writerows(summaries.values())
    write_plot(out, curves)
    print(json.dumps(aggregate, indent=2), flush=True)


def write_plot(out, curves):
    # Portable SVG: no plotting dependency needed in Kaggle or on a CPU laptop.
    colors = dict(zip(ARMS, ("#64748b", "#b7791f", "#2563eb", "#059669", "#db2777")))
    rows = [r for curve in curves.values() for r in curve if r["stage"] in ("target_start", "target")]
    if not rows:
        return
    lo, hi = min(r["val_nll"] for r in rows)-.05, max(r["val_nll"] for r in rows)+.05
    xmax = max(r["total_tokens"] for r in rows)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">',
             '<rect width="900" height="520" fill="white"/>',
             '<text x="70" y="28" font-family="sans-serif" font-size="20">Validation loss vs total training tokens (all stages counted)</text>',
             '<path d="M70 50V440H850" fill="none" stroke="#334155"/>']
    for i in range(5):
        y = 440-i*95
        value = lo+(hi-lo)*i/4
        parts.append(f'<text x="12" y="{y+4}" font-family="sans-serif" font-size="12">{value:.2f}</text>')
        parts.append(f'<text x="{70+i*195}" y="462" font-family="sans-serif" font-size="12">{xmax*i/4:,.0f}</text>')
    for (arm, seed), curve in curves.items():
        points = " ".join(f'{70+780*r["total_tokens"]/xmax:.1f},{440-380*(r["val_nll"]-lo)/(hi-lo):.1f}'
                          for r in curve if r["stage"] in ("target_start", "target"))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[arm]}" stroke-width="1.8" opacity="0.65"><title>{arm}, seed {seed}</title></polyline>')
    for i, arm in enumerate(ARMS):
        parts.append(f'<text x="{70+i*160}" y="500" fill="{colors[arm]}" font-family="sans-serif" font-size="12">{arm}</text>')
    parts.append('</svg>')
    (out / "curves.svg").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "all"))
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--data", default="data/pilot")
    parser.add_argument("--out", default="runs/pilot")
    parser.add_argument("--text", help="Existing UTF-8/plain text corpus; no download needed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    c = load_config(args.config)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Determinism is checked, not silently best-effort; GPU CUBLAS config must be
    # set before launching Python, as done by the notebook.
    torch.use_deterministic_algorithms(True)
    if args.command in ("prepare", "all"):
        prepare(args, c, device)
    if args.command in ("run", "all"):
        load_start = time.perf_counter()
        arrays, manifest = load_data(args.data, c)
        warmup_training(c, arrays["train"], device)
        manifest["load_seconds"] = time.perf_counter()-load_start
        root = Path(args.out)
        root.mkdir(parents=True, exist_ok=True)
        env_path = root / "environment.json"
        if env_path.exists():
            raise FileExistsError("Output directory already has a run. Choose a new --out path.")
        dump(env_path, {"config": c, "python": sys.version, "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(), "device": device,
                        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                        "cuda": torch.version.cuda, "source_sha256": digest(__file__),
                        "tf32": torch.backends.cuda.matmul.allow_tf32,
                        "data_manifest_sha256": digest(Path(args.data)/"manifest.json"),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "interpretation": "CPU/smoke outputs validate execution only; not a GPU speed claim."})
        # Counterbalance arm order by seed to reduce systematic warm-up/order bias.
        for i, seed in enumerate(c["seeds"]):
            order = args.arms[i % len(args.arms):]+args.arms[:i % len(args.arms)]
            for arm in order:
                run_one(c, arm, seed, arrays, manifest, device, root / f"{arm}_{seed}")
    if args.command in ("run", "report", "all"):
        report(args.out, c)


if __name__ == "__main__":
    main()
