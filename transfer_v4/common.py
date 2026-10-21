from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

ARMS = ("scratch", "grid_nca", "graph_nca", "shuffled_nca", "shuffled_graph", "prefix_grid", "prefix_graph")
SYNTHETIC_ARMS = ARMS[1:]
DATA_KEYS = {"grid_nca": "nca", "graph_nca": "graph",
             "shuffled_nca": "shuffled", "shuffled_graph": "shuffled_graph",
             "prefix_grid": "prefix_grid", "prefix_graph": "prefix_graph"}
REFERENCE_COMMIT = "bdd1c71e04eb2e5a2c91481faf278e95846b4431"
STUDY = {"id": "prefix_frequency_control_v1", "target": "paired_scratch_endpoint",
         "primary_pairs": [["grid_nca", "graph_nca"], ["prefix_graph", "graph_nca"]],
         "alpha_per_comparison": .025, "minimum_cohorts": 6,
         "control": "initial_frame_and_supervised_prefix_histogram",
         "crossing": "first_recorded_no_interpolation_all_cohorts_required"}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parent
    return {p.name: sha_file(p) for p in sorted(root.glob("*.py"))}


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(c)
    return c


def validate_config(c):
    if c["schema_version"] != 4:
        raise ValueError("Expected a mechanism-transfer v4 configuration")
    if c["purpose"] not in ("development", "execution_check"):
        raise ValueError("Unknown run purpose")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("Model seeds must be unique")
    m, n = c["model"], c["nca"]
    if m["width"] % m["heads"] or (m["width"]//m["heads"]) % 2:
        raise ValueError("RoPE requires an even head dimension")
    if n["grid"] % 2 or n["states"] < 2:
        raise ValueError("Expected an even grid and a categorical alphabet")
    if n["states"]**4+2 > 65536:
        raise ValueError("Synthetic token IDs must fit uint16")
    if m["context"] < 2*frame_length(c):
        raise ValueError("Context must contain at least two complete frames")
    if any(c[stage]["steps"] <= 0 for stage in ("ppt", "language")):
        raise ValueError("PPT and language budgets are separate positive lengths")
    if not 0 <= n["filter_low"] < n["filter_high"]:
        raise ValueError("Invalid complexity band")
    if c["purpose"] == "development" and not c["gate"]["enforce"]:
        raise ValueError("Development runs must enforce the held-out synthetic gate")
    if c["purpose"] == "development" and c["text"]["tokenizer"] != "gpt2":
        raise ValueError("Byte fixtures are restricted to execution checks")
    if c["precision"] != "fp32":
        raise ValueError("This audited version uses fp32 in both stages")
    if n["grid"] < 4 or not 0 <= c["graph"]["rewire_fraction"] <= 1:
        raise ValueError("Graph needs side >=4 and a rewiring fraction in [0,1]")
    if c["purpose"] == "development" and c["graph"]["rewire_fraction"] <= 0:
        raise ValueError("Development graph treatment must change topology")
    if c["graph"]["rule_selection"] != "intersection" or c["graph"]["replication"] != "independent_cache_per_model_seed":
        raise ValueError("Use paired intersection filtering and independent replicate caches")
    if c.get("study") != STUDY:
        raise ValueError("The v4 study criterion is frozen; declare a new version to change it")
    if c["purpose"] == "development":
        if len(c["seeds"]) < STUDY["minimum_cohorts"] or set(c["seeds"]) & {17, 29, 43}:
            raise ValueError("Use at least six fresh cohorts, excluding the inspected v3 seeds")
        if c["text"]["evaluate_test"]:
            raise ValueError("V4 is a validation-stage replication/mechanism study; reserve the official test")
    if any(c[key] <= 0 for key in ("eval_every", "eval_batch", "batch", "accumulate", "ppt_eval_rules")):
        raise ValueError("Batch and evaluation counts must be positive")


def frame_length(c):
    return (c["nca"]["grid"]//2)**2+2


def effective_tokens(c):
    return c["batch"]*c["accumulate"]*c["model"]["context"]


def lr_at(step, stage):
    length = stage["steps"]
    warm = max(1, min(length, round(length*stage["warmup_fraction"])))
    if step < warm:
        return stage["lr"]*(step+1)/warm
    fraction = (step-warm)/max(1, length-warm-1)
    return stage["lr"]*(.1+.9*(1+math.cos(math.pi*fraction))/2)
