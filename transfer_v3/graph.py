"""Slot-labeled, degree-preserving graph NCA and paired rule selection.

Grid and graph use the SAME message function. Only the donor-node indices change.
The 2x2 tokenizer uses fixed node labels, not an arbitrary graph layout optimizer.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import torch

from .common import frame_length, sync
from .nca import complexity, encode_frames, lattice, rollout, sample_rules


def topology_hash(edges):
    return hashlib.sha256(np.asarray(edges, dtype="<i8").tobytes()).hexdigest()


def validate_topology(edges, side):
    edges = np.asarray(edges)
    nodes = side * side
    if edges.shape != (nodes, 9) or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Expected nine integer donor slots per node")
    if np.any(edges < 0) or np.any(edges >= nodes) or not np.array_equal(edges[:, 4], np.arange(nodes)):
        raise ValueError("Invalid donor index or center/self slot")
    if any(len(set(row.tolist())) != 9 for row in edges):
        raise ValueError("Duplicate neighbors or extra self edges")
    if any(not np.array_equal(np.sort(edges[:, j]), np.arange(nodes)) for j in range(9)):
        raise ValueError("Each slot must preserve one incoming/outgoing edge per node")


def rewire(side, fraction, seed):
    if side < 4 or not 0 <= fraction <= 1:
        raise ValueError("Invalid graph size/fraction")
    original = lattice(side)
    edges = original.copy()
    rng = np.random.default_rng(seed)
    slots = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    desired = math.ceil(fraction * side * side * 8)
    attempts = swaps = 0
    while np.count_nonzero(edges[:, slots] != original[:, slots]) < desired:
        attempts += 1
        if attempts > 1000 * side * side * 8:
            raise RuntimeError("Cannot reach declared rewiring fraction without violating graph invariants")
        j = int(rng.choice(slots))
        u, v = rng.choice(side * side, 2, replace=False)
        a, b = edges[u, j], edges[v, j]
        if b in edges[u] or a in edges[v]:
            continue
        edges[u, j], edges[v, j] = b, a
        swaps += 1
    validate_topology(edges, side)
    changed = float(np.mean(edges[:, slots] != original[:, slots]))
    return edges, {"seed": seed, "requested_changed_fraction": fraction,
                   "actual_changed_fraction": changed, "accepted_swaps": swaps,
                   "attempts": attempts, "sha256": topology_hash(edges),
                   "grid_sha256": topology_hash(original),
                   "kind": "Directed slot-labeled donor graph; fixed self slot; eight distinct other donors; per-slot degree preserved"}


def paired_rollout(rules, c, frames, generator, device, edges):
    before = generator.get_state()
    grid = rollout(rules, c, frames, generator, device)
    after = generator.get_state()
    generator.set_state(before)
    graph = rollout(rules, c, frames, generator, device, edges=edges)
    if not torch.equal(after, generator.get_state()):
        raise RuntimeError("Paired rollout random draws diverged")
    return grid, graph


@torch.no_grad()
def generate_paired(c, count, split, replicate_seed, device, edges):
    n = c["nca"]
    if n["channel_padding"] != "spatial_only":
        raise ValueError("Spatial-only padding is required")
    seed = n["data_seed"] + 100003 * replicate_seed + (0 if split == "train" else 1000003)
    generator = torch.Generator(device=device).manual_seed(seed)
    frames = math.ceil((c["model"]["context"] + 1) / frame_length(c))
    arrays = {"nca": [], "graph": []}
    profile_scores = {k: [] for k in arrays}
    fresh_scores = {k: [] for k in arrays}
    fixed = {k: [] for k in arrays}
    saved_rules, ids = {}, []
    candidates = accepted = 0
    individual_passes = {k: 0 for k in arrays}
    started = time.perf_counter()
    while accepted < count:
        b = min(n["candidate_batch"], n["max_candidates"] - candidates)
        if b <= 0:
            raise RuntimeError(f"Paired complexity filter exhausted {candidates} candidates at {accepted}/{count}; no relaxation")
        rules = sample_rules(b, n["states"], generator, device)
        profiles = paired_rollout(rules, c, n["profile_frames"], generator, device, edges)
        scores = {k: complexity(encode_frames(x, n["grid"], n["states"]).cpu().numpy(), c)
                  for k, x in zip(arrays, profiles)}
        passes = {k: (v > n["filter_low"]) & (v < n["filter_high"]) for k, v in scores.items()}
        for k in arrays:
            individual_passes[k] += int(passes[k].sum())
        selected = np.flatnonzero(passes["nca"] & passes["graph"])[:count - accepted]
        candidate_start = candidates
        candidates += b
        if len(selected):
            chosen = {k: v[selected.tolist()] for k, v in rules.items()}
            actuals = paired_rollout(chosen, c, frames, generator, device, edges)
            for k, actual in zip(arrays, actuals):
                tokens = encode_frames(actual, n["grid"], n["states"]).cpu().numpy().astype(np.uint16)
                arrays[k].append(tokens)
                profile_scores[k].extend(scores[k][selected].tolist())
                fresh_scores[k].extend(complexity(tokens, c).tolist())
                fixed[k].extend((actual[:, -1] == actual[:, -2]).all(-1).cpu().tolist())
            for k, v in chosen.items():
                saved_rules.setdefault(k, []).append(v.cpu().numpy())
            ids.extend(f"{split}:{seed}:{candidate_start + int(i)}" for i in selected)
            accepted += len(selected)
        if candidates % (n["candidate_batch"] * 20) == 0 or accepted == count:
            print(f"Paired {split}, replicate {replicate_seed}: {accepted}/{count}; candidates={candidates}", flush=True)
    sync(device)
    stats = {"replicate_seed": replicate_seed, "generation_seed": seed, "accepted": accepted,
             "candidates": candidates, "rule_ids": ids, "individual_band_passes": individual_passes,
             "generation_seconds": time.perf_counter() - started,
             "selection": "Same rule retained only if BOTH independent grid/graph profiles pass the same strict band",
             "randomness": "Common initial states and sampling draws within each paired rollout; fresh actual rollout after screening",
             "fresh_rollout_note": "Fresh initial conditions may produce complexity outside the screening band"}
    for k in arrays:
        stats[k] = {"profile_scores": profile_scores[k],
                    "profile_gzip_quartiles": np.quantile(profile_scores[k], [.25, .5, .75]).tolist(),
                    "fresh_rollout_gzip_quartiles": np.quantile(fresh_scores[k], [.25, .5, .75]).tolist(),
                    "final_fixed_transition_fraction": float(np.mean(fixed[k]))}
    return ({k: np.concatenate(v) for k, v in arrays.items()},
            {k: np.concatenate(v) for k, v in saved_rules.items()}, stats)
