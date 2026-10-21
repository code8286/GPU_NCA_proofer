from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import DATA_KEYS, dump, effective_tokens, lr_at, seed_all, sha_file, sync
from .model import LM, language_model
from .nca import control_probabilities, evaluate_controls, targets


def append(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False)+"\n")


def batch(array, c, rng, device, synthetic):
    length, size = c["model"]["context"], c["batch"]
    if synthetic:
        indices = rng.integers(0, len(array), size=size)
        values = array[indices, :length+1]
    else:
        if len(array) <= length:
            raise ValueError("Language training split is shorter than context")
        indices = rng.integers(0, len(array)-length, size=size)
        values = array[indices[:, None]+np.arange(length+1)[None]]
    tensor = torch.as_tensor(values.astype(np.int64), device=device)
    labels = targets(tensor, c) if synthetic else tensor[:, 1:]
    return tensor[:, :-1], labels, indices


def eval_rows(array, c, synthetic):
    length = c["model"]["context"]
    if synthetic:
        return array[:min(len(array), c["ppt_eval_rules"]), :length+1]
    count = min(c["language_eval_tokens"]//length, (len(array)-1)//length)
    if count < 1:
        raise ValueError("Evaluation split too short")
    # Target tokens do not overlap across rows. Adjacent rows share only one
    # conditioning boundary token, as in the historical non-overlap evaluation.
    return np.stack([array[i*length:i*length+length+1] for i in range(count)])


@torch.no_grad()
def evaluate(model, array, c, device, synthetic):
    model.eval()
    rows = eval_rows(array, c, synthetic)
    total_loss = patch_loss = 0.
    count = patch_count = correct = 0
    for start in range(0, len(rows), c["eval_batch"]):
        z = torch.as_tensor(rows[start:start+c["eval_batch"]].astype(np.int64), device=device)
        labels = targets(z, c) if synthetic else z[:, 1:]
        logits = model(z[:, :-1])
        losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=-100)
        mask = labels != -100
        total_loss += float(losses[mask].sum())
        count += int(mask.sum())
        if synthetic:
            patch_mask = mask & (labels < c["nca"]["states"]**4)
            patch_loss += float(losses[patch_mask].sum())
            patch_count += int(patch_mask.sum())
            correct += int(((logits.argmax(-1) == labels) & patch_mask).sum())
    model.train()
    result = {"nll": total_loss/count, "targets": count}
    if synthetic:
        result.update(patch_nll=patch_loss/patch_count, patch_accuracy=correct/patch_count,
                      patch_targets=patch_count, rules=len(rows))
    return result


def optimizer_for(model, stage, synthetic):
    if synthetic:
        return torch.optim.Adam(model.parameters(), lr=stage["lr"], weight_decay=stage["weight_decay"])
    return torch.optim.AdamW(model.parameters(), lr=stage["lr"], weight_decay=stage["weight_decay"])


def update(model, optimizer, array, c, stage, local_step, rng, device, synthetic, order_hash):
    lr = lr_at(local_step, stage)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    prediction_targets = 0
    for _ in range(c["accumulate"]):
        x, y, indices = batch(array, c, rng, device, synthetic)
        order_hash.update(indices.astype("<i8").tobytes())
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        (loss/c["accumulate"]).backward()
        loss_sum += float(loss.detach())/c["accumulate"]
        prediction_targets += int((y != -100).sum())
    if stage["clip"] is not None:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), stage["clip"], error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
    else:
        # A finite global-norm check without clipping; avoids synchronizing the
        # GPU separately for every parameter tensor.
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return loss_sum, prediction_targets, lr


def quality_gate(initial, selected, controls, c):
    relative = (initial["patch_nll"]-selected["patch_nll"])/initial["patch_nll"]
    baseline = min(controls["unigram_patch_nll"], controls["copy_mixture_patch_nll"])
    margin = baseline-selected["patch_nll"]
    passed = relative >= c["gate"]["minimum_relative_improvement"] and margin >= c["gate"]["baseline_margin_nats"]
    return {"passed": bool(passed), "enforced": c["gate"]["enforce"],
            "relative_patch_nll_improvement": relative, "improvement_over_best_control_nats": margin,
            "required_relative_improvement": c["gate"]["minimum_relative_improvement"],
            "required_baseline_margin_nats": c["gate"]["baseline_margin_nats"],
            "control_scope": "Unseen-rule validation; control probabilities fitted only on synthetic train prefixes"}


def train_ppt(c, arm, seed, arrays, manifest_hash, root, device):
    if arm not in DATA_KEYS:
        raise ValueError("Scratch has no PPT stage")
    sync(device)
    started = time.perf_counter()
    folder = Path(root)/f"{arm}_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    kind = DATA_KEYS[arm]
    train, validation = arrays[f"{kind}_train"], arrays[f"{kind}_validation"]
    model = LM(c["model"], c["nca"]["states"]**4+2, core_seed=seed, head_seed=seed+50_000).to(device)
    initialization = torch.load(Path(root)/f"initialization_{seed}.pt", map_location=device, weights_only=True)
    model.core.load_state_dict(initialization["core"], strict=True)
    initial_core = model.core_hash()
    stage = c["ppt"]
    optimizer = optimizer_for(model, stage, synthetic=True)
    rng = np.random.default_rng(seed+101)
    order_hash = hashlib.sha256()
    probabilities, mixture = control_probabilities(train, c)
    selected_rows = eval_rows(validation, c, True)
    controls = evaluate_controls(selected_rows, c, probabilities, mixture)
    initial = evaluate(model, validation, c, device, True)
    best, best_step = initial, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = 0.
    predicted = 0
    window_loss = 0.
    window_count = 0
    append(folder/"ppt_metrics.jsonl", {"ppt_step": 0, "train_loss": None, "validation": initial,
                                       "train_seconds": 0., "stage_seconds": time.perf_counter()-started})
    for k in range(stage["steps"]):
        sync(device)
        tick = time.perf_counter()
        loss, count, lr = update(model, optimizer, train, c, stage, k, rng, device, True, order_hash)
        sync(device)
        train_seconds += time.perf_counter()-tick
        window_loss += loss
        window_count += 1
        predicted += count
        if (k+1) % c["eval_every"] == 0 or k+1 == stage["steps"]:
            measured = evaluate(model, validation, c, device, True)
            sync(device)
            if measured["patch_nll"] < best["patch_nll"]:
                best, best_step = measured, k+1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            row = {"ppt_step": k+1, "ppt_tokens": (k+1)*effective_tokens(c),
                   "train_loss": window_loss/window_count, "lr": lr, "validation": measured,
                   "train_seconds": train_seconds, "stage_seconds": time.perf_counter()-started}
            append(folder/"ppt_metrics.jsonl", row)
            print(f"PPT {arm} seed={seed} {k+1}/{stage['steps']}: held-out patch NLL={measured['patch_nll']:.4f}, best control={min(controls['unigram_patch_nll'], controls['copy_mixture_patch_nll']):.4f}", flush=True)
            window_loss = 0.
            window_count = 0
    checkpoint = {"schema_version": 4, "config": c, "stage": "ppt", "arm": arm, "seed": seed,
                  "vocab": model.vocab, "data_manifest_sha256": manifest_hash,
                  "core_initial_sha256": initial_core, "selected_step": best_step,
                  "model": best_state, "validation": best}
    torch.save(checkpoint, folder/"ppt_selected.pt")
    # Preserve both selected and final states if validation selection differs.
    if best_step != stage["steps"]:
        final_checkpoint = dict(checkpoint, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                selected_step=stage["steps"], validation=measured)
        torch.save(final_checkpoint, folder/"ppt_final.pt")
    gate = quality_gate(initial, best, controls, c)
    summary = {"arm": arm, "seed": seed, "ppt_updates": stage["steps"],
               "ppt_tokens": stage["steps"]*effective_tokens(c), "ppt_prediction_targets": predicted,
               "initial": initial, "selected": best, "selected_step": best_step,
               "controls": controls, "gate": gate, "core_initial_sha256": initial_core,
               "train_order_sha256": order_hash.hexdigest(), "train_seconds": train_seconds,
               "stage_seconds": time.perf_counter()-started,
               "checkpoint": "ppt_selected.pt", "checkpoint_sha256": sha_file(folder/"ppt_selected.pt"),
               "final_checkpoint_sha256": sha_file(folder/"ppt_final.pt") if (folder/"ppt_final.pt").exists() else sha_file(folder/"ppt_selected.pt"),
               "selection_policy": "Minimum held-out-rule patch NLL; full declared PPT budget is charged regardless of selected step"}
    dump(folder/"ppt_summary.json", summary)
    return summary


def load_ppt_core(folder, c, arm, seed, manifest_hash):
    folder = Path(folder)
    summary = json.loads((folder/"ppt_summary.json").read_text())
    path = folder/summary["checkpoint"]
    if sha_file(path) != summary["checkpoint_sha256"]:
        raise ValueError("PPT checkpoint checksum mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["config"] != c or saved["arm"] != arm or saved["seed"] != seed or saved["data_manifest_sha256"] != manifest_hash:
        raise ValueError("PPT checkpoint identity mismatch")
    core = {k[len("core."):]: v for k, v in saved["model"].items() if k.startswith("core.")}
    return core, summary


def train_language(c, arm, seed, arrays, manifest, manifest_hash, root, device):
    sync(device)
    setup_started = time.perf_counter()
    folder = Path(root)/f"{arm}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"language_metrics.jsonl").exists():
        raise FileExistsError("Refusing to mix language runs; use a fresh output directory")
    seed_all(seed)
    ppt_state, ppt = (None, None) if arm == "scratch" else load_ppt_core(folder, c, arm, seed, manifest_hash)
    model = language_model(c, manifest["text"]["vocab"], seed, ppt_state).to(device)
    initialization = torch.load(Path(root)/f"initialization_{seed}.pt", map_location=device, weights_only=True)
    model.embedding.load_state_dict(initialization["embedding"], strict=True)
    if ppt_state is None:
        model.core.load_state_dict(initialization["core"], strict=True)
    start_core, start_head = model.core_hash(), model.head_hash()
    stage = c["language"]
    optimizer = optimizer_for(model, stage, synthetic=False)
    if optimizer.state:
        raise RuntimeError("Language optimizer must start empty")
    rng = np.random.default_rng(seed+202)
    order_hash, schedule_hash = hashlib.sha256(), hashlib.sha256()
    train_seconds = 0.
    # Joint candidate screening/cache preparation is conservatively charged in
    # full to each synthetic arm. It is not an isolated generator speed metric.
    nca_prepare = sum(v["prepare_seconds"] for v in manifest["nca"].values()) if ppt else 0.
    setup_cost = manifest["text"]["seconds"]+nca_prepare+manifest.get("shared_finalize_seconds", 0.)
    ppt_cost = ppt["stage_seconds"] if ppt else 0.
    sync(device)
    setup_cost += time.perf_counter()-setup_started
    started = time.perf_counter()

    def record(step, train_loss, lr):
        value = evaluate(model, arrays["text_validation"], c, device, False)
        sync(device)
        elapsed = time.perf_counter()-started
        row = {"arm": arm, "seed": seed, "language_step": step,
               "language_tokens": step*effective_tokens(c),
               "ppt_tokens": ppt["ppt_tokens"] if ppt else 0,
               "total_training_tokens": step*effective_tokens(c)+(ppt["ppt_tokens"] if ppt else 0),
               "train_loss": train_loss, "lr": lr, "validation_nll": value["nll"],
               "validation_targets": value["targets"], "language_train_seconds": train_seconds,
               "language_seconds": elapsed, "cumulative_seconds": setup_cost+ppt_cost+elapsed}
        append(folder/"language_metrics.jsonl", row)
        print(f"LANGUAGE {arm} seed={seed} {step}/{stage['steps']}: validation NLL={value['nll']:.4f}", flush=True)
        return row

    first = record(0, None, None)
    window_loss = 0.
    window_count = 0
    for k in range(stage["steps"]):
        sync(device)
        tick = time.perf_counter()
        loss, _, lr = update(model, optimizer, arrays["text_train"], c, stage, k, rng, device, False, order_hash)
        sync(device)
        train_seconds += time.perf_counter()-tick
        schedule_hash.update(np.array([lr], dtype="<f8").tobytes())
        window_loss += loss
        window_count += 1
        if (k+1) % c["eval_every"] == 0 or k+1 == stage["steps"]:
            last = record(k+1, window_loss/window_count, lr)
            window_loss = 0.
            window_count = 0
    # Final validation is never selected using a test set. Confirmation data are
    # neither downloaded nor evaluated unless enabled in the pre-run config.
    test = evaluate(model, arrays["text_test"], c, device, False) if c["text"]["evaluate_test"] else None
    checkpoint = {"schema_version": 4, "config": c, "stage": "language", "arm": arm, "seed": seed,
                  "vocab": model.vocab, "data_manifest_sha256": manifest_hash,
                  "model": model.state_dict(), "language_step": stage["steps"],
                  "validation": {"nll": last["validation_nll"], "targets": last["validation_targets"]},
                  "test": test}
    torch.save(checkpoint, folder/"language_final.pt")
    summary = {"arm": arm, "seed": seed, "purpose": c["purpose"],
               "language_updates": stage["steps"], "language_tokens": stage["steps"]*effective_tokens(c),
               "ppt_updates": c["ppt"]["steps"] if ppt else 0, "ppt_tokens": ppt["ppt_tokens"] if ppt else 0,
               "parameters": sum(p.numel() for p in model.parameters()),
               "core_initial_sha256": ppt["core_initial_sha256"] if ppt else start_core,
               "language_start_core_sha256": start_core, "language_head_initial_sha256": start_head,
               "language_optimizer_initial_state_entries": 0,
               "language_train_order_sha256": order_hash.hexdigest(), "language_schedule_sha256": schedule_hash.hexdigest(),
               "initial_validation_nll": first["validation_nll"], "final_validation_nll": last["validation_nll"],
               "language_train_seconds": train_seconds, "language_seconds": last["language_seconds"],
               "cumulative_seconds": last["cumulative_seconds"], "test": test,
               "checkpoint": "language_final.pt", "checkpoint_sha256": sha_file(folder/"language_final.pt")}
    dump(folder/"language_summary.json", summary)
    return summary
