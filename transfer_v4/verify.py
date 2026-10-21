from __future__ import annotations

import json
from pathlib import Path

import torch
import numpy as np
import hashlib

from .common import ARMS, DATA_KEYS, dump, effective_tokens, sha_file, state_sha, source_hashes
from .graph import validate_topology
from .data import load_data
from .model import LM
from .training import evaluate
from .controls import validate_prefix_control
from .plan import verify_plan
from .analysis import common_target_analysis


def verify(root, data_root, device="cpu"):
    root = Path(root)
    env = json.loads((root/"environment.json").read_text())
    c = env["config"]
    verify_plan(root, c)
    if source_hashes() != env["source_sha256"]:
        raise ValueError("Verification source differs from the recorded runner")
    datasets, _ = load_data(data_root, c)
    for seed in c["seeds"]:
        arrays = datasets[str(seed)]
        validate_topology(arrays["graph_edges"], c["nca"]["grid"])
        rule_sets = []
        for split in ("train", "validation"):
            for original, control in (("nca", "prefix_grid"), ("graph", "prefix_graph")):
                validate_prefix_control(arrays[f"{original}_{split}"], arrays[f"{control}_{split}"], c)
            with np.load(Path(data_root)/f"replicate_{seed}/rules_{split}.npz", allow_pickle=False) as rules:
                rule_sets.append({hashlib.sha256(b"".join(rules[k][i].tobytes() for k in sorted(rules.files))).hexdigest()
                                  for i in range(len(rules[rules.files[0]]))})
            for original, shuffled in (("nca", "shuffled"), ("graph", "shuffled_graph")):
                a, b = arrays[f"{original}_{split}"], arrays[f"{shuffled}_{split}"]
                if not np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1)):
                    raise ValueError("Shuffled trajectory histogram changed")
                markers = a >= c["nca"]["states"]**4
                if not np.array_equal(markers, b >= c["nca"]["states"]**4) or not np.array_equal(a[markers], b[markers]):
                    raise ValueError("Shuffled markers changed")
        if rule_sets[0] & rule_sets[1]:
            raise ValueError("Actual train/validation rule weights overlap")
    manifest_hash = sha_file(Path(data_root)/"manifest.json")
    if manifest_hash != env["data_manifest_sha256"]:
        raise ValueError("Data manifest changed")
    records, language = [], {}
    selected_count = 0
    extra_final_count = 0
    initializations = {}
    for seed in c["seeds"]:
        entry = env["initializations"][str(seed)]
        path = root/entry["file"]
        if sha_file(path) != entry["sha256"]:
            raise ValueError("Saved initialization checksum mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["config"] != c or saved["seed"] != seed:
            raise ValueError("Saved initialization identity mismatch")
        if state_sha(saved["core"]) != entry["core_sha256"] or state_sha(saved["embedding"]) != entry["head_sha256"]:
            raise ValueError("Saved initialization state mismatch")
        initializations[seed] = entry
    for arm in ARMS:
        for seed in c["seeds"]:
            arrays = datasets[str(seed)]
            folder = root/f"{arm}_{seed}"
            for stage in ("ppt", "language"):
                path = folder/f"{stage}_summary.json"
                if not path.exists():
                    continue
                summary = json.loads(path.read_text())
                checkpoint = folder/summary["checkpoint"]
                if sha_file(checkpoint) != summary["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint changed: {checkpoint.name}")
                saved = torch.load(checkpoint, map_location=device, weights_only=True)
                if saved["config"] != c or saved["arm"] != arm or saved["seed"] != seed or saved["stage"] != stage:
                    raise ValueError("Checkpoint metadata mismatch")
                if saved["data_manifest_sha256"] != manifest_hash:
                    raise ValueError("Checkpoint/data mismatch")
                model = LM(c["model"], saved["vocab"], core_seed=seed, head_seed=seed)
                model.load_state_dict(saved["model"], strict=True)
                model.to(device)
                if stage == "ppt":
                    key = DATA_KEYS[arm]+"_validation"
                    if saved["core_initial_sha256"] != initializations[seed]["core_sha256"]:
                        raise ValueError("PPT initial backbone differs from saved common initialization")
                    measured = evaluate(model, arrays[key], c, device, True)
                    errors = {k: abs(measured[k]-saved["validation"][k]) for k in ("nll", "patch_nll", "patch_accuracy")}
                else:
                    if summary["language_updates"] != c["language"]["steps"] or summary["language_tokens"] != c["language"]["steps"]*effective_tokens(c):
                        raise ValueError("Language budgets differ")
                    measured = evaluate(model, arrays["text_validation"], c, device, False)
                    errors = {"validation_nll": abs(measured["nll"]-saved["validation"]["nll"])}
                    if saved["test"] is not None:
                        test = evaluate(model, arrays["text_test"], c, device, False)
                        errors["test_nll"] = abs(test["nll"]-saved["test"]["nll"])
                    language[arm, seed] = summary
                if any(v > 1e-4 for v in errors.values()):
                    raise ValueError(f"Reloaded loss differs: {arm}, {seed}, {stage}, {errors}")
                records.append({"arm": arm, "seed": seed, "stage": stage,
                                "checksum_valid": True, "absolute_errors": errors})
                selected_count += 1
                print(f"Verified {stage} {arm} seed={seed}: max error {max(errors.values()):.3g}", flush=True)
                if stage == "ppt" and (folder/"ppt_final.pt").exists():
                    if sha_file(folder/"ppt_final.pt") != summary["final_checkpoint_sha256"]:
                        raise ValueError("Final PPT checkpoint checksum mismatch")
                    final = torch.load(folder/"ppt_final.pt", map_location=device, weights_only=True)
                    if final["config"] != c or final["arm"] != arm or final["seed"] != seed or final["data_manifest_sha256"] != manifest_hash:
                        raise ValueError("Final PPT checkpoint identity mismatch")
                    model.load_state_dict(final["model"], strict=True)
                    measured = evaluate(model, arrays[key], c, device, True)
                    final_errors = {k: abs(measured[k]-final["validation"][k]) for k in ("nll", "patch_nll", "patch_accuracy")}
                    if max(final_errors.values()) > 1e-4:
                        raise ValueError("Final PPT checkpoint loss mismatch")
                    records.append({"arm": arm, "seed": seed, "stage": "ppt_final",
                                    "absolute_errors": final_errors, "sha256": sha_file(folder/"ppt_final.pt")})
                    extra_final_count += 1
    for seed in c["seeds"]:
        summaries = [language[a, seed] for a in ARMS if (a, seed) in language]
        if summaries and len(summaries) != len(ARMS):
            raise ValueError("Incomplete language arm set")
        for key in ("core_initial_sha256", "language_head_initial_sha256", "language_train_order_sha256",
                    "language_schedule_sha256", "language_tokens", "language_updates"):
            if len({s[key] for s in summaries}) > 1:
                raise ValueError(f"Unmatched language control: {key}, seed {seed}")
        if any(s["language_optimizer_initial_state_entries"] != 0 for s in summaries):
            raise ValueError("Language optimizer was not fresh")
        for s in summaries:
            entry = initializations[seed]
            if s["core_initial_sha256"] != entry["core_sha256"] or s["language_head_initial_sha256"] != entry["head_sha256"]:
                raise ValueError("Language common initialization differs from saved tensors")
            if s["arm"] == "scratch":
                expected_core = entry["core_sha256"]
            else:
                ppt = torch.load(root/f"{s['arm']}_{seed}/ppt_selected.pt", map_location="cpu", weights_only=True)
                expected_core = state_sha({k[5:]: v for k, v in ppt["model"].items() if k.startswith("core.")})
            if expected_core != s["language_start_core_sha256"]:
                raise ValueError("Language did not start from the declared PPT core")
    status = json.loads((root/"status.json").read_text())
    expected = len(c["seeds"])*(13 if status["state"] == "complete" else 6)
    if selected_count != expected:
        raise ValueError(f"Expected {expected} selected-stage checkpoints; verified {selected_count}")
    curves = {(arm, seed): [json.loads(line) for line in (root/f"{arm}_{seed}/language_metrics.jsonl").read_text().splitlines()]
              for arm, seed in language}
    steps = [0]+list(range(c["eval_every"], c["language"]["steps"]+1, c["eval_every"]))
    if steps[-1] != c["language"]["steps"]:
        steps.append(c["language"]["steps"])
    for (arm, seed), rows in curves.items():
        if [r["language_step"] for r in rows] != steps:
            raise ValueError("Raw language curve has missing, repeated or extra evaluations")
        if any(r["arm"] != arm or r["seed"] != seed or r["language_tokens"] != r["language_step"]*effective_tokens(c) for r in rows):
            raise ValueError("Raw language curve identity/exposure mismatch")
        if abs(rows[-1]["validation_nll"]-language[arm, seed]["final_validation_nll"]) > 1e-4:
            raise ValueError("Raw curve endpoint differs from the verified checkpoint summary")
    if json.loads((root/"report.json").read_text())["primary_analysis"] != common_target_analysis(curves, c):
        raise ValueError("Primary analysis differs from raw recorded language curves")
    result = {"device": device, "status": status, "stage_checkpoints_verified": len(records),
              "selected_stage_checkpoints_verified": selected_count,
              "additional_final_ppt_checkpoints_verified": extra_final_count,
              "saved_initializations_verified": len(initializations),
              "records": records, "language_controls_checked": bool(language),
              "note": "Checkpoint consistency does not prove learning acceleration"}
    dump(root/"verification.json", result)
    return result
