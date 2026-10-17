from __future__ import annotations

import json
from pathlib import Path

import torch

from .common import ARMS, dump, effective_tokens, sha_file
from .data import load_data
from .model import LM
from .training import evaluate


def verify(root, data_root, device="cpu"):
    root = Path(root)
    env = json.loads((root/"environment.json").read_text())
    c = env["config"]
    arrays, _ = load_data(data_root, c)
    manifest_hash = sha_file(Path(data_root)/"manifest.json")
    if manifest_hash != env["data_manifest_sha256"]:
        raise ValueError("Data manifest changed")
    records, language = [], {}
    selected_count = 0
    extra_final_count = 0
    for arm in ARMS:
        for seed in c["seeds"]:
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
                    key = "nca_validation" if arm == "grid_nca" else "shuffled_validation"
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
    status = json.loads((root/"status.json").read_text())
    expected = len(c["seeds"])*(5 if status["state"] == "complete" else 2)
    if selected_count != expected:
        raise ValueError(f"Expected {expected} selected-stage checkpoints; verified {selected_count}")
    result = {"device": device, "status": status, "stage_checkpoints_verified": len(records),
              "selected_stage_checkpoints_verified": selected_count,
              "additional_final_ppt_checkpoints_verified": extra_final_count,
              "records": records, "language_controls_checked": bool(language),
              "note": "Checkpoint consistency does not prove learning acceleration"}
    dump(root/"verification.json", result)
    return result
