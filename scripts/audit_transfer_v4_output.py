"""Audit a returned v4 ZIP with trusted local sources and independent CPU replay.

Imported code is read as data, never executed. Original ZIP and extracted GPU
verification remain unchanged. Use fresh output directories and run_logged.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import shutil
import zipfile
from pathlib import PurePosixPath
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from audit_transfer_v2_output import read, require
from transfer_v4.common import ARMS, DATA_KEYS, dump, effective_tokens, lr_at, sha_file, source_hashes
from transfer_v4.data import load_data
from transfer_v4.graph import topology_hash, validate_topology
from transfer_v4.nca import control_probabilities, evaluate_controls, lattice
from transfer_v4.verify import verify
from transfer_v4.controls import prefix_shuffled, validate_prefix_control
from transfer_v4.nca import shuffled
from transfer_v4.plan import verify_plan
from transfer_v4.common import STUDY


def curve(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values and all(v is not None and v > 0 for v in values) else None


def extract(archive, target):
    """V4 has 78 stage checkpoints; retain bounded, path-safe streaming extraction."""
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        require(len(infos) < 10000 and len(infos) == len({i.filename.casefold() for i in infos}), "Duplicate/excessive ZIP members")
        require(sum(i.file_size for i in infos) < 6_000_000_000, "Unexpected expanded size")
        for entry in infos:
            require(entry.orig_filename == entry.filename, "ZIP filename normalized or truncated by the runtime")
            path = PurePosixPath(entry.filename)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in entry.filename and ":" not in entry.filename, "Unsafe ZIP path")
            require((entry.external_attr >> 16) & 0o170000 != 0o120000, "ZIP symlink")
            destination = (target/Path(*path.parts)).resolve()
            require(destination.is_relative_to(target.resolve()), "ZIP path escapes destination")
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(entry) as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024*1024)
        return len(infos)


def independent_primary_analysis(run, c):
    """Recompute crossings and exact sign probabilities without runner analysis code."""
    groups = []
    pairs = list(STUDY["primary_pairs"])+[("prefix_grid", "grid_nca"), ("shuffled_graph", "graph_nca"),
                                         ("shuffled_nca", "grid_nca"), ("scratch", "graph_nca")]
    for reference_arm, arm in pairs:
        rows = []
        for seed in c["seeds"]:
            scratch = curve(run/f"scratch_{seed}/language_metrics.jsonl")
            threshold = scratch[-1]["validation_nll"]
            ref = next((r for r in curve(run/f"{reference_arm}_{seed}/language_metrics.jsonl") if r["validation_nll"] <= threshold), None)
            hit = next((r for r in curve(run/f"{arm}_{seed}/language_metrics.jsonl") if r["validation_nll"] <= threshold), None)
            row = {"seed": seed, "target_nll": threshold, "reference_reached": ref is not None,
                   "graph_reached": hit is not None, "reference_step": ref["language_step"] if ref else None,
                   "graph_step": hit["language_step"] if hit else None}
            for key in ("language_tokens", "language_seconds", "cumulative_seconds"):
                row[key+"_speedup"] = ref[key]/hit[key] if ref and hit and ref[key] > 0 and hit[key] > 0 else None
            rows.append(row)
        item = {"reference_arm": reference_arm, "arm": arm, "rows": rows,
                "target_policy": STUDY["target"], "declared_cohorts": len(c["seeds"])}
        for key in ("language_tokens", "language_seconds", "cumulative_seconds"):
            item["geomean_"+key+"_speedup_all_cohorts"] = geometric([r[key+"_speedup"] for r in rows])
        values = [r["language_tokens_speedup"] for r in rows]
        wins = sum(v is not None and v > 1 for v in values)
        p = sum(math.comb(len(rows), k) for k in range(wins, len(rows)+1))/2**len(rows) if all(v is not None for v in values) else None
        item.update(strict_token_wins=wins, ties_count_as_nonwins=True, one_sided_sign_p=p,
                    alpha=STUDY["alpha_per_comparison"],
                    criterion_met=bool(p is not None and len(rows) >= STUDY["minimum_cohorts"] and p <= STUDY["alpha_per_comparison"] and item["geomean_language_tokens_speedup_all_cohorts"] > 1))
        groups.append(item)
    for item in groups[2:]:
        for key in ("one_sided_sign_p", "alpha", "criterion_met"):
            item.pop(key)
        item["scope"] = "Descriptive secondary comparison; no additional significance claim"
    return {"study": STUDY, "comparisons": groups[:2], "secondary_common_target_comparisons": groups[2:],
            "eligible_for_research_interpretation": c["purpose"] == "development",
            "supports_graph_advantage_on_declared_target": c["purpose"] == "development" and all(g["criterion_met"] for g in groups[:2]),
            "multiplicity": "Two co-primary one-sided sign tests, alpha .025 each (Bonferroni family bound .05)",
            "scope": "Language-token transfer advantage on an inspected validation task; not end-to-end GPU speed or untouched confirmation",
            "censoring": "Any missing crossing withholds the all-cohort ratio and significance claim; no imputation or success-only subset"}


def comparison(curves, arm, reference_arm, seed, keys):
    reference = curves[reference_arm][-1]["validation_nll"]
    base = next(r for r in curves[reference_arm] if r["validation_nll"] <= reference)
    hit = next((r for r in curves[arm] if r["validation_nll"] <= reference), None)
    row = {"arm": arm, "seed": seed, "reference_nll": reference,
           "reference_reached": hit is not None,
           "language_step_at_reference": hit["language_step"] if hit else None,
           "final_validation_nll_delta": curves[arm][-1]["validation_nll"] - reference}
    for key in keys:
        row[key+"_speedup"] = base[key]/hit[key] if hit and hit[key] > 0 else None
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    archive_hash = sha_file(a.archive)
    count = extract(a.archive, a.out/"artifacts")
    run, data = a.out/"artifacts/runs/development", a.out/"artifacts/data/development"
    env = read(run/"environment.json")
    c, hashes = env["config"], source_hashes()
    require(c == read(ROOT/"configs/transfer_v4.json") == read(a.out/"artifacts/configs/transfer_v4.json"), "Development configuration differs")
    require(hashes == env["source_sha256"], "Recorded runner differs from trusted local sources")
    for name, expected in hashes.items():
        require(sha_file(a.out/"artifacts/transfer_v4"/name) == expected, f"Source mismatch: {name}")
    verify_plan(run, c)
    require(sha_file(run/"FROZEN_PROTOCOL.md") == sha_file(ROOT/"docs/TRANSFER_V4_PROTOCOL.md"), "Archived protocol differs from prospective canonical protocol")
    arrays, manifest = load_data(data, c)
    require(manifest["generation"]["source_sha256"] == hashes, "Generation source differs")
    require(sha_file(data/"manifest.json") == env["data_manifest_sha256"], "Data manifest differs")
    require(not c["text"]["evaluate_test"] and not any("text_test" in arr for arr in arrays.values()), "Official test unexpectedly loaded")
    require(read(run/"status.json")["state"] == "complete", "This result auditor expects a completed language suite")
    records, finals, comparisons, treatment_rows, cohorts = [], [], [], [], []
    report = read(run/"report.json")
    language_keys = ("language_tokens", "language_train_seconds", "language_seconds", "total_training_tokens", "cumulative_seconds")
    pair_keys = ("language_tokens", "language_seconds", "cumulative_seconds")
    for seed in c["seeds"]:
        arr, rep = arrays[str(seed)], manifest["replicates"][str(seed)]
        validate_topology(arr["graph_edges"], c["nca"]["grid"])
        require(topology_hash(arr["graph_edges"]) == rep["graph"]["sha256"], "Topology identity differs")
        slots = [0,1,2,3,5,6,7,8]
        changed = float(np.mean(arr["graph_edges"][:, slots] != lattice(c["nca"]["grid"])[:, slots]))
        require(changed == rep["graph"]["actual_changed_fraction"] and changed >= c["graph"]["rewire_fraction"], "Rewiring treatment differs")
        cohorts.append({"seed": seed, "topology_sha256": rep["graph"]["sha256"], "changed_fraction": changed,
                        "train_rules": rep["nca"]["train"]["accepted"], "validation_rules": rep["nca"]["validation"]["accepted"]})
        for split in ("train", "validation"):
            require(not (set(rep["nca"]["train"]["rule_ids"]) & set(rep["nca"]["validation"]["rule_ids"])), "Rule IDs overlap")
            for kind, old, new in (("nca", "shuffled", "prefix_grid"), ("graph", "shuffled_graph", "prefix_graph")):
                parent = arr[f"{kind}_{split}"]
                old_seed = c["nca"]["data_seed"]+100003*seed+(10 if split == "train" else 20)
                new_seed = c["nca"]["data_seed"]+100003*seed+(30 if split == "train" else 40)
                require(np.array_equal(shuffled(parent, c, old_seed), arr[f"{old}_{split}"]), "Original shuffled control cannot be regenerated")
                require(np.array_equal(prefix_shuffled(parent, c, new_seed), arr[f"{new}_{split}"]), "Prefix control cannot be regenerated")
                validate_prefix_control(parent, arr[f"{new}_{split}"], c)
            for kind in ("nca", "graph"):
                scores = np.array(rep["nca"][split][kind]["profile_scores"])
                require(len(scores) == rep["nca"][split]["accepted"] and np.all((scores > c["nca"]["filter_low"]) & (scores < c["nca"]["filter_high"])), "Joint complexity eligibility differs")
        train_length = len(arr["text_train"])
        order, schedule = hashlib.sha256(), hashlib.sha256()
        rng = np.random.default_rng(seed+202)
        for step in range(c["language"]["steps"]):
            for _ in range(c["accumulate"]):
                order.update(rng.integers(0, train_length-c["model"]["context"], size=c["batch"]).astype("<i8").tobytes())
            schedule.update(np.array([lr_at(step, c["language"])], dtype="<f8").tobytes())
        ppt_order = hashlib.sha256()
        rng = np.random.default_rng(seed+101)
        for _ in range(c["ppt"]["steps"]*c["accumulate"]):
            ppt_order.update(rng.integers(0, len(arr["nca_train"]), size=c["batch"]).astype("<i8").tobytes())
        curves = {arm: curve(run/f"{arm}_{seed}/language_metrics.jsonl") for arm in ARMS}
        for arm in ARMS:
            rows = curves[arm]
            summary = read(run/f"{arm}_{seed}/language_summary.json")
            require([r["language_step"] for r in rows] == [0]+list(range(c["eval_every"], c["language"]["steps"]+1, c["eval_every"])), "Incomplete language curve")
            require(all(r["language_tokens"] == r["language_step"]*effective_tokens(c) for r in rows), "Token accounting differs")
            require(summary["final_validation_nll"] == rows[-1]["validation_nll"] and summary["initial_validation_nll"] == rows[0]["validation_nll"], "Curve/summary differs")
            require(summary["test"] is None and summary["language_train_order_sha256"] == order.hexdigest(), "Test policy or language order differs")
            lr_error = max(abs(r["lr"]-lr_at(r["language_step"]-1, c["language"])) for r in rows if r["lr"] is not None)
            require(lr_error < 1e-15, "Language learning-rate values differ")
            record = {"seed": seed, "arm": arm, "batch_order_reconstructed": True,
                      "lr_max_absolute_error": lr_error,
                      "full_schedule_hash_reproduced_across_runtimes": summary["language_schedule_sha256"] == schedule.hexdigest()}
            finals.append({"arm": arm, "seed": seed, "nll": rows[-1]["validation_nll"],
                           "perplexity": math.exp(rows[-1]["validation_nll"])})
            if arm != "scratch":
                kind = DATA_KEYS[arm]
                ppt = read(run/f"{arm}_{seed}/ppt_summary.json")
                require(ppt["ppt_updates"] == c["ppt"]["steps"] and ppt["ppt_tokens"] == c["ppt"]["steps"]*effective_tokens(c), "PPT budget differs")
                require(ppt["train_order_sha256"] == ppt_order.hexdigest(), "PPT rule order differs")
                probabilities, mixture = control_probabilities(arr[kind+"_train"], c)
                controls = evaluate_controls(arr[kind+"_validation"][:c["ppt_eval_rules"]], c, probabilities, mixture)
                require(all(abs(controls[k]-ppt["controls"][k]) < 1e-10 for k in controls), "Synthetic baseline differs")
                selected = min(curve(run/f"{arm}_{seed}/ppt_metrics.jsonl"), key=lambda r:r["validation"]["patch_nll"])
                require(selected["ppt_step"] == ppt["selected_step"], "PPT checkpoint selection differs")
                relative = 1-ppt["selected"]["patch_nll"]/ppt["initial"]["patch_nll"]
                margin = min(controls["unigram_patch_nll"], controls["copy_mixture_patch_nll"])-ppt["selected"]["patch_nll"]
                passed = relative >= c["gate"]["minimum_relative_improvement"] and margin >= c["gate"]["baseline_margin_nats"]
                require(passed == ppt["gate"]["passed"], "Synthetic gate differs")
                record["synthetic"] = {"patch_nll": ppt["selected"]["patch_nll"], "relative_improvement": relative,
                                       "control_margin": margin, "passed": passed, "selected_step": ppt["selected_step"]}
                calculated = comparison(curves, arm, "scratch", seed, language_keys)
                imported = next(r for r in report["comparisons"] if r["arm"] == arm and r["seed"] == seed)
                require(all(calculated[k] == imported[k] for k in calculated), "Scratch-reference comparison differs")
                comparisons.append(calculated)
            records.append(record)
        for reference_arm, arm in (("grid_nca","graph_nca"), ("shuffled_graph","graph_nca"), ("shuffled_nca","grid_nca")):
            calculated = comparison(curves, arm, reference_arm, seed, pair_keys)
            group = next(r for r in report["treatment_comparisons"] if r["arm"] == arm and r["reference_arm"] == reference_arm)
            imported = next(r for r in group["comparisons"] if r["seed"] == seed)
            require(all(calculated[k] == imported[k] for k in calculated), "Treatment reference comparison differs")
            treatment_rows.append(dict(calculated, reference_arm=reference_arm))
    require(len({r["topology_sha256"] for r in cohorts}) == len(c["seeds"]), "Topology replicates repeated")
    for group in report["treatment_comparisons"]:
        rows = [r for r in treatment_rows if (r["arm"],r["reference_arm"]) == (group["arm"],group["reference_arm"])]
        for key in pair_keys:
            require(geometric([r[key+"_speedup"] for r in rows]) == group["geomean_"+key+"_speedup_all_seeds"], "Censoring/aggregate differs")
    independent_primary = independent_primary_analysis(run, c)
    require(independent_primary == report["primary_analysis"], "Independent common-target/sign-test analysis differs")
    dump(a.out/"metric_audit.json", {"primary_analysis": independent_primary, "finals": finals, "comparisons": comparisons,
                                    "treatment_comparisons": treatment_rows, "controls": records, "cohorts": cohorts})
    print("Archive CRC/sources/data, paired topology/filter treatment, reconstructed batches, fitted controls, gates and raw comparisons passed.", flush=True)
    verification_hash = sha_file(run/"verification.json")
    torch.set_num_threads(2)
    def replay_evaluation(model, array, config, device, synthetic):
        result = trusted_evaluate(model, array, config, device, synthetic)
        replay_progress.append({"stage": "ppt" if synthetic else "language", "nll": result["nll"]})
        dump(a.out/"cpu_progress.json", {"completed_evaluations": len(replay_progress), "records": replay_progress})
        return result
    replay_progress = []
    from transfer_v4.training import evaluate as trusted_evaluate
    with patch("transfer_v4.verify.evaluate", side_effect=replay_evaluation), patch("transfer_v4.verify.dump", side_effect=lambda path,value: dump(a.out/"cpu_verification.json",value)):
        verification = verify(run, data, "cpu")
    require(sha_file(run/"verification.json") == verification_hash, "Imported GPU verification modified")
    result = {"archive_name": a.archive.name, "archive_sha256": archive_hash,
              "members_crc_checked": count, "trusted_source_sha256": hashes,
              "data_manifest_sha256": env["data_manifest_sha256"], "gpu": env["gpu"],
              "imported_gpu_verification_sha256": verification_hash,
              "cpu_checkpoints_verified": verification["selected_stage_checkpoints_verified"],
              "cpu_initializations_verified": verification["saved_initializations_verified"],
              "max_cpu_evaluation_absolute_error": max(v for r in verification["records"] for v in r["absolute_errors"].values()),
              "language_data_sha256": {k: manifest["files"][k] for k in ("text_train.npy","text_validation.npy")},
              "test_split_present": False, "duration_seconds": time.perf_counter()-started}
    dump(a.out/"audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
