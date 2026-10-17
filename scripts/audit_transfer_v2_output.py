"""Audit a returned v2 archive with trusted local code; never alter imported evidence.

Run through scripts/run_logged.py. Extracted Python is compared as data, never
imported. CPU re-evaluation writes outside the imported run directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from transfer_v2.common import ARMS, dump, effective_tokens, sha_file, source_hashes
from transfer_v2.data import load_data
from transfer_v2.nca import control_probabilities, evaluate_controls
from transfer_v2.verify import verify


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract(archive, target):
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        require(len(infos) == len({i.filename.casefold() for i in infos}), "Duplicate ZIP paths")
        require(sum(i.file_size for i in infos) < 2_000_000_000, "Unexpected expanded size")
        for i in infos:
            p = PurePosixPath(i.filename)
            require(not p.is_absolute() and ".." not in p.parts
                    and "\\" not in i.filename and ":" not in i.filename, "Unsafe ZIP path")
            require((i.external_attr >> 16) & 0o170000 != 0o120000, "ZIP symlink")
            destination = (target / Path(*p.parts)).resolve()
            require(destination.is_relative_to(target.resolve()), "ZIP path escapes destination")
            if i.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Streaming to EOF also validates each member's CRC.
                with z.open(i) as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
        return len(infos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    artifacts = args.out / "artifacts"
    started = time.perf_counter()
    archive_hash = sha_file(args.archive)
    count = extract(args.archive, artifacts)
    run, data = artifacts / "runs/development", artifacts / "data/development"
    env = read(run / "environment.json")
    c = env["config"]
    require(c == read(ROOT / "configs/transfer_v2.json") == read(artifacts / "configs/transfer_v2.json"),
            "Returned configuration differs from declared development configuration")
    hashes = source_hashes()
    require(hashes == env["source_sha256"], "Recorded runner source differs from trusted local code")
    for name, expected in hashes.items():
        require(sha_file(artifacts / "transfer_v2" / name) == expected, f"Source mismatch: {name}")
    arrays, manifest = load_data(data, c)
    require(manifest["generation"]["source_sha256"] == hashes, "Generation source differs")
    require(sha_file(data / "manifest.json") == env["data_manifest_sha256"], "Manifest mismatch")
    require(not c["text"]["evaluate_test"] and "text_test" not in arrays, "Confirmation split used")
    train_ids = set(manifest["nca"]["train"]["rule_ids"])
    val_ids = set(manifest["nca"]["validation"]["rule_ids"])
    require(not train_ids.intersection(val_ids), "Rule IDs overlap")
    # Identity of actual rule parameters, independent of human-readable split IDs.
    import hashlib
    rule_hashes = []
    for split in ("train", "validation"):
        with np.load(data / f"rules_{split}.npz", allow_pickle=False) as rules:
            rule_hashes.append({hashlib.sha256(b"".join(rules[k][i].tobytes()
                               for k in sorted(rules.files))).hexdigest()
                                for i in range(len(rules[rules.files[0]]))})
    require(not rule_hashes[0].intersection(rule_hashes[1]), "Rule weights overlap")
    for split in ("train", "validation"):
        original, shuffled = arrays[f"nca_{split}"], arrays[f"shuffled_{split}"]
        require(np.array_equal(np.sort(original, axis=1), np.sort(shuffled, axis=1)),
                "Shuffled trajectory histogram changed")
        cutoff = c["nca"]["states"] ** 4
        markers = original >= cutoff
        require(np.array_equal(original[markers], shuffled[markers])
                and np.array_equal(markers, shuffled >= cutoff), "Shuffled markers changed")
    ppt = []
    for arm, prefix in (("grid_nca", "nca"), ("shuffled_nca", "shuffled")):
        probabilities, mixture = control_probabilities(arrays[f"{prefix}_train"], c)
        controls = evaluate_controls(arrays[f"{prefix}_validation"][:c["ppt_eval_rules"]],
                                     c, probabilities, mixture)
        for seed in c["seeds"]:
            summary = read(run / f"{arm}_{seed}/ppt_summary.json")
            require(all(abs(controls[k] - summary["controls"][k]) < 1e-10 for k in controls),
                    "Synthetic baseline recomputation differs")
            curve = [json.loads(s) for s in (run / f"{arm}_{seed}/ppt_metrics.jsonl").read_text().splitlines()]
            selected = min(curve, key=lambda r: r["validation"]["patch_nll"])
            require(selected["ppt_step"] == summary["selected_step"], "PPT selection differs")
            initial, final = summary["initial"]["patch_nll"], summary["selected"]["patch_nll"]
            relative = (initial - final) / initial
            margin = min(controls["unigram_patch_nll"], controls["copy_mixture_patch_nll"]) - final
            passed = relative >= c["gate"]["minimum_relative_improvement"] and margin >= c["gate"]["baseline_margin_nats"]
            require(passed == summary["gate"]["passed"], "Gate recomputation differs")
            ppt.append({"arm": arm, "seed": seed, "selected_step": summary["selected_step"],
                        "patch_nll": final, "relative_improvement": relative,
                        "margin_over_control": margin, "passed": passed})
    comparisons, finals = [], []
    saved_report = read(run / "report.json")
    for seed in c["seeds"]:
        curves = {a: [json.loads(s) for s in (run / f"{a}_{seed}/language_metrics.jsonl").read_text().splitlines()]
                  for a in ARMS}
        reference = curves["scratch"][-1]["validation_nll"]
        base = next(r for r in curves["scratch"] if r["validation_nll"] <= reference)
        for arm in ARMS:
            curve = curves[arm]
            require(curve[0]["language_step"] == 0 and curve[-1]["language_step"] == c["language"]["steps"],
                    "Incomplete language curve")
            require(all(r["language_tokens"] == r["language_step"] * effective_tokens(c) for r in curve),
                    "Language token accounting differs")
            summary = read(run / f"{arm}_{seed}/language_summary.json")
            require(summary["final_validation_nll"] == curve[-1]["validation_nll"], "Final curve/summary differ")
            require(summary["test"] is None, "Unexpected confirmation evaluation")
            finals.append({"arm": arm, "seed": seed, "nll": curve[-1]["validation_nll"],
                           "perplexity": math.exp(curve[-1]["validation_nll"])})
            if arm == "scratch":
                continue
            hit = next((r for r in curve if r["validation_nll"] <= reference), None)
            row = {"arm": arm, "seed": seed, "reference_nll": reference,
                   "language_step_at_reference": hit["language_step"] if hit else None,
                   "final_validation_nll_delta": curve[-1]["validation_nll"] - reference}
            for key in ("language_tokens", "language_train_seconds", "language_seconds", "total_training_tokens", "cumulative_seconds"):
                row[f"{key}_speedup"] = base[key] / hit[key] if hit and hit[key] > 0 else None
            reported = next(r for r in saved_report["comparisons"] if r["arm"] == arm and r["seed"] == seed)
            require(all(row[k] == reported[k] for k in row), "Raw curves/report differ")
            comparisons.append(row)
    dump(args.out / "metric_audit.json", {"comparisons": comparisons, "finals": finals, "ppt": ppt})
    with (args.out / "comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"Archive CRC, {count} safe paths, sources, corpus hashes, actual rule disjointness, shuffled controls and raw metrics passed.", flush=True)
    # verify() normally writes verification.json beside the run. Redirect that
    # one output so the original GPU verification remains byte-for-byte intact.
    imported_verify_hash = sha_file(run / "verification.json")
    torch.set_num_threads(2)
    with patch("transfer_v2.verify.dump", side_effect=lambda path, result: dump(args.out / "cpu_verification.json", result)):
        verification = verify(run, data, "cpu")
    require(sha_file(run / "verification.json") == imported_verify_hash, "Imported verification modified")
    result = {"archive_name": args.archive.name, "archive_sha256": archive_hash,
              "zip_members_crc_checked": count, "trusted_source_sha256": hashes,
              "data_manifest_sha256": env["data_manifest_sha256"],
              "imported_gpu_verification_sha256": imported_verify_hash,
              "status": read(run / "status.json"), "gpu": env.get("gpu"),
              "cpu_checkpoints_verified": verification["stage_checkpoints_verified"],
              "max_cpu_evaluation_absolute_error": max(v for r in verification["records"] for v in r["absolute_errors"].values()),
              "controls_and_metrics_recomputed": True, "test_split_present": False,
              "duration_seconds": time.perf_counter() - started}
    dump(args.out / "audit.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
