"""Check stage-boundary controls and export compact evidence from a completed audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from transfer_v2.common import ARMS, dump, lr_at, sha_file, state_sha
from transfer_v2.model import language_model


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    audit = read(args.review / "audit.json")
    require(audit["cpu_checkpoints_verified"] == 15, "Full checkpoint audit required")
    run = args.review / "artifacts/runs/development"
    data = args.review / "artifacts/data/development"
    c = read(run / "environment.json")["config"]
    manifest = read(data / "manifest.json")
    train_length = len(np.load(data / "text_train.npy", allow_pickle=False, mmap_mode="r"))
    torch.set_num_threads(2)
    controls = []
    for seed in c["seeds"]:
        reference = read(run / f"scratch_{seed}/language_summary.json")
        rng = np.random.default_rng(seed + 202)
        order, schedule = hashlib.sha256(), hashlib.sha256()
        for step in range(c["language"]["steps"]):
            for _ in range(c["accumulate"]):
                indices = rng.integers(0, train_length - c["model"]["context"], size=c["batch"])
                order.update(indices.astype("<i8").tobytes())
            schedule.update(np.array([lr_at(step, c["language"])], dtype="<f8").tobytes())
        initial = language_model(c, manifest["text"]["vocab"], seed)
        for arm in ARMS:
            summary = read(run / f"{arm}_{seed}/language_summary.json")
            expected_core = initial.core_hash()
            if arm != "scratch":
                saved = torch.load(run / f"{arm}_{seed}/ppt_selected.pt", map_location="cpu", weights_only=True)
                core = {k[len("core."):]: v for k, v in saved["model"].items() if k.startswith("core.")}
                expected_core = state_sha(core)
            expected = {"language_start_core_sha256": expected_core,
                        "core_initial_sha256": initial.core_hash(),
                        "language_head_initial_sha256": initial.head_hash(),
                        "language_train_order_sha256": order.hexdigest(),
                        "language_schedule_sha256": schedule.hexdigest()}
            mismatches = {k: {"recorded": summary[k], "reconstructed": v}
                          for k, v in expected.items() if summary[k] != v}
            require(summary["language_train_order_sha256"] == order.hexdigest(), "Batch order differs")
            if arm != "scratch":
                require(summary["language_start_core_sha256"] == expected_core, "PPT transfer differs")
            else:
                require(summary["language_start_core_sha256"] == summary["core_initial_sha256"], "Scratch head start")
            for key in ("core_initial_sha256", "language_head_initial_sha256", "language_schedule_sha256"):
                require(summary[key] == reference[key], f"Unpaired recorded {key}")
            rows = [json.loads(s) for s in (run / f"{arm}_{seed}/language_metrics.jsonl").read_text().splitlines()]
            lr_error = max(abs(r["lr"] - lr_at(r["language_step"] - 1, c["language"])) for r in rows if r["lr"] is not None)
            require(lr_error < 1e-15, "Logged learning rates differ")
            controls.append({"arm": arm, "seed": seed, "strict_checks_passed": True,
                             "logged_lr_max_absolute_error": lr_error,
                             "cross_runtime_unmatched_hashes": mismatches})
    # A retained step-zero smoke backbone permits a numerical portability check.
    # Its metrics are never included in the development results.
    from transfer_v2.model import LM
    smoke = torch.load(args.review / "artifacts/runs/smoke/grid_nca_17/ppt_selected.pt", map_location="cpu", weights_only=True)
    require(smoke["selected_step"] == 0, "Expected retained step-zero portability fixture")
    local = LM(smoke["config"]["model"], smoke["vocab"], 17, 10017).core.state_dict()
    imported = {k[5:]: v for k, v in smoke["model"].items() if k.startswith("core.")}
    portability_error = max(float((local[k] - imported[k]).abs().max()) for k in local)
    dump(args.review / "stage_control_audit.json", {"strict_checks_passed": True, "records": controls,
         "local_torch": str(torch.__version__), "imported_torch": read(run / "environment.json")["torch"],
         "cross_runtime_bitwise_reproduction": not any(r["cross_runtime_unmatched_hashes"] for r in controls),
         "smoke_step_zero_core_max_absolute_error": portability_error,
         "scope": "Actual selected-PPT transfer hashes and reconstructed batch orders match. Within-run initialization/head/schedule hashes match across arms, and logged learning rates match numerically. Newer Windows runtime does not recreate all initialization/schedule bytes; do not claim independent bitwise regeneration of those values. Optimizer freshness is recorded by the trusted runner."})
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = {"archive_name": audit["archive_name"], "archive_sha256": audit["archive_sha256"],
                  "note": "GPU records copied unchanged from the returned development run; independent CPU files are derived checks. Smoke fixtures excluded. Weights and corpora remain private local evidence.", "files": {}}

    def copy(src, name, origin):
        destination = args.out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)
        provenance["files"][name] = {"sha256": sha_file(destination), "origin": origin}

    for name in ("audit.json", "metric_audit.json", "comparisons.csv", "cpu_verification.json", "stage_control_audit.json"):
        copy(args.review / name, name, f"Independent local CPU audit: {name}")
    for name in ("environment.json", "status.json", "report.json", "verification.json", "learning_curves.png", "ppt_diagnostics.png"):
        copy(run / name, f"gpu/{name}", f"runs/development/{name}")
    for path in sorted(run.glob("*/*")):
        if path.name.endswith(("_metrics.jsonl", "_summary.json")):
            relative = path.relative_to(run).as_posix()
            copy(path, f"gpu/{relative}", f"runs/development/{relative}")
    metrics = read(args.review / "metric_audit.json")
    aggregate = []
    for arm in ARMS:
        rows = [r for r in metrics["finals"] if r["arm"] == arm]
        mean = statistics.mean(r["nll"] for r in rows)
        comparisons = [r for r in metrics["comparisons"] if r["arm"] == arm]
        geometric = lambda key: math.exp(statistics.mean(math.log(r[key]) for r in comparisons))
        aggregate.append({"arm": arm, "mean_final_validation_nll": mean,
                          "perplexity_from_mean_nll": math.exp(mean),
                          "geomean_language_token_speedup": geometric("language_tokens_speedup") if comparisons else 1.,
                          "geomean_language_time_speedup": geometric("language_seconds_speedup") if comparisons else 1.,
                          "geomean_cumulative_time_speedup": geometric("cumulative_seconds_speedup") if comparisons else 1.})
    dump(args.out / "aggregate.json", aggregate)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for arm, color, label in (("grid_nca", "#2563eb", "Grid NCA"), ("shuffled_nca", "#db2777", "Shuffled NCA")):
        differences = []
        for seed in c["seeds"]:
            curves = [[json.loads(s) for s in (run / f"{a}_{seed}/language_metrics.jsonl").read_text().splitlines()]
                      for a in (arm, "scratch")]
            steps = np.array([r["language_step"] for r in curves[0]])
            delta = np.array([r["validation_nll"] - b["validation_nll"] for r, b in zip(*curves)])
            keep = steps >= 500
            differences.append(delta[keep])
            ax.plot(steps[keep], delta[keep], color=color, alpha=.25, linewidth=1)
        ax.plot(steps[keep], np.mean(differences, axis=0), color=color, linewidth=2.5, label=label)
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1, label="No PPT control")
    ax.set(xlabel="Identical language-stage updates", ylabel="Validation NLL minus paired no-PPT NLL",
           title="Both synthetic initializations improve language learning\nThin lines: paired seeds; thick lines: mean; lower is better")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(args.out / "paired_language_gain.png", dpi=180)
    plt.close(fig)
    dump(args.out / "PROVENANCE.json", provenance)
    print(json.dumps({"stage_controls_passed": len(controls), "aggregate": aggregate,
                      "archive_sha256": audit["archive_sha256"],
                      "cpu_max_error": audit["max_cpu_evaluation_absolute_error"]}, indent=2))


if __name__ == "__main__":
    main()
