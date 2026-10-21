"""Export compact v3 GPU/CPU evidence and paired learning comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from audit_transfer_v3_output import read, require, geometric
from transfer_v3.common import ARMS, dump, sha_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    audit = read(a.review/"audit.json")
    require(audit["cpu_checkpoints_verified"] == 27 and audit["cpu_initializations_verified"] == 3,
            "Completed independent v3 checkpoint/initialization audit required")
    run = a.review/"artifacts/runs/development"
    data = a.review/"artifacts/data/development"
    metrics = read(a.review/"metric_audit.json")
    manifest, env = read(data/"manifest.json"), read(run/"environment.json")
    c = env["config"]
    a.out.mkdir(parents=True, exist_ok=False)
    provenance = {"archive_name": audit["archive_name"], "archive_sha256": audit["archive_sha256"],
                  "note": "Development GPU records copied unchanged; independent CPU audits and derived plots are separate. Execution fixtures, weights and corpora are excluded.", "files": {}}

    def copy(src, name, origin):
        dst = a.out/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        provenance["files"][name] = {"sha256": sha_file(dst), "origin": origin}

    for name in ("audit.json", "metric_audit.json", "cpu_verification.json"):
        copy(a.review/name, name, "Independent local CPU audit: "+name)
    for name in ("environment.json", "status.json", "report.json", "verification.json", "summary.csv", "learning_curves.png", "ppt_diagnostics.png"):
        copy(run/name, "gpu/"+name, "runs/development/"+name)
    for path in sorted(run.glob("*/*")):
        if path.name.endswith(("_metrics.jsonl", "_summary.json")):
            name = path.relative_to(run).as_posix()
            copy(path, "gpu/"+name, "runs/development/"+name)
    report = read(run/"report.json")
    aggregate = []
    for arm in ARMS:
        rows = [r for r in metrics["finals"] if r["arm"] == arm]
        mean = statistics.mean(r["nll"] for r in rows)
        comparisons = [r for r in metrics["comparisons"] if r["arm"] == arm]
        result = {"arm": arm, "mean_final_validation_nll": mean, "perplexity_from_mean_nll": math.exp(mean)}
        for key in ("language_tokens", "language_seconds", "cumulative_seconds"):
            value = geometric([r[key+"_speedup"] for r in comparisons]) if comparisons else 1.
            if comparisons:
                imported = next(r for r in report["aggregate"] if r["arm"] == arm)
                require(value == imported["geomean_"+key+"_speedup_all_seeds"], "Scratch aggregate differs")
            result["geomean_"+key+"_speedup_vs_scratch"] = value
        aggregate.append(result)
    dump(a.out/"aggregate.json", aggregate)
    with (a.out/"treatment_comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics["treatment_comparisons"][0]))
        writer.writeheader()
        writer.writerows(metrics["treatment_comparisons"])
    # This additional calculation uses the SAME paired scratch target for both
    # grid and graph. Keep it distinct from the predeclared grid-endpoint result.
    secondary = []
    for seed in c["seeds"]:
        grid = next(r for r in metrics["comparisons"] if r["seed"] == seed and r["arm"] == "grid_nca")
        graph = next(r for r in metrics["comparisons"] if r["seed"] == seed and r["arm"] == "graph_nca")
        secondary.append({"seed": seed, "reference_nll": grid["reference_nll"],
                          "grid_language_updates": grid["language_step_at_reference"],
                          "graph_language_updates": graph["language_step_at_reference"],
                          "language_token_speedup_graph_over_grid": graph["language_tokens_speedup"]/grid["language_tokens_speedup"]})
    dump(a.out/"secondary_shared_scratch_target.json", {
        "scope": "Additional common-target comparison; not the primary paired grid-endpoint criterion",
        "records": secondary, "geomean_language_token_speedup_graph_over_grid": geometric([r["language_token_speedup_graph_over_grid"] for r in secondary])})
    diagnostics = {"total_prepare_seconds": manifest["total_prepare_seconds"], "replicates": {}}
    for seed, rep in manifest["replicates"].items():
        diagnostics["replicates"][seed] = {"graph": rep["graph"], "splits": {}}
        for split, stats in rep["nca"].items():
            diagnostics["replicates"][seed]["splits"][split] = {
                k: v for k, v in stats.items() if k not in ("rule_ids", "nca", "graph")}
            for kind in ("nca", "graph"):
                diagnostics["replicates"][seed]["splits"][split][kind] = {
                    k: v for k, v in stats[kind].items() if k != "profile_scores"}
    dump(a.out/"generator_diagnostics.json", diagnostics)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    comparisons = (("graph_nca", "grid_nca", "Graph minus grid"),
                   ("graph_nca", "shuffled_graph", "Graph minus shuffled graph"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = ("#2563eb", "#059669", "#d97706")
    for ax, (arm, reference, title) in zip(axes, comparisons):
        deltas = []
        for seed, color in zip(c["seeds"], colors):
            curves = [[json.loads(s) for s in (run/f"{kind}_{seed}/language_metrics.jsonl").read_text().splitlines()]
                      for kind in (arm, reference)]
            steps = np.array([r["language_step"] for r in curves[0]])
            delta = np.array([r["validation_nll"]-ref["validation_nll"] for r, ref in zip(*curves)])
            keep = steps >= 500
            deltas.append(delta[keep])
            ax.plot(steps[keep], delta[keep], color=color, alpha=.7, linewidth=1, label=f"Seed {seed}")
        ax.plot(steps[keep], np.mean(deltas, axis=0), color="#111827", linewidth=2.5, label="Paired mean")
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Identical language-stage updates", ylabel="Validation NLL difference; lower favors graph")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("V3 development result: small average differences, mixed paired outcomes")
    fig.savefig(a.out/"paired_graph_transfer.png", dpi=180)
    fig.savefig(a.out/"paired_graph_transfer.svg")
    plt.close(fig)
    dump(a.out/"PROVENANCE.json", provenance)
    print(json.dumps({"aggregate": aggregate, "primary_treatment_comparisons": report["treatment_comparisons"],
                      "archive_sha256": audit["archive_sha256"], "cpu_max_error": audit["max_cpu_evaluation_absolute_error"]}, indent=2))


if __name__ == "__main__":
    main()
