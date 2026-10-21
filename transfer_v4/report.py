from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .common import ARMS, SYNTHETIC_ARMS, dump
from .analysis import common_target_analysis


def first_hit(curve, threshold):
    return next((r for r in curve if r["validation_nll"] <= threshold), None)


def report(root, c):
    root = Path(root)
    status = json.loads((root/"status.json").read_text())
    summaries, curves = {}, {}
    for arm in ARMS:
        for seed in c["seeds"]:
            folder = root/f"{arm}_{seed}"
            if (folder/"language_summary.json").exists():
                summaries[arm, seed] = json.loads((folder/"language_summary.json").read_text())
                curves[arm, seed] = [json.loads(s) for s in (folder/"language_metrics.jsonl").read_text().splitlines()]
    comparisons = []
    for seed in c["seeds"]:
        if ("scratch", seed) not in summaries:
            continue
        reference = summaries["scratch", seed]["final_validation_nll"]
        base_hit = first_hit(curves["scratch", seed], reference)
        for arm in SYNTHETIC_ARMS:
            if (arm, seed) not in summaries:
                continue
            hit = first_hit(curves[arm, seed], reference)
            row = {"arm": arm, "seed": seed, "reference_nll": reference,
                   "reference_policy": "Paired scratch final validation loss at its declared N language updates",
                   "reference_reached": hit is not None,
                   "language_step_at_reference": hit["language_step"] if hit else None,
                   "final_validation_nll_delta": summaries[arm, seed]["final_validation_nll"]-reference}
            for key in ("language_tokens", "language_train_seconds", "language_seconds", "total_training_tokens", "cumulative_seconds"):
                row[f"{key}_speedup"] = base_hit[key]/hit[key] if hit and hit[key] > 0 else None
            comparisons.append(row)
    aggregate = []
    for arm in SYNTHETIC_ARMS:
        rows = [r for r in comparisons if r["arm"] == arm]
        if not rows:
            continue
        deltas = [r["final_validation_nll_delta"] for r in rows]
        a = {"arm": arm, "paired_seeds": len(rows), "seeds_reaching_reference": sum(r["reference_reached"] for r in rows),
             "mean_final_validation_nll_delta": statistics.mean(deltas)}
        for key in ("language_tokens", "language_seconds", "cumulative_seconds"):
            values = [r[f"{key}_speedup"] for r in rows]
            # Do not publish a success-only mean when any seed is censored.
            complete = all(v is not None and v > 0 for v in values)
            a[f"geomean_{key}_speedup_all_seeds"] = math.exp(statistics.mean(math.log(v) for v in values)) if complete else None
            if complete and len(values) >= 2:
                rng = np.random.default_rng(101)
                samples = np.exp(np.mean(rng.choice(np.log(values), (10000, len(values)), replace=True), axis=1))
                a[f"exploratory_95_ci_{key}"] = np.quantile(samples, [.025, .975]).tolist()
            else:
                a[f"exploratory_95_ci_{key}"] = None
        aggregate.append(a)
    treatment_comparisons = []
    for reference_arm, arm in (("grid_nca", "graph_nca"), ("shuffled_graph", "graph_nca"), ("shuffled_nca", "grid_nca")):
        rows = []
        for seed in c["seeds"]:
            if (reference_arm, seed) not in summaries or (arm, seed) not in summaries:
                continue
            threshold = summaries[reference_arm, seed]["final_validation_nll"]
            base = first_hit(curves[reference_arm, seed], threshold)
            hit = first_hit(curves[arm, seed], threshold)
            row = {"arm": arm, "reference_arm": reference_arm, "seed": seed,
                   "reference_nll": threshold, "reference_reached": hit is not None,
                   "language_step_at_reference": hit["language_step"] if hit else None,
                   "final_validation_nll_delta": summaries[arm, seed]["final_validation_nll"] - threshold}
            for key in ("language_tokens", "language_seconds", "cumulative_seconds"):
                row[key+"_speedup"] = base[key]/hit[key] if hit and hit[key] > 0 else None
            rows.append(row)
        if rows:
            item = {"arm": arm, "reference_arm": reference_arm, "paired_replicates": len(rows),
                    "seeds_reaching_reference": sum(r["reference_reached"] for r in rows),
                    "mean_final_validation_nll_delta": statistics.mean(r["final_validation_nll_delta"] for r in rows),
                    "comparisons": rows}
            for key in ("language_tokens", "language_seconds", "cumulative_seconds"):
                values = [r[key+"_speedup"] for r in rows]
                item["geomean_"+key+"_speedup_all_seeds"] = math.exp(statistics.mean(math.log(v) for v in values)) if all(v is not None and v > 0 for v in values) else None
            treatment_comparisons.append(item)
    result = {"schema_version": 4, "purpose": c["purpose"], "status": status,
              "primary_question": "Does graph PPT beat grid PPT and exact-prefix shuffled graph PPT at the same paired scratch-endpoint target?",
              "primary_analysis": common_target_analysis(curves, c),
              "secondary_question": "Do language-stage savings repay PPT and generation costs?",
              "graph_status": "Paired grid/graph frozen rules, same complexity band and independent topology/cache per paired model seed",
              "treatment_comparisons": treatment_comparisons,
              "comparisons": comparisons, "aggregate": aggregate,
              "uncertainty_note": "Six fresh paired model/data/topology cohorts do not separately estimate these variance sources. Validation task was inspected in earlier versions. Evaluation crossings are discrete; a gate pass is not proof of dynamics-specific transfer. Legacy reference-endpoint comparisons below are secondary.",
              "test_policy": "Reserved and not loaded" if not c["text"]["evaluate_test"] else "Predeclared final evaluation only"}
    dump(root/"report.json", result)
    plot_ppt(root, c)
    if summaries:
        flat = [{k: v for k, v in s.items() if not isinstance(v, dict)} for s in summaries.values()]
        with (root/"summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)
        plot(root, c, curves)
    print(json.dumps({"status": status, "primary_analysis": result["primary_analysis"], "aggregate": aggregate}, indent=2), flush=True)
    return result


def plot_ppt(root, c):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for arm, color in (("grid_nca", "#2563eb"), ("graph_nca", "#059669"),
                       ("shuffled_nca", "#db2777"), ("shuffled_graph", "#d97706"),
                       ("prefix_grid", "#9333ea"), ("prefix_graph", "#0891b2")):
        for i, seed in enumerate(c["seeds"]):
            folder = Path(root)/f"{arm}_{seed}"
            if not (folder/"ppt_summary.json").exists():
                continue
            rows = [json.loads(s) for s in (folder/"ppt_metrics.jsonl").read_text().splitlines()]
            summary = json.loads((folder/"ppt_summary.json").read_text())
            label = arm if i == 0 else None
            trained = [r for r in rows if r["train_loss"] is not None]
            axes[0].plot([r["ppt_step"] for r in trained], [r["train_loss"] for r in trained], color=color, alpha=.65, label=label)
            axes[1].plot([r["ppt_step"] for r in rows], [r["validation"]["patch_nll"] for r in rows], color=color, alpha=.65)
            axes[2].plot([r["ppt_step"] for r in rows], [r["validation"]["patch_accuracy"] for r in rows], color=color, alpha=.65)
            if i == 0:
                bound = min(summary["controls"]["unigram_patch_nll"], summary["controls"]["copy_mixture_patch_nll"])
                axes[1].axhline(bound, color=color, linestyle="--", linewidth=.8)
    for ax, title, ylabel in zip(axes, ("Training objective", "Unseen-rule patch prediction", "Unseen-rule accuracy"),
                                 ("Cross-entropy (conditioning masked)", "Patch NLL (dashed: best fitted control)", "Patch accuracy")):
        ax.set_title(title)
        ax.set_xlabel("PPT updates")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Synthetic-stage diagnostics — a gate pass does not establish language transfer")
    fig.savefig(Path(root)/"ppt_diagnostics.png", dpi=180)
    fig.savefig(Path(root)/"ppt_diagnostics.svg")
    plt.close(fig)


def plot(root, c, curves):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    colors = {"scratch": "#64748b", "grid_nca": "#2563eb", "graph_nca": "#059669",
              "shuffled_nca": "#db2777", "shuffled_graph": "#d97706",
              "prefix_grid": "#9333ea", "prefix_graph": "#0891b2"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    keys = ("language_tokens", "language_seconds", "cumulative_seconds")
    for (arm, seed), rows in curves.items():
        for ax, key in zip(axes, keys):
            scale = 1e6 if key == "language_tokens" else 1
            ax.plot([r[key]/scale for r in rows], [r["validation_nll"] for r in rows],
                    color=colors[arm], alpha=.6, label=arm if seed == c["seeds"][0] else None)
    for ax, title, xlabel in zip(axes, ("Language exposure", "Language-stage time", "PPT + language cost"),
                                 ("Language tokens (millions)", "Seconds since language step 0", "Recorded cumulative seconds")):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Validation NLL (nats per language token)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    prefix = "Execution fixture; no research inference" if c["purpose"] == "execution_check" else "Graph vs grid pre-pre-training into identical language stages"
    fig.suptitle(prefix)
    fig.savefig(root/"learning_curves.png", dpi=180)
    fig.savefig(root/"learning_curves.svg")
    plt.close(fig)
