"""Reinterpret pilot 1 on language exposure and inspect its synthetic distribution.

This is retrospective diagnosis, not a replacement for a clean equal-stage run.
"""
import argparse
import gzip
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

import experiment as e


ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "Kaggle Outputs/gnn_nca_results_1_review"
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=REVIEW,
                        help="Full audit directory containing extracted artifacts")
    args = parser.parse_args()
    review = args.review
    artifacts = review / "artifacts"
    if not (artifacts / "runs").is_dir() or not (artifacts / "data").is_dir():
        parser.error("Full artifacts missing; run audit_kaggle_output.py --archive PATH --out PATH first")
    RUN = next(p for p in (artifacts / "runs").glob("pilot_*") if p.is_dir())
    DATA = next(p for p in (artifacts / "data").glob("pilot_*") if p.is_dir())
    c = json.loads((RUN / "environment.json").read_text())["config"]
    manifest = json.loads((DATA / "manifest.json").read_text())
    curves = {}
    for arm in e.ARMS:
        for seed in c["seeds"]:
            rows = [json.loads(line) for line in (RUN / f"{arm}_{seed}" / "metrics.jsonl").read_text().splitlines()]
            eligible = [r for r in rows if r["stage"] == "target" or r["stage"] == "target_start" or
                        (arm == "scratch" and r["stage"] in ("pre", "initial"))]
            # Scratch received target text in both v1 stages. Count ALL of it; do not
            # reset the scratch exposure clock at its artificial optimizer boundary.
            by_text_steps = {r["target_text_tokens"]//(c["batch"]*c["context"]): r for r in eligible}
            curves[arm, seed] = by_text_steps

    aligned = []
    for text_steps in (50, 100, 150, 200, 300, 500, 900):
        row = {"target_text_updates": text_steps, "target_text_tokens": text_steps*c["batch"]*c["context"]}
        for arm in e.ARMS:
            row[arm] = statistics.mean(curves[arm, s][text_steps]["val_nll"] for s in c["seeds"])
        aligned.append(row)

    diagnostics = {}
    for name in ("grid", "graph", "shuffled"):
        tokens = np.load(DATA / f"{name}.npy", allow_pickle=False)
        frames = tokens.reshape(c["trajectories"], c["frames"], -1)
        patches = frames[:, :, 1:-1]-256
        fixed = np.all(patches[:, -1] == patches[:, -2], axis=1)
        ratios = []
        for row in patches:
            # The author filter uses marker-free int32 patch IDs. We measure that
            # serialization here, but have no 10-step burn-in in this cached v1 data,
            # so this is NOT a reproduction of the author's rule screening score.
            raw = row.astype("<i4").tobytes()
            ratios.append(len(gzip.compress(raw, compresslevel=9, mtime=0))/len(raw))
        probabilities = np.bincount(patches.flatten(), minlength=16)/patches.size
        p = probabilities[probabilities > 0]
        diagnostics[name] = {
            "marker_fraction": float(np.mean(tokens >= 272)),
            "final_fixed_transition_fraction": float(np.mean(fixed)),
            "patch_unigram_entropy_bits": float(-np.sum(p*np.log2(p))),
            "max_patch_unigram_entropy_bits": 4,
            "marker_free_int32_gzip_quartiles_no_burnin": np.quantile(ratios, [.25, .5, .75]).tolist(),
            "warning": "Not directly comparable to author screening: different alphabet, trajectory length and no burn-in; original logged ratios also included markers and used uint16.",
            "original_logged_stats": manifest["synthetic"][name]
        }

    colors = dict(zip(e.ARMS, ("#64748b", "#b7791f", "#2563eb", "#059669", "#db2777")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm in e.ARMS:
        xs = sorted(set.intersection(*(set(curves[arm, s]) for s in c["seeds"])))
        xs = [x for x in xs if 0 < x <= 900]
        matrix = np.array([[curves[arm, s][x]["val_nll"] for x in xs] for s in c["seeds"]])
        mean = matrix.mean(0)
        for ax in axes:
            ax.plot(np.array(xs)*c["batch"]*c["context"]/1e6, mean, label=arm, color=colors[arm])
            ax.fill_between(np.array(xs)*c["batch"]*c["context"]/1e6, matrix.min(0), matrix.max(0), color=colors[arm], alpha=.1)
            ax.set_xlabel("Target-language tokens seen (millions)")
            ax.set_ylabel("Validation NLL (nats per byte)")
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[1].set_xlim(0, 1.3)
    axes[1].set_ylim(2.1, 3.3)
    axes[1].set_title("Early language training")
    fig.suptitle("Retrospective language-exposure alignment — v1 scheduler remains confounded")
    fig.savefig(review / "language_aligned_curves.png", dpi=180)
    fig.savefig(review / "language_aligned_curves.svg")
    plt.close(fig)
    result = {"status": "retrospective diagnosis, not confirmatory evidence",
              "caveat": "Scratch optimizer restarted after 100 TEXT updates; NCA language optimizer started after 100 SYNTHETIC updates. Equal-language-token alignment cannot repair this scheduler difference or recover missing checkpoints.",
              "aligned_validation": aligned, "synthetic_diagnostics": diagnostics,
              "missing_measurements": ["NCA train/held-out-rule loss", "pre-pre-training checkpoint", "fresh scratch language-only run with matched text-stage scheduler", "matched-language-budget final scratch test checkpoint"]}
    e.dump(review / "prepretraining_diagnosis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
