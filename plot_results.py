"""Export standard Matplotlib figures from recorded measurements."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from experiment import ARMS


def plot(root):
    root = Path(root)
    env = json.loads((root / "environment.json").read_text())
    colors = dict(zip(ARMS, ("#64748b", "#b7791f", "#2563eb", "#059669", "#db2777")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm in ARMS:
        for index, seed in enumerate(env["config"]["seeds"]):
            path = root / f"{arm}_{seed}" / "metrics.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            # Eligible byte-LM evaluations; no synthetic-symbol pre-stage losses.
            rows = [r for r in rows if r["stage"] in ("target_start", "target") or
                    (arm in ("scratch", "reserve_text") and r["stage"] == "pre")]
            for ax, key, scale in zip(axes, ("total_tokens", "pipeline_seconds"), (1e6, 1)):
                ax.plot([r[key]/scale for r in rows], [r["val_nll"] for r in rows],
                        color=colors[arm], alpha=.7, linewidth=1.5,
                        label=arm if index == 0 else None)
    for ax in axes:
        ax.axhline(env["config"]["target_nll"], color="#111827", linestyle="--", linewidth=.8, label="Declared target")
        ax.set_ylabel("Validation NLL (nats per byte)")
        ax.grid(alpha=.2)
    axes[0].set_xlabel("Total training tokens (millions; both stages)")
    axes[1].set_xlabel("Pipeline time (seconds; preparation included)")
    axes[0].legend(fontsize=8)
    device = env["device"].upper()
    fig.suptitle(f"Graph-NCA transfer — {device}, {len(env['config']['seeds'])} paired seed(s)")
    fig.savefig(root / "curves.svg")
    fig.savefig(root / "curves.png", dpi=180)
    plt.close(fig)
    print(f"Exported {root / 'curves.svg'} and curves.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/pilot")
    plot(p.parse_args().out)
