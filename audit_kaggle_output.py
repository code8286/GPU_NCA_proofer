"""Independently audit the first returned Kaggle ZIP using trusted local code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import time
import zipfile

import numpy as np
import torch

import experiment as e


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "Kaggle Outputs/gnn_nca_results_1.zip"
OUTPUT = ROOT / "Kaggle Outputs/gnn_nca_results_1_review"
ARTIFACTS = OUTPUT / "artifacts"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    global ARCHIVE, OUTPUT, ARTIFACTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE,
                        help="Original pilot-1 results ZIP (not distributed with source)")
    parser.add_argument("--out", type=Path, default=OUTPUT,
                        help="Derived audit directory; original ZIP remains unchanged")
    args = parser.parse_args()
    ARCHIVE, OUTPUT = args.archive, args.out
    ARTIFACTS = OUTPUT / "artifacts"
    if not ARCHIVE.is_file():
        parser.error("Original results archive not found; supply --archive PATH")
    start = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None, "Corrupted ZIP member"
        names = z.namelist()
        assert len(names) == len(set(names)), "Duplicate archive members"
        assert sum(i.file_size for i in z.infolist()) < 1_000_000_000
        for member in z.infolist():
            p = PurePosixPath(member.filename)
            assert not p.is_absolute() and ".." not in p.parts and "\\" not in member.filename
            assert ":" not in member.filename
            assert (member.external_attr >> 16) & 0o170000 != 0o120000, "Archive symlink"
            target = (ARTIFACTS / Path(*p.parts)).resolve()
            assert target.is_relative_to(ARTIFACTS.resolve())
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = z.read(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == content, f"Existing extracted file changed: {target}"
            else:
                target.write_bytes(content)
        pilot_names = [n for n in names if n.startswith("runs/pilot_") and n.endswith("/environment.json")]
        assert len(pilot_names) == 1
        run_name = pilot_names[0].rsplit("/", 1)[0]
        env = json.loads(z.read(pilot_names[0]))
        c = env["config"]
        manifest_name = next(n for n in names if n.startswith("data/pilot_") and n.endswith("/manifest.json"))
        manifest = json.loads(z.read(manifest_name))
        data_name = manifest_name.rsplit("/", 1)[0]
        # Treat all returned text/code as data. Only trusted local code is executed.
        source_matches = z.read("experiment.py") == (ROOT / "experiment.py").read_bytes()
        assert source_matches
        assert sha(z.read("experiment.py")) == env["source_sha256"]
        assert sha(z.read(manifest_name)) == env["data_manifest_sha256"]
        assert c == manifest["config"] == json.loads(z.read("configs/pilot.json"))
        for filename, expected in manifest["files"].items():
            assert sha(z.read(f"{data_name}/{filename}")) == expected, filename
        source_name = next(n for n in names if n.endswith("/input.txt"))
        assert sha(z.read(source_name)) == manifest["source_sha256"]
        source = z.read(source_name)
        data = {n: np.load(io.BytesIO(z.read(f"{data_name}/{n}.npy")), allow_pickle=False)
                for n in ("reserve", "train", "val", "test", "grid", "graph", "shuffled")}
        last_end = 0
        for name in ("reserve", "train", "val", "test"):
            split = manifest["splits"][name]
            assert split["start_byte"] == last_end
            a, b = split["start_byte"], split["end_byte"]
            np.testing.assert_array_equal(data[name], np.frombuffer(source[a:b], dtype=np.uint8))
            last_end = b
        assert last_end == len(source)
        np.testing.assert_array_equal(np.sort(data["graph"], axis=1), np.sort(data["shuffled"], axis=1))
        np.testing.assert_array_equal(data["graph"] >= 272, data["shuffled"] >= 272)
        assert not np.array_equal(data["graph"], data["shuffled"])
        edges = np.load(io.BytesIO(z.read(f"{data_name}/graph_edges.npy")), allow_pickle=False)
        for node, row in enumerate(edges):
            assert len(set(row)) == 8 and node not in row
        for relation in edges.T:
            np.testing.assert_array_equal(np.sort(relation), np.arange(c["grid"]**2))
        curves, summaries, hits = {}, {}, {}
        for arm in e.ARMS:
            for seed in c["seeds"]:
                prefix = f"{run_name}/{arm}_{seed}"
                summary = json.loads(z.read(f"{prefix}/summary.json"))
                rows = [json.loads(line) for line in z.read(f"{prefix}/metrics.jsonl").decode().splitlines()]
                assert summary["arm"] == arm and summary["seed"] == seed
                assert summary["parameters"] == 1_857_024
                assert summary["total_tokens"] == c["batch"]*c["context"]*c["total_steps"]
                assert rows[-1]["step"] == c["total_steps"] == 1000
                assert rows[-1]["total_tokens"] == summary["total_tokens"]
                assert rows[-1]["val_nll"] == summary["final_val_nll"]
                assert all(r["arm"] == arm and r["seed"] == seed for r in rows)
                assert all(math.isfinite(r["val_nll"]) for r in rows)
                assert all(a["step"] <= b["step"] for a, b in zip(rows, rows[1:]))
                assert all(a["pipeline_seconds"] <= b["pipeline_seconds"] for a, b in zip(rows, rows[1:]))
                assert sha(z.read(f"{prefix}/model.pt")) == summary["model_sha256"]
                hit = next(r for r in rows if (r["stage"] in ("target_start", "target") or
                           (arm in ("scratch", "reserve_text") and r["stage"] == "pre")) and r["val_nll"] <= c["target_nll"])
                hits[arm, seed] = hit
                summaries[arm, seed] = summary
                curves[arm, seed] = rows
        for seed in c["seeds"]:
            assert len({summaries[a, seed]["initial_sha256"] for a in e.ARMS}) == 1
            assert len({curves[a, seed][0]["val_nll"] for a in e.ARMS}) == 1
        original = json.loads(z.read(f"{run_name}/report.json"))
        assert len(original["comparisons"]) == 12
        for pair in original["comparisons"]:
            base, seed = pair["baseline"], pair["seed"]
            for key in ("total_tokens", "target_text_tokens", "train_seconds", "pipeline_seconds"):
                expected = hits[base, seed][key]/hits["graph_nca", seed][key]
                assert math.isclose(expected, pair[f"{key}_speedup"], abs_tol=1e-12)
            delta = summaries["graph_nca", seed]["test_nll"] - summaries[base, seed]["test_nll"]
            assert math.isclose(delta, pair["graph_minus_baseline_test_nll"], abs_tol=1e-12)
        gpu_verification = json.loads(z.read(f"{run_name}/verification.json"))
        assert gpu_verification["checkpoints_verified"] == 15
        assert all(x["absolute_error"] == 0 and x["checkpoint_sha256_valid"] for x in gpu_verification["records"])
    print("Archive, source, data, budgets, 15 checkpoints, and all 12 speed comparisons passed integrity checks.", flush=True)

    # Recompute every held-out loss locally on CPU, preserving the returned GPU
    # verification.json unchanged. No extracted script is imported or executed.
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tests = e.eval_batches(data["test"], c, "cpu")
    local = []
    for (arm, seed), summary in summaries.items():
        path = ARTIFACTS / run_name / f"{arm}_{seed}/model.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["config"] == c and saved["arm"] == arm and saved["seed"] == seed
        assert saved["initial_sha256"] == summary["initial_sha256"]
        model = e.LM(c)
        model.load_state_dict(saved["model"], strict=True)
        assert sum(p.numel() for p in model.parameters()) == summary["parameters"]
        loss = e.evaluate(model, tests)
        error = abs(loss-summary["test_nll"])
        assert error < 1e-4, (arm, seed, error)
        local.append({"arm": arm, "seed": seed, "gpu_test_nll": summary["test_nll"],
                      "cpu_recomputed_test_nll": loss, "absolute_error": error})
        print(f"CPU verified {arm} seed={seed}: NLL={loss:.8f}; GPU difference={error:.2g}", flush=True)

    table = []
    for arm in e.ARMS:
        losses = [summaries[arm, s]["test_nll"] for s in c["seeds"]]
        row = {"arm": arm, "mean_test_nll": statistics.mean(losses),
               "sample_sd_test_nll": statistics.stdev(losses),
               "geometric_mean_byte_perplexity": math.exp(statistics.mean(losses)),
               "mean_seconds_to_target": statistics.mean(hits[arm, s]["pipeline_seconds"] for s in c["seeds"]),
               "mean_total_tokens_to_target": statistics.mean(hits[arm, s]["total_tokens"] for s in c["seeds"]),
               "mean_target_text_tokens_to_target": statistics.mean(hits[arm, s]["target_text_tokens"] for s in c["seeds"])}
        table.append(row)
    timing = {"summed_training_seconds": sum(s["train_seconds"] for s in summaries.values()),
              "summed_last_logged_pipeline_seconds": sum(rows[-1]["pipeline_seconds"] for rows in curves.values()),
              "note": "Summed per-arm recorded costs, not exact notebook wall time; excludes startup, final checkpoints, verification, plotting and archive creation."}
    result = {"archive_sha256": e.digest(ARCHIVE), "gpu": env["gpu"], "pilot_runs_completed": 15,
              "gpu_saved_verification_count": 15, "local_cpu_verification": local,
              "max_cpu_gpu_nll_error": max(x["absolute_error"] for x in local),
              "all_integrity_checks_passed": True, "all_reported_speedups_recomputed": True,
              "table": table, "timing": timing, "original_aggregates": original["aggregate"],
              "audit_seconds": time.perf_counter()-start}
    e.dump(OUTPUT / "audit.json", result)
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    scratch, grid, graph = (next(r for r in table if r["arm"] == a) for a in ("scratch", "grid_nca", "graph_nca"))
    speed = next(r for r in original["aggregate"] if r["baseline"] == "scratch")["geomean_pipeline_speedup_reached_pairs_only"]
    lines = ["# Review of Kaggle output 1", "",
             "**Protocol qualification:** this is an audit of the original fixed-total-budget pilot, not a clean equal-language-stage transfer experiment. Scratch received 1000 text updates; synthetic arms received 100 synthetic + 900 text updates. See `../../METHODOLOGY_AUDIT.md` for the paper/code comparison and scheduler qualification.",
             "", "## Finding", "",
             "The run completed successfully, but this pilot does **not** show that graph-NCA pre-training accelerates language-model training.",
             f"Graph NCA took **{1/speed:.3f}x as long** to reach the declared 2.5 validation-nats/byte target as target-text-only training (paired geometric ratio).",
             f"Its final held-out NLL was higher by **{graph['mean_test_nll']-scratch['mean_test_nll']:.6f} nats/byte**, corresponding to **{100*(math.exp(graph['mean_test_nll']-scratch['mean_test_nll'])-1):.2f}% higher geometric-mean byte perplexity**.",
             "All three seeds favored text-only over graph NCA on both time to target and final held-out loss.",
             "", "## Checks completed", "",
             "- All 15 pilot runs reached step 1000: five arms and three paired seeds.",
             "- Every run used 1,857,024 parameters and 4,096,000 total training tokens; total pilot exposure was 61,440,000 tokens.",
             "- Returned experiment source exactly matches the trusted local source and its recorded SHA-256.",
             "- ZIP CRCs, all dataset checksums, source text, split ranges, and checkpoint SHA-256 hashes passed.",
             "- Initialization hashes and initial validation loss match within each paired seed.",
             "- Rewired graph preserves incoming/outgoing degree and has no self/duplicate edges; shuffled data preserve per-row frequencies and marker positions.",
             "- All 12 reported pairwise comparisons were independently recomputed from the raw logs.",
             "- Kaggle's saved GPU verification reports zero error for 15 reloaded checkpoints.",
             f"- All 15 checkpoints were independently re-evaluated locally on CPU; maximum CPU/GPU loss difference was **{result['max_cpu_gpu_nll_error']:.3g}**, below the declared 0.0001 tolerance.",
             "", "## Results", "", "Means across three seeds. Lower loss, perplexity, time and tokens are better.", "",
             "| Arm | Test NLL, nats/byte | Byte perplexity, geometric mean | Pipeline seconds to target, arithmetic mean | Total tokens to target, arithmetic mean |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for r in table:
        lines.append(f"| {r['arm']} | {r['mean_test_nll']:.4f} | {r['geometric_mean_byte_perplexity']:.3f} | {r['mean_seconds_to_target']:.3f} | {r['mean_total_tokens_to_target']:,.0f} |")
    lines += ["", "The target is validation NLL <= 2.5; time/token estimates are the first recorded evaluation reaching it, without interpolation.",
              "Evaluations occur every 50 updates (204,800 tokens), so small differences in threshold crossing are coarse. Loss is evaluated on 65,536 fixed held-out bytes for validation and likewise for test.",
              "The text-only baseline trains on target text in both stages; the separate-text baseline uses a disjoint same-domain reserve before target training.",
              "", "## Interpretation", "",
              "1. **No practical speedup against text-only.** The graph pipeline speedup is 0.698x (baseline time / graph time). Equivalently graph takes about 43% longer at this target. Its final test loss is worse in every seed.",
              "2. **No demonstrated graph-topology benefit over grid NCA.** Both cross the target at the same recorded update in every paired seed. The 1.005x aggregate timing ratio is about 0.5%, with mixed signs and an exploratory interval spanning one. Graph's mean test NLL is slightly worse.",
              "3. **No convincing evidence that temporal trajectory structure helps.** Graph vs shuffled graph has virtually identical mean test loss (delta -0.000215 nats/byte). A roughly 6% aggregate threshold timing advantage is driven mainly by one seed crossing one evaluation earlier. This is not evidence of a general structural transfer gain.",
              "4. **Some token savings disappear when all training is counted.** Graph's target-text token speedups against scratch are 1.00x, 1.00x and 1.33x, but total-token speedups are only 0.667x, 0.667x and 0.800x. Synthetic exposure is extra work and must remain in the accounting.",
              "5. **These results apply to this small pilot.** They do not disprove NCA transfer at other scales or with other rules. Only one synthetic dataset/topology is shared across the three model seeds. Three seeds and seed-bootstrap intervals remain exploratory.",
              "", "## Runtime and generator diagnostics", "",
              f"Hardware recorded: **{env['gpu']}**, PyTorch {env['torch']}, fp32. Summed training time was **{timing['summed_training_seconds']/60:.2f} minutes**; summed last-logged per-arm pipeline costs were **{timing['summed_last_logged_pipeline_seconds']/60:.2f} minutes**. These are not the complete notebook wall time.",
              "The earlier 30–90 minute planning estimate was overly conservative for this actual GPU run.",
              "Graph synthetic preparation was recorded as 0.151 seconds, versus 0.327 seconds for grid, but each generator was timed once in a fixed order. Kernel warmup and order can explain this difference; it does not establish a GNN generation-throughput advantage.",
              "About 36.3% of graph trajectories were at a fixed point on their final transition (grid: 35.7%). This is a useful warning about trivial dynamics, not a proven explanation of transfer performance.",
              "", "## Next experiment", "",
              "Preserve this run as a negative pilot. Before another full suite, inspect trajectories and pre-register a generator study with nontrivial dynamics and matched complexity bands. Treat embedding reset/transfer policy as a separate ablation: synthetic arms restore byte embeddings while text arms preserve them. Any tuned follow-up needs untouched confirmation data; do not use this test set to select a winning configuration.",
              "The current results compare graph topology in a frozen random-rule NCA data generator. They do not compare optimized GPU graph kernels against convolution kernels or put a GNN inside the language model.",
              "", "## Files", "",
              "- `audit.json`: independent audit and CPU verification measurements.",
              "- `comparison.csv`: aggregate metrics recalculated from the raw logs.",
              f"- `artifacts/{run_name}/`: original Kaggle metrics, plots, checkpoints, and GPU verification (preserved unchanged).",
              "- The original ZIP is preserved unchanged."]
    (OUTPUT / "REVIEW.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"table": table, "timing": timing, "max_cpu_gpu_nll_error": result["max_cpu_gpu_nll_error"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
