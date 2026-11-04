"""Prepare matched neural adaptation; readiness-gated fitting, no OOD entry point."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from transfer_v4.common import state_sha
from .run import digest, sources, write_new
from .complete_m2 import verify_plan as verify_m2
from .hybrid_v2 import HybridParameters, verify_case
from .numeric_adapter import NumericAdapter, PARAMETERS, encode, evaluate, learning_rate, crossing

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
LEGACY = ROOT/"Kaggle Outputs/nca_transfer_v4_run1_review/artifacts/runs/development"
ARMS = ["scratch", "grid_nca", "graph_nca", "prefix_grid", "prefix_graph"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    return sources() | {"transfer_v4/model.py": sha(ROOT/"transfer_v4/model.py"),
                        "transfer_v4/common.py": sha(ROOT/"transfer_v4/common.py")}


def eligibility():
    path = ROOT/"docs/results/readiness_v5_run1/summary.json"
    report = read(path)
    results_path = path.with_name("results.json")
    results = read(results_path)
    if sha(results_path) != read(path.with_name("results_receipt.json"))["results_sha256"]:
        raise ValueError("Readiness results receipt differs")
    selected = [r for r in results["cohorts"] if r["seed"] == 59]
    cohort_ready = (len(selected) == 2 and {r["arm"] for r in selected} == {"grid_nca", "graph_nca"}
                    and all(r["gate"]["passed"] for r in selected))
    # This integration deliberately supports the retained v4 starts ONLY.
    # No hand-edited 'ready' flag can admit a different cohort or bypass v5.
    return {"report": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
            "results_sha256": sha(results_path), "cohort_gates": [r["gate"] for r in selected],
            "approved_initializations": report["approved_initializations"],
            "allowed": cohort_ready and results["language_allowed"] and not report["language_blocked_by_guard"] and
                       len(report["approved_initializations"]) > 0,
            "reason": "retained v4 random-rule initializations fail stronger readiness; no exception inferred"}


def registry(c):
    records = []
    for arm in c["arms"]:
        folder = LEGACY/f"{arm}_{c['cohort_seed']}"
        checkpoint = folder/"language_final.pt"
        summary = read(folder/"language_summary.json")
        public = ROOT/f"docs/results/transfer_v4_run1/gpu/{arm}_{c['cohort_seed']}/language_summary.json"
        if sha(public) != sha(folder/"language_summary.json") or sha(checkpoint) != summary["checkpoint_sha256"]:
            raise ValueError("Historical source identity differs")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["stage"] != "language" or saved["language_step"] != 2000 or saved["arm"] != arm or saved["seed"] != c["cohort_seed"]:
            raise ValueError("Wrong historical stage/cohort")
        core = {k.removeprefix("core."): v for k, v in saved["model"].items() if k.startswith("core.")}
        model = NumericAdapter(saved["config"]["model"], core, c["adapter_seed"])
        adapter = {k: v for k, v in model.state_dict().items() if not k.startswith("core.")}
        ppt = None if arm == "scratch" else read(folder/"ppt_summary.json")
        records.append({"arm": arm, "seed": saved["seed"], "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha(checkpoint), "full_model_sha256": state_sha(saved["model"]),
            "core_sha256": state_sha(core), "adapter_sha256": state_sha(adapter),
            "adaptation_initial_sha256": state_sha(model.state_dict()), "architecture": saved["config"]["model"],
            "language_updates": summary["language_updates"], "language_tokens": summary["language_tokens"],
            "language_order": summary["language_train_order_sha256"], "language_schedule": summary["language_schedule_sha256"],
            "language_head_initial": summary["language_head_initial_sha256"],
            "imported_cost": {"hardware": "Tesla T4 GPU, historical", "language_seconds": summary["language_seconds"],
                "cumulative_seconds": summary["cumulative_seconds"], "ppt_updates": 0 if ppt is None else ppt["ppt_updates"],
                "ppt_tokens": 0 if ppt is None else ppt["ppt_tokens"], "ppt_stage_seconds": 0 if ppt is None else ppt["stage_seconds"],
                "status": "imported measurement; cumulative already includes original preparation, do not add stage costs again"}})
        del saved, model
    for key in ("seed", "adapter_sha256", "language_updates", "language_tokens", "language_order", "language_schedule", "language_head_initial"):
        if len({r[key] for r in records}) != 1:
            raise ValueError("Unmatched historical language/adaptation initialization policy: "+key)
    if any(r["architecture"] != records[0]["architecture"] for r in records):
        raise ValueError("Architecture mismatch")
    return records


def prepare(out, completion):
    c = read(HERE/"training_config.json")
    if c["arms"] != ARMS or c["cohort_seed"] != 59 or not c["require_stronger_synthetic_readiness"] or c["confirmation_or_ood_allowed"]:
        raise ValueError("Keep the declared cohort/all arms/readiness and ID-only policy")
    torch.set_num_threads(c["threads"])
    m2 = verify_m2(completion)
    checks, replay = read(completion/"CHECKS.json"), read(completion/"REPLAY.json")
    if not checks["passed"] or not replay["passed"] or replay["checks_sha256"] != sha(completion/"CHECKS.json") or checks["plan_sha256"] != m2["sha256"]:
        raise ValueError("Completed numerical certificate and replay required")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"PREPARATION_STARTED.json", {"scope": "reference/data/checkpoint preparation, zero neural updates"})
    records = registry(c)
    rng = np.random.default_rng(c["data_seed"])
    data, seen = {}, set()
    tick = time.perf_counter()
    for split, count in (("train", c["train_cases"]), ("development", c["development_cases"])):
        data[split] = []
        for i in range(count):
            parameters = {k: float(rng.uniform(*c["parameter_ranges"][k])) for k in PARAMETERS}
            case_id = digest(parameters)
            if case_id in seen:
                raise ValueError("Whole-tuple split overlap")
            seen.add(case_id)
            result = verify_case(HybridParameters(**parameters))
            row = {"case_id": case_id, "parameters": parameters, "split": split, **result}
            write_new(out/f"cases/{split}_{i}.json", row)
            if not result["accepted"]:
                raise ValueError("Reference failed: retained without resampling; no training freeze")
            data[split].append(row)
        write_new(out/f"{split}.json", data[split])
    reference_seconds = time.perf_counter()-tick
    order = np.random.default_rng(c["order_seed"]).integers(0, c["train_cases"], (c["updates"], c["batch_profiles"])).tolist()
    write_new(out/"ORDER.json", order)
    points = len(data["train"][0]["reference"]["eta"])
    if any(len(r["reference"]["eta"]) != points for rows in data.values() for r in rows):
        raise ValueError("All queries require the same physical mesh")
    payload = {"config": c, "sources": source_hashes(), "registry": records, "eligibility": eligibility(),
               "completion_plan_sha256": m2["sha256"], "files": {n: sha(out/n) for n in ("train.json", "development.json", "ORDER.json")},
               "expected_per_arm": {"updates": c["updates"], "query_tokens": c["updates"]*c["batch_profiles"]*points*2,
                                    "supervised_scalars": c["updates"]*c["batch_profiles"]*points*7},
               "reference_CPU_seconds": reference_seconds,
               "protocol_sha256": sha(HERE/"MILESTONE3_PROTOCOL.md"),
               "scope": "historical single-cohort numeric-adapter preparation; no fitting or confirmation"}
    write_new(out/"PLAN.json", {**payload, "sha256": digest(payload)})
    write_new(out/"GATE.json", payload["eligibility"])
    return payload


def verify_plan(out):
    p = read(out/"PLAN.json")
    if p["sha256"] != digest({k: v for k, v in p.items() if k != "sha256"}) or p["sources"] != source_hashes():
        raise ValueError("Training source/plan changed")
    if any(sha(out/n) != h for n, h in p["files"].items()) or p["registry"] != registry(p["config"]) or p["eligibility"] != eligibility():
        raise ValueError("Frozen dataset/order/starts/readiness changed")
    return p


def fit(out):
    p = verify_plan(out)
    if not p["eligibility"]["allowed"]:
        raise ValueError(p["eligibility"]["reason"])
    # The selected retained v4 cohort is currently rejected before this marker.
    write_new(out/"FIT_STARTED.json", {"plan_sha256": p["sha256"]})
    c = p["config"]
    torch.set_num_threads(c["threads"])
    torch.use_deterministic_algorithms(True)
    data = {s: read(out/f"{s}.json") for s in ("train", "development")}
    order = read(out/"ORDER.json")
    for record in p["registry"]:
        saved = torch.load(ROOT/record["checkpoint"], map_location="cpu", weights_only=True)
        core = {k.removeprefix("core."): v for k, v in saved["model"].items() if k.startswith("core.")}
        model = NumericAdapter(record["architecture"], core, c["adapter_seed"])
        if state_sha(model.state_dict()) != record["adaptation_initial_sha256"]:
            raise ValueError("Actual complete adaptation start differs")
        optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
        folder = out/"models"/record["arm"]
        folder.mkdir(parents=True, exist_ok=False)
        curve, query_tokens, seconds = [], 0, 0.0
        for step in range(c["updates"]+1):
            if step:
                tick = time.perf_counter()
                encoded = [encode(data["train"][i], c) for i in order[step-1]]
                x, y = torch.cat([r[0] for r in encoded]), torch.cat([r[1] for r in encoded])
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(step, c)
                loss = torch.nn.functional.mse_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss; preserve failed attempt")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                query_tokens += 2*len(x)
                seconds += time.perf_counter()-tick
                write_new(folder/f"update_{step}.json", {"step": step, "loss": float(loss.detach()), "query_tokens": query_tokens,
                          "order": order[step-1], "lr": optimizer.param_groups[0]["lr"]})
            if step in c["evaluation_updates"]:
                rows, predictions = evaluate(model, data["development"], c)
                write_new(folder/f"predictions_{step}.json", predictions)
                curve.append({"step": step, "query_tokens": query_tokens, "cases": rows, "training_seconds": seconds})
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                            "plan_sha256": p["sha256"]}, folder/f"step_{step}.pt")
        if query_tokens != p["expected_per_arm"]["query_tokens"]:
            raise ValueError("Actual exposure differs")
        write_new(folder/"SUMMARY.json", {"arm": record["arm"], "curve": curve, "first": crossing(curve, c),
                    "sustained": crossing(curve, c, True), "imported_cost": record["imported_cost"],
                    "training_CPU_seconds": seconds, "claim": "no controlled mixed-hardware throughput ratio"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "fit"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.completion is None:
            parser.error("--completion certificate directory is required")
        p = prepare(args.output, args.completion)
        print(json.dumps({"starts": len(p["registry"]), "training_allowed": p["eligibility"]["allowed"], "exposure": p["expected_per_arm"]}))
    elif args.command == "verify":
        p = verify_plan(args.output)
        print(json.dumps({"frozen_inputs_and_starts_verified": True, "training_allowed": p["eligibility"]["allowed"]}))
    else:
        fit(args.output)


if __name__ == "__main__":
    main()
