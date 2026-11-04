"""Run the separately declared local milestone 3 exploratory exception."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from scientific_transfer import training as original
from scientific_transfer.numeric_adapter import decision
from transfer_v4.common import state_sha

ROOT = original.ROOT
PREP = ROOT / "runs/scientific_transfer/m3_preparation"
PROTOCOL = ROOT / "docs/SCIENTIFIC_M3_EXPLORATORY.md"


def freeze(out):
    parent = original.verify_plan(PREP)
    published = ROOT / "docs/results/scientific_transfer_completion/milestone3_preparation/PLAN.json"
    if original.sha(published) != original.sha(PREP / "PLAN.json") or parent["eligibility"]["allowed"]:
        raise ValueError("Expected the retained, readiness-rejected preparation")
    out.mkdir(parents=True, exist_ok=False)
    original.write_new(out / "PLAN.json", {
        "parent_sha256": original.sha(published),
        "protocol_sha256": original.sha(PROTOCOL),
        "runner_sha256": original.sha(__file__),
        "torch_version": str(torch.__version__),
        "scope": "single-cohort exploratory local CPU adaptation",
        "synthetic_readiness_passed": False,
        "ood_authorized": False,
    })


def fit(out, arm):
    parent = original.verify_plan(PREP)
    plan = original.read(out / "PLAN.json")
    if (plan["parent_sha256"] != original.sha(PREP / "PLAN.json")
            or plan["protocol_sha256"] != original.sha(PROTOCOL)
            or plan["runner_sha256"] != original.sha(__file__)
            or plan["torch_version"] != str(torch.__version__)
            or parent["eligibility"]["allowed"]):
        raise ValueError("Frozen exploratory identities differ")
    config = parent["config"]
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    record = next(r for r in parent["registry"] if r["arm"] == arm)
    saved = torch.load(ROOT / record["checkpoint"], map_location="cpu", weights_only=True)
    core = {key.removeprefix("core."): value for key, value in saved["model"].items()
            if key.startswith("core.")}
    model = original.NumericAdapter(record["architecture"], core, config["adapter_seed"])
    if state_sha(model.state_dict()) != record["adaptation_initial_sha256"]:
        raise ValueError("Initial tensors differ")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    train = [original.encode(case, config) for case in original.read(PREP / "train.json")]
    dev = original.read(PREP / "development.json")
    order = original.read(PREP / "ORDER.json")
    folder = out / arm
    folder.mkdir(exist_ok=False)
    curve, updates = [], []
    tokens = scalars = 0
    train_seconds = evaluation_seconds = 0.0
    for step in range(config["updates"] + 1):
        if step:
            tick = time.perf_counter()
            batch = [train[i] for i in order[step - 1]]
            x, y = (torch.cat(values) for values in zip(*batch))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = original.learning_rate(step, config)
            optimizer.param_groups[0]["lr"] = lr
            loss = torch.nn.functional.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            tokens += 2 * len(x)
            scalars += y.numel()
            train_seconds += time.perf_counter() - tick
            updates.append({"step": step, "lr": lr, "loss": float(loss.detach())})
        if step in config["evaluation_updates"]:
            tick = time.perf_counter()
            cases, predictions = original.evaluate(model, dev, config)
            evaluation_seconds += time.perf_counter() - tick
            for case in cases:
                if not case["valid"]:
                    predictions[case["case_id"]] = None
            original.write_new(folder / f"predictions_{step}.json", predictions)
            curve.append({"step": step, "query_tokens": tokens, "cases": cases})
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "plan_sha256": original.sha(out / "PLAN.json")},
                       folder / f"step_{step}.pt")
            original.write_new(folder / f"updates_through_{step}.json", updates)
            print(arm, step, "ID competence", decision(cases, config), flush=True)
    exposure = {"updates": len(updates), "query_tokens": tokens, "supervised_scalars": scalars}
    if exposure != parent["expected_per_arm"]:
        raise ValueError("Actual exposure differs")
    original.write_new(folder / "SUMMARY.json", {
        "arm": arm, "exposure": exposure, "curve": curve,
        "initial_sha256": record["adaptation_initial_sha256"],
        "first": original.crossing(curve, config), "sustained": original.crossing(curve, config, True),
        "endpoint_competent": decision(curve[-1]["cases"], config),
        "training_CPU_seconds": train_seconds, "evaluation_CPU_seconds": evaluation_seconds,
        "imported_cost": record["imported_cost"],
    })


def report(out, output):
    config = original.read(PREP / "PLAN.json")["config"]
    summaries = [original.read(out / arm / "SUMMARY.json") for arm in config["arms"]]
    result = {"plan": original.read(out / "PLAN.json"), "summaries": summaries,
              "id_gate_passed": all(s["endpoint_competent"] for s in summaries),
              "ood_executed": False}
    original.write_new(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "fit", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=original.ARMS)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
    elif args.command == "fit":
        fit(args.out, args.arm)
    else:
        report(args.out, args.report)
