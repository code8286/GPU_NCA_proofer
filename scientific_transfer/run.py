"""CLI: plan, generate train/dev references, evaluate guesses, benchmark correction."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.interpolate import CubicHermiteSpline

from .metrics import score
from .physics import Parameters, initial_guess, rhs, solve

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sources():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.iterdir()) if p.suffix in {".py", ".json", ".md", ".txt"}}


def write_new(path, value):
    # Existing evidence is immutable. Serialize before opening to avoid partial files.
    content = json.dumps(value, indent=2, allow_nan=False)+"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def make_plan(config):
    if config["model_id"] != "steady_bridge_v1" or config["training_enabled"] or config["manuscript_replication_enabled"]:
        raise ValueError("Only the non-training steady bridge is implemented")
    rng = np.random.default_rng(config["seed"])
    rows, seen = [], set()
    # Whole parameter tuples are partitioned; meshes from one tuple never cross splits.
    for split, count in config["counts"].items():
        if not isinstance(count, int) or count < 1:
            raise ValueError("Each split needs a positive integer count")
        ranges = dict(config["ranges"])
        if split == "extrapolation":
            ranges.update(config["extrapolation"])
        for _ in range(count):
            p = {}
            for name, (low, high) in ranges.items():
                if not np.isfinite([low, high]).all() or high <= low:
                    raise ValueError("Invalid parameter interval")
                p[name] = float(rng.uniform(low, high))
            Parameters(**p)
            case_id = digest(p)
            if case_id in seen:
                raise ValueError("Duplicate parameter tuple")
            seen.add(case_id)
            rows.append({"case_id": case_id, "split": split, "parameters": p})
    payload = {"config": config, "sources": sources(), "rows": rows,
               "environment": {"python": platform.python_version(),
                               "numpy": np.__version__, "scipy": scipy.__version__},
               "status": "development_solver_plan_not_training_freeze"}
    return {**payload, "plan_sha256": digest(payload)}


def validate_plan(plan):
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if digest(payload) != plan["plan_sha256"] or plan["sources"] != sources():
        raise ValueError("Plan/source mismatch: create a separately identified new plan")


def reference_case(row, config):
    p = Parameters(**row["parameters"])
    base = solve(p, **config["solver"])
    check = solve(p, **config["verification"])
    # Compare common physical locations; never rescale the longer verification domain.
    x = np.asarray(base["eta"])
    check_x, check_y = np.asarray(check["eta"]), np.asarray(check["state"])
    extended = CubicHermiteSpline(check_x, check_y, rhs(check_x, check_y, p), axis=1)(x)
    profile_delta = float(np.max(np.abs(np.asarray(base["state"])[[1, 3, 5]]-extended[[1, 3, 5]])))
    wall_delta = max(abs(base["wall"][k]-check["wall"][k]) for k in base["wall"])
    accepted = bool(base["accepted"] and check["accepted"]
                    and max(profile_delta, wall_delta) <= config["agreement_absolute"])
    return {**row, "reference": base, "verification": check,
            "profile_agreement_max": profile_delta, "wall_agreement_max": wall_delta,
            "accepted": accepted}


def generate(plan, split):
    validate_plan(plan)
    if split not in {"train", "development"}:
        raise ValueError("Assessment/extrapolation solution generation is gated")
    cases = []
    for row in plan["rows"]:
        if row["split"] != split:
            continue
        try:
            cases.append(reference_case(row, plan["config"]))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            cases.append({**row, "accepted": False,
                          "error": {"type": type(exc).__name__, "message": str(exc)}})
    return {"plan_sha256": plan["plan_sha256"], "model_id": plan["config"]["model_id"],
            "split": split, "cases": cases, "attempted": len(cases),
            "accepted": sum(c["accepted"] for c in cases), "measurement": "new_CPU_reference_solves"}


def evaluate(dataset, predictions):
    expected = {c["case_id"] for c in dataset["cases"]}
    if set(predictions)-expected:
        raise ValueError("Predictions include unknown case IDs")
    rows = []
    for case in dataset["cases"]:
        key = case["case_id"]
        if not case["accepted"]:
            rows.append({"case_id": key, "status": "reference_failed"})
        elif key not in predictions:
            rows.append({"case_id": key, "status": "missing_prediction"})
        else:
            ref = case["reference"]
            metrics = score(ref["eta"], predictions[key], ref["state"], Parameters(**case["parameters"]))
            rows.append({"case_id": key, "status": "scored" if metrics["valid"] else "invalid_prediction", **metrics})
    return {"plan_sha256": dataset["plan_sha256"], "rows": rows,
            "total": len(rows), "scored": sum(r["status"] == "scored" for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", type=Path, default=HERE/"config.json")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("generate")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--split", choices=["train", "development"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("analytic-guesses")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for command in ("evaluate", "correct"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", type=Path, required=True)
        p.add_argument("--predictions", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    if args.command == "plan":
        result = make_plan(read(args.config))
    elif args.command == "generate":
        result = generate(read(args.plan), args.split)
    elif args.command == "analytic-guesses":
        dataset = read(args.dataset)
        result = {c["case_id"]: initial_guess(np.asarray(c["reference"]["eta"])).tolist()
                  for c in dataset["cases"] if c["accepted"]}
    else:
        dataset, predictions = read(args.dataset), read(args.predictions)
        result = evaluate(dataset, predictions)
        if args.command == "correct":
            corrections = []
            for case, scored in zip(dataset["cases"], result["rows"]):
                if scored["status"] != "scored":
                    corrections.append({"case_id": case["case_id"], "status": scored["status"]})
                    continue
                ref = case["reference"]
                settings = dict(length=ref["length"], points=len(ref["eta"]), tol=ref["tol"])
                p = Parameters(**case["parameters"])
                # Both are new solves, including the ordinary-guess comparison.
                try:
                    ordinary = solve(p, **settings)
                    proposed = solve(p, guess=predictions[case["case_id"]], **settings)
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    corrections.append({"case_id": case["case_id"], "status": "correction_failed",
                                        "verified_correction": False,
                                        "error": {"type": type(exc).__name__, "message": str(exc)}})
                    continue
                truth = np.asarray(ref["state"])
                error = float(np.max(np.abs(np.asarray(proposed["state"])-truth)))
                corrections.append({"case_id": case["case_id"], "ordinary": ordinary,
                                    "proposed": proposed, "reference_agreement_max": error,
                                    "verified_correction": bool(proposed["accepted"] and error <= 1e-4)})
            result["corrections"] = corrections
            result["timing_scope"] = "solver_only_CPU; excludes neural inference/training/preparation; no speedup claim"
    write_new(args.output, result)
    print(json.dumps({"command": args.command, "output_written": True,
                      "accepted": result.get("accepted"), "attempted": result.get("attempted")}, allow_nan=False))


if __name__ == "__main__":
    main()
