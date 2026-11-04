"""Freeze, check and independently replay milestone2's declared completion contract."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from .hybrid_v2 import HybridParameters, verify_case
from .spectral_hybrid import solve as spectral_solve
from .run import digest, write_new

HERE = Path(__file__).resolve().parent
BOUND = ["physics.py", "materials.py", "hybrid_v2.py", "spectral_hybrid.py",
         "independent_reference.py", "complete_m2.py", "completion_config.json",
         "DERIVATION_V2.md", "MILESTONE2_COMPLETION.md", "run.py"]


def hashes():
    return {n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in BOUND}


def freeze(out):
    c = json.loads((HERE/"completion_config.json").read_text())
    payload = {"config": c, "model_sources": hashes(),
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    write_new(out/"PLAN.json", {**payload, "sha256": digest(payload)})


def verify_plan(out):
    p = json.loads((out/"PLAN.json").read_text())
    if p["sha256"] != digest({k: v for k, v in p.items() if k != "sha256"}) or p["model_sources"] != hashes():
        raise ValueError("Completion plan/model sources changed")
    return p


def calculate(plan):
    tick = time.perf_counter()
    c = plan["config"]
    cases = {}
    for name, parameters in c["cases"].items():
        p = HybridParameters(**parameters)
        try:
            reference = verify_case(p)
            comparisons = []
            for n in c["spectral_points"]:
                independent = spectral_solve(p, points=n)
                error = float(np.max(np.abs(np.array(independent["state"])[[1, 3, 5]]-
                                            np.array(reference["reference"]["state"])[[1, 3, 5]])))
                wall_error = max(abs(independent["wall"][k]-reference["reference"]["wall"][k])
                                 for k in independent["wall"])
                comparisons.append({"points": n, "solve": independent, "profile_error": error,
                                    "wall_error": wall_error, "passed": bool(independent["accepted"] and
                                    error <= c["cross_method_profile_absolute"] and wall_error <= c["cross_method_wall_absolute"])})
            cases[name] = {"reference": reference, "independent": comparisons,
                           "passed": bool(reference["accepted"] and all(r["passed"] for r in comparisons))}
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            cases[name] = {"passed": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
    return {"plan_sha256": plan["sha256"], "cases": cases,
            "passed": all(v["passed"] for v in cases.values()), "neural_updates": 0,
            "seconds": time.perf_counter()-tick,
            "scope": "milestone2 numerical completion of declared steady_hybrid_v2; manuscript audit remains partial"}


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "check", "replay"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output)
        print("Frozen completion contract before numerical checks")
        return
    plan = verify_plan(args.output)
    result = calculate(plan)
    if args.command == "replay":
        original = json.loads((args.output/"CHECKS.json").read_text())
        if without_timing(original) != without_timing(result):
            raise ValueError("Numerical replay differs (timers excepted)")
        write_new(args.output/"REPLAY.json", {"passed": True, "plan_sha256": plan["sha256"],
                  "checks_sha256": hashlib.sha256((args.output/"CHECKS.json").read_bytes()).hexdigest(),
                  "scope": "separate invocation same-code numerical replay; timers excluded"})
    else:
        write_new(args.output/"CHECKS.json", result)
    print(json.dumps({"passed": result["passed"], "cases": {k: v["passed"] for k, v in result["cases"].items()}}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
