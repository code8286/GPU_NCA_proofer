"""Freeze and run numerical milestones 1-2; no training/OOD entry point."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .hybrid_v2 import HybridParameters, verify_case
from .independent_reference import homann
from .materials import ratios
from .run import digest, sources, write_new

HERE = Path(__file__).resolve().parent


def freeze(out):
    config = json.loads((HERE/"milestone2_config.json").read_text())
    if config["training_enabled"] or config["manuscript_replication_enabled"]:
        raise ValueError("Only declared numerical development checks are enabled")
    payload = {"config": config, "sources": sources(),
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__},
               "scope": "new local CPU numerical checks, no neural updates"}
    write_new(out/"PLAN.json", {**payload, "sha256": digest(payload)})
    return payload


def check(out):
    plan = json.loads((out/"PLAN.json").read_text())
    if plan["sha256"] != digest({k: v for k, v in plan.items() if k != "sha256"}) or plan["sources"] != sources():
        raise ValueError("Frozen plan/source mismatch")
    tick = time.perf_counter()
    c = plan["config"]
    cases = {}
    for name, parameters in c["cases"].items():
        try:
            cases[name] = verify_case(HybridParameters(**parameters))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            cases[name] = {"accepted": False, "error": str(exc)}
    settings = c["independent_reference"]
    spectral = [homann(n, settings["length"], settings["prandtl"]) for n in settings["points"]]
    base = cases["pure"]["reference"]["wall"]
    comparisons = [{"fpp0_error": abs(s["fpp0"]-base["mu_ratio_fpp0"]),
                    "thermal_error": abs(s["minus_theta_p0"]-base["minus_total_conductivity_theta_p0"]),
                    "published_rounded_error": abs(s["fpp0"]-settings["published_homann_fpp0_rounded"])} for s in spectral]
    limit = settings["wall_agreement_absolute"]
    passed = bool(all(v["accepted"] for v in cases.values()) and
                  all(s["accepted"] for s in spectral) and
                  all(max(r.values()) <= limit for r in comparisons) and
                  all(v == 1 for v in ratios().values()))
    receipt = {"plan_sha256": plan["sha256"], "model_id": c["model_id"], "passed": passed,
               "cases": cases, "independent_reference": spectral, "comparisons": comparisons,
               "zero_loading_ratios": ratios(), "seconds_including_reference_checks": time.perf_counter()-tick,
               "neural_updates": 0, "manuscript_replication": False,
               "assessment_or_extrapolation_generated": False}
    write_new(out/"CHECKS.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "check"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.output) if args.command == "freeze" else check(args.output)
    print(json.dumps({k: result[k] for k in ("passed", "comparisons") if k in result}))
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
