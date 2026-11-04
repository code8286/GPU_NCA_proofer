"""Public-only audit of M2 certificate numbers, source identities and M3 metadata."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scientific_transfer.complete_m2 import verify_plan


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    base = ROOT/"docs/results/scientific_transfer_completion"
    for name, expected in read(base/"PROVENANCE.json").items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Public exact-copy checksum differs: "+name)
    m2 = base/"milestone2_complete"
    plan = verify_plan(m2)
    checks, replay = read(m2/"CHECKS.json"), read(m2/"REPLAY.json")
    assert checks["plan_sha256"] == plan["sha256"] == replay["plan_sha256"]
    assert hashlib.sha256((m2/"CHECKS.json").read_bytes()).hexdigest() == replay["checks_sha256"]
    assert checks["passed"] and replay["passed"] and checks["neural_updates"] == 0
    assert set(checks["cases"]) == set(plan["config"]["cases"])
    for case in checks["cases"].values():
        assert case["passed"] and case["reference"]["accepted"]
        assert [r["points"] for r in case["independent"]] == plan["config"]["spectral_points"]
        original = case["reference"]["reference"]
        for row in case["independent"]:
            numerical = row["solve"]
            a, b = np.array(original["state"]), np.array(numerical["state"])
            assert a.shape == b.shape == (7, 161) and np.isfinite(a).all() and np.isfinite(b).all()
            profile = float(np.max(np.abs(a[[1, 3, 5]]-b[[1, 3, 5]])))
            wall = max(abs(numerical["wall"][k]-original["wall"][k]) for k in numerical["wall"])
            assert profile == row["profile_error"] and wall == row["wall_error"]
            assert profile <= 1e-5 and wall <= 1e-5 and row["passed"]
            assert numerical["accepted"] and numerical["root_success"]
            assert numerical["collocation_residual_max"] <= 1e-7 and numerical["boundary_max"] <= 1e-7
            assert numerical["dense_defect_max"] <= 1e-4
    assert not read(base/"failed_first_attempt/CHECKS.json")["passed"]
    m3 = read(base/"milestone3_preparation/PLAN.json")
    gate = read(base/"milestone3_preparation/GATE.json")
    assert m3["eligibility"] == gate and not gate["allowed"]
    records = m3["registry"]
    assert [r["arm"] for r in records] == m3["config"]["arms"]
    for key in ("adapter_sha256", "language_updates", "language_tokens", "language_order", "language_schedule", "language_head_initial"):
        assert len({r[key] for r in records}) == 1
    for name, expected in m3["sources"].items():
        path = ROOT/name if name.startswith("transfer_v4/") else ROOT/"scientific_transfer"/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert m3["expected_per_arm"] == {"updates": 400, "query_tokens": 515200, "supervised_scalars": 1803200}
    print("PASS: public numerical decisions/errors/replay/source identities; retained failure; five-arm metadata and closed gate.")
    print("No private weights/data/cache/network or neural inference used; raw M3 dataset/checkpoint verification is separate.")


if __name__ == "__main__":
    main()
