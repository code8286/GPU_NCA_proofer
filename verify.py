"""Re-load saved checkpoints and independently reproduce held-out test loss."""
import argparse
import json
from pathlib import Path

import torch

import experiment as e


def verify(root, data_root, device="cpu"):
    root = Path(root)
    env = json.loads((root / "environment.json").read_text())
    c = env["config"]
    data, _ = e.load_data(data_root, c)
    tests = e.eval_batches(data["test"], c, device)
    seen = {}
    records = []
    for folder in sorted(root.iterdir()):
        if not (folder / "summary.json").exists():
            continue
        summary = json.loads((folder / "summary.json").read_text())
        assert e.digest(folder / "model.pt") == summary["model_sha256"], "Checkpoint checksum mismatch"
        saved = torch.load(folder / "model.pt", map_location=device, weights_only=True)
        assert saved["config"] == c
        assert saved["arm"] == summary["arm"] and saved["seed"] == summary["seed"]
        assert saved["initial_sha256"] == summary["initial_sha256"]
        assert summary["total_tokens"] == c["total_steps"]*c["batch"]*c["context"]
        if summary["seed"] in seen:
            assert summary["initial_sha256"] == seen[summary["seed"]], "Unpaired initialization"
        seen[summary["seed"]] = summary["initial_sha256"]
        model = e.LM(c).to(device)
        model.load_state_dict(saved["model"], strict=True)
        loss = e.evaluate(model, tests)
        delta = abs(loss-summary["test_nll"])
        if delta > 1e-4:
            raise AssertionError(f"Test loss mismatch: {folder.name}, delta={delta}")
        records.append({"arm": saved["arm"], "seed": saved["seed"],
                        "reproduced_test_nll": loss, "absolute_error": delta,
                        "checkpoint_sha256_valid": True})
    if not records:
        raise ValueError("No completed checkpoints found")
    result = {"device": device, "checkpoints_verified": len(records), "records": records,
              "limitation": "Verification establishes artifact consistency, not a scientific speedup."}
    e.dump(root / "verification.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/pilot")
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    torch.set_num_threads(4)
    verify(a.out, a.data, a.device)
