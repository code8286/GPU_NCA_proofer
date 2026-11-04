"""Recompute fixed ID scores from the retained exploratory predictions."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scientific_transfer import training as original
from scientific_transfer.hybrid_v2 import HybridParameters, wall
from scientific_transfer.numeric_adapter import decision
from scientific_transfer.physics import boundary


def check(out):
    prep = original.ROOT / "runs/scientific_transfer/m3_preparation"
    parent = original.read(prep / "PLAN.json")
    config = parent["config"]
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    if original.sha(prep / "development.json") != parent["files"]["development.json"]:
        raise ValueError("Development references differ")
    references = {row["case_id"]: row for row in original.read(prep / "development.json")}
    checked = 0
    for record in parent["registry"]:
        folder = out / record["arm"]
        summary = original.read(folder / "SUMMARY.json")
        assert summary["initial_sha256"] == record["adaptation_initial_sha256"]
        assert summary["exposure"] == parent["expected_per_arm"]
        assert [point["step"] for point in summary["curve"]] == config["evaluation_updates"]
        updates = original.read(folder / f"updates_through_{config['updates']}.json")
        assert len(updates) == config["updates"]
        for step, update in enumerate(updates, 1):
            assert update["step"] == step and np.isfinite(update["loss"])
            assert update["lr"] == original.learning_rate(step, config)
        for point in summary["curve"]:
            predictions = original.read(folder / f"predictions_{point['step']}.json")
            checkpoint = torch.load(folder / f"step_{point['step']}.pt",
                                    map_location="cpu", weights_only=True)
            assert checkpoint["step"] == point["step"]
            assert checkpoint["plan_sha256"] == original.sha(out / "PLAN.json")
            slots = checkpoint["optimizer"]["state"]
            assert bool(slots) == bool(point["step"])
            assert all(int(slot["step"]) == point["step"] for slot in slots.values())
            core = {key.removeprefix("core."): value for key, value in checkpoint["model"].items()
                    if key.startswith("core.")}
            model = original.NumericAdapter(record["architecture"], core, config["adapter_seed"])
            model.load_state_dict(checkpoint["model"], strict=True)
            if point["step"] == 0:
                assert original.state_sha(model.state_dict()) == record["adaptation_initial_sha256"]
            _, reproduced = original.evaluate(model, list(references.values()), config)
            for case_id, state in reproduced.items():
                if predictions[case_id] is None:
                    assert not np.isfinite(state).all()
                else:
                    np.testing.assert_array_equal(state, predictions[case_id])
            rows = {row["case_id"]: row for row in point["cases"]}
            assert predictions.keys() == rows.keys() == references.keys()
            for case_id, case in references.items():
                state = predictions[case_id]
                row = rows[case_id]
                assert row["valid"] == (state is not None)
                if state is not None:
                    state = np.asarray(state)
                    reference = np.asarray(case["reference"]["state"])
                    assert state.shape == reference.shape and np.isfinite(state).all()
                    predicted_wall = wall(state, HybridParameters(**case["parameters"]))
                    expected = [
                        np.sqrt(np.mean((state[[1, 3, 5]] - reference[[1, 3, 5]]) ** 2)),
                        max(abs(value - case["reference"]["wall"][key])
                            for key, value in predicted_wall.items()),
                        max(abs(boundary(state[:, 0], state[:, -1]))),
                    ]
                    np.testing.assert_allclose(
                        [row[k] for k in ("profile_rmse", "wall_absolute_max", "boundary_max")],
                        expected, rtol=0, atol=1e-12)
                checked += 1
        assert summary["endpoint_competent"] == decision(summary["curve"][-1]["cases"], config)
        assert summary["first"] == original.crossing(summary["curve"], config)
        assert summary["sustained"] == original.crossing(summary["curve"], config, True)
    print(f"Replayed 25 checkpoints and {checked} ID predictions. Verified five crossing decisions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    check(parser.parse_args().out)
