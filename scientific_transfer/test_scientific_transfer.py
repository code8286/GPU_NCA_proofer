import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scientific_transfer.metrics import score
from scientific_transfer.physics import Parameters, boundary, initial_guess, rhs, solve
from scientific_transfer.run import evaluate, generate, make_plan, reference_case, validate_plan, write_new


class ScientificTransferTests(unittest.TestCase):
    def config(self):
        return json.loads(Path(__file__).with_name("config.json").read_text())

    def test_parameters_reject_nonphysical(self):
        for kwargs in ({"prandtl": 0}, {"magnetic": -1}, {"rotation": float("nan")}):
            with self.assertRaises(ValueError):
                Parameters(**kwargs)

    def test_constant_freestream_satisfies_interior_equations(self):
        x = np.linspace(0, 2, 11)
        y = np.zeros((7, len(x)))
        y[0], y[1] = x, 1
        expected = np.zeros_like(y)
        expected[0] = 1
        np.testing.assert_allclose(rhs(x, y, Parameters(magnetic=0.4)), expected)
        self.assertGreater(np.max(np.abs(boundary(y[:, 0], y[:, -1]))), 0)

    def test_magnetic_and_rotation_terms(self):
        x = np.linspace(0, 2, 11)
        y = initial_guess(x)
        difference = rhs(x, y, Parameters(magnetic=2, rotation=3))-rhs(x, y, Parameters())
        np.testing.assert_allclose(difference[2], 4*(y[1]-1)-3*y[3]**2)
        np.testing.assert_allclose(difference[4], 4*y[3])

    def test_plan_deterministic_disjoint_and_tamper_checked(self):
        plan = make_plan(self.config())
        self.assertEqual(plan, make_plan(self.config()))
        self.assertEqual(len({r["case_id"] for r in plan["rows"]}), 20)
        for row in plan["rows"]:
            if row["split"] == "extrapolation":
                self.assertGreater(row["parameters"]["magnetic"], 1)
        validate_plan(plan)
        plan["config"]["seed"] += 1
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_reserved_generation_and_training_blocked(self):
        plan = make_plan(self.config())
        with self.assertRaises(ValueError):
            generate(plan, "assessment")
        config = self.config()
        config["training_enabled"] = True
        with self.assertRaises(ValueError):
            make_plan(config)

    def test_reference_convergence_and_far_boundary(self):
        row = {"case_id": "baseline", "parameters": {}, "split": "development"}
        case = reference_case(row, self.config())
        self.assertTrue(case["accepted"], {k: case[k] for k in
                        ("profile_agreement_max", "wall_agreement_max", "accepted")})
        y = np.asarray(case["reference"]["state"])
        self.assertLess(np.max(np.abs(boundary(y[:, 0], y[:, -1]))), 1e-6)
        self.assertTrue((np.diff(y[5]) <= 1e-6).all())
        self.assertGreater(case["reference"]["wall"]["minus_theta_p0"], 0)

    def test_failed_solver_not_accepted(self):
        result = solve(Parameters(), points=5, max_nodes=5, tol=1e-10)
        self.assertFalse(result["accepted"])
        self.assertNotEqual(result["status"], 0)

    def test_metrics_detect_boundary_wall_and_invalid_guesses(self):
        x = np.linspace(0, 10, 161)
        y = initial_guess(x)
        metrics = score(x, y, y, Parameters())
        self.assertEqual(metrics["profile_rmse"], 0)
        wrong = y.copy()
        wrong[2, 0] += 1
        self.assertEqual(score(x, wrong, y, Parameters())["wall_absolute_errors"]["fpp0"], 1)
        wrong[5, 0] += 1
        self.assertGreater(score(x, wrong, y, Parameters())["boundary_max"], 0.5)
        wrong[0, 0] = np.nan
        self.assertFalse(score(x, wrong, y, Parameters())["valid"])
        self.assertFalse(score(x, [["bad"]], y, Parameters())["valid"])

    def test_evaluation_missing_failure_and_unknown_coverage(self):
        data = {"plan_sha256": "test", "cases": [
            {"case_id": "failure", "accepted": False},
            {"case_id": "missing", "accepted": True}]}
        result = evaluate(data, {})
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["scored"], 0)
        with self.assertRaises(ValueError):
            evaluate(data, {"unknown": []})

    def test_evidence_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"evidence.json"
            write_new(p, {"original": True})
            with self.assertRaises(FileExistsError):
                write_new(p, {})
            self.assertEqual(json.loads(p.read_text()), {"original": True})


if __name__ == "__main__":
    unittest.main()
