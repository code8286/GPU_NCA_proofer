import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transfer_v4.model import Backbone
from transfer_v4.common import state_sha
from scientific_transfer import complete_m2
from scientific_transfer.hybrid_v2 import HybridParameters
from scientific_transfer.spectral_hybrid import solve
from scientific_transfer.numeric_adapter import NumericAdapter, decision, crossing
from scientific_transfer.training import eligibility, fit


class CompletionAndIntegrationTests(unittest.TestCase):
    def test_full_radiative_independent_discretization(self):
        result = solve(HybridParameters(phi_zinc=.01, phi_titania=.01,
                       radiation=.2, temperature_ratio=1.2, magnetic=.2, rotation=.2, buoyancy=.1))
        self.assertTrue(result["accepted"], {k: v for k, v in result.items() if k not in ("state", "eta")})

    def test_pure_full_system_has_no_spurious_off_node_mode(self):
        for n in (49, 65):
            result = solve(HybridParameters(), points=n)
            self.assertTrue(result["accepted"], {k: v for k, v in result.items() if k not in ("state", "eta")})
            self.assertLess(np.max(np.abs(np.array(result["state"])[1])), 1.1)

    def test_completion_plan_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            complete_m2.freeze(out)
            complete_m2.verify_plan(out)
            plan = json.loads((out/"PLAN.json").read_text())
            plan["config"]["spectral_points"] = [9]
            (out/"PLAN.json").write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                complete_m2.verify_plan(out)

    def test_replay_ignores_only_timers(self):
        a = {"seconds": 1, "cases": [{"seconds": 2, "state": [1, 2]}]}
        b = {"seconds": 9, "cases": [{"seconds": 4, "state": [1, 2]}]}
        self.assertEqual(complete_m2.without_timing(a), complete_m2.without_timing(b))
        b["cases"][0]["state"][0] += 1
        self.assertNotEqual(complete_m2.without_timing(a), complete_m2.without_timing(b))

    def test_adapter_matching_and_core_preservation_and_gradient(self):
        architecture = {"layers": 1, "width": 16, "heads": 2}
        torch.manual_seed(7)
        core = Backbone(architecture).state_dict()
        a = NumericAdapter(architecture, core, 13)
        other = {k: v+0.001 for k, v in core.items()}
        b = NumericAdapter(architecture, other, 13)
        heads = lambda m: {k: v for k, v in m.state_dict().items() if not k.startswith("core.")}
        self.assertEqual(state_sha(heads(a)), state_sha(heads(b)))
        self.assertEqual(state_sha(a.core.state_dict()), state_sha(core))
        self.assertNotEqual(state_sha(a.core.state_dict()), state_sha(b.core.state_dict()))
        prediction = a(torch.ones(3, 9))
        self.assertEqual(tuple(prediction.shape), (3, 7))
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters()))
        initial_core = state_sha(a.core.state_dict())
        optimizer = torch.optim.AdamW(a.parameters(), lr=1e-3)
        optimizer.step()
        self.assertNotEqual(state_sha(a.core.state_dict()), initial_core)

    def test_complete_coverage_invalids_and_censoring(self):
        c = dict(require_all_development_cases=True, development_cases=2,
                 id_profile_rmse_max=.02, id_wall_absolute_max=.05, id_boundary_max=.05, updates=400)
        rows = [dict(case_id=str(i), valid=True, profile_rmse=.01, wall_absolute_max=.01, boundary_max=.01) for i in range(2)]
        self.assertTrue(decision(rows, c))
        self.assertFalse(decision(rows[:1], c))
        wrong = copy.deepcopy(rows)
        wrong[0]["valid"] = False
        self.assertFalse(decision(wrong, c))
        curve = [{"step": 100, "query_tokens": 100, "cases": rows},
                 {"step": 400, "query_tokens": 400, "cases": wrong}]
        self.assertEqual(crossing(curve, c)["step"], 100)
        self.assertFalse(crossing(curve, c, True)["reached"])
        self.assertIsNone(crossing(curve[1:], c)["query_tokens"])

    def test_real_retained_readiness_rejects_before_fit_marker(self):
        gate = eligibility()
        self.assertFalse(gate["allowed"])
        self.assertTrue(all(not r["passed"] for r in gate["cohort_gates"]))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch("scientific_transfer.training.verify_plan", return_value={"eligibility": gate}):
                with self.assertRaisesRegex(ValueError, "fail stronger readiness"):
                    fit(out)
            self.assertFalse((out/"FIT_STARTED.json").exists())


if __name__ == "__main__":
    unittest.main()
