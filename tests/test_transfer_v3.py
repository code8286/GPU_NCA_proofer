"""Regression coverage for independent-stage scientific controls."""
import copy
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from transfer_v3.common import effective_tokens, load_config, validate_config
from transfer_v3.model import LM, language_model
from transfer_v3.nca import (complexity, convolution_logits, encode_frames, lattice,
                             message_logits, sample_rules, shuffled, targets)
from transfer_v3.report import first_hit
from transfer_v3.training import optimizer_for, quality_gate


ROOT = Path(__file__).resolve().parents[1]


from transfer_v3.graph import rewire, generate_paired, paired_rollout, validate_topology


def generate_split(c, count, split, device):
    edges, _ = rewire(c['nca']['grid'], c['graph']['rewire_fraction'], 19)
    arrays, rules, stats = generate_paired(c, count, split, 17, device, edges)
    return arrays['nca'], rules, stats


class TransferV3Controls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        (ROOT/"tmp").mkdir(exist_ok=True)
        self.c = load_config(ROOT/"configs/transfer_v3_smoke.json")

    def test_real_configuration_has_independent_budgets(self):
        c = load_config(ROOT/"configs/transfer_v3.json")
        self.assertEqual(c["ppt"]["steps"], 1000)
        self.assertEqual(c["language"]["steps"], 2000)
        self.assertEqual(effective_tokens(c), 4096)
        self.assertEqual(c["nca"]["states"], 10)
        self.assertTrue(c["gate"]["enforce"])
        self.assertFalse(c["text"]["evaluate_test"])

    def test_default_model_accepts_real_bpe_vocabulary_and_backpropagates(self):
        c = load_config(ROOT/"configs/transfer_v3.json")
        model = language_model(c, 50257, 17)
        logits = model(torch.zeros((1, 16), dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (1, 16, 50257))
        F.cross_entropy(logits.reshape(-1, 50257), torch.zeros(16, dtype=torch.long)).backward()
        self.assertTrue(torch.isfinite(model.core.blocks[0].qkv.weight.grad).all())
        self.assertGreater(sum(p.numel() for p in model.parameters()), 10_000_000)

    def test_development_cannot_bypass_synthetic_gate(self):
        c = load_config(ROOT/"configs/transfer_v3.json")
        c["gate"]["enforce"] = False
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_message_passing_matches_categorical_spatial_convolution(self):
        g = torch.Generator().manual_seed(17)
        rules = sample_rules(3, 10, g, "cpu")
        x = torch.randint(10, (3, 12*12), generator=g)
        a = message_logits(x, torch.tensor(lattice(12)), rules, 10)
        b = convolution_logits(x, rules, 12, 10)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a.argmax(-1), b.argmax(-1))

    def test_categorical_patch_encoding_is_bijective(self):
        ids = torch.arange(10000)
        cells = torch.stack([(ids//10**j) % 10 for j in range(4)], -1)[:, None]
        encoded = encode_frames(cells, 2, 10)
        torch.testing.assert_close(encoded[:, 1], ids)
        self.assertTrue(torch.all(encoded[:, 0] == 10000))
        self.assertTrue(torch.all(encoded[:, 2] == 10001))

    def test_initial_context_mask_and_patch_only_metric(self):
        frames = torch.randint(10, (2, 3, 144))
        sequence = encode_frames(frames, 12, 10)[:, :77]
        y = targets(sequence, self.c)
        self.assertTrue(torch.all(y[:, :37] == -100))
        self.assertTrue(torch.all(y[:, 37:] >= 0))
        patches = targets(sequence, self.c, patch_only=True)
        self.assertEqual(int((patches != -100).sum()), 72)
        self.assertTrue(torch.all(patches[patches != -100] < 10000))

    def test_gzip_score_ignores_markers_and_integer_input_width(self):
        x = encode_frames(torch.randint(10, (2, 10, 144)), 12, 10).numpy()
        changed = x.copy().reshape(2, 10, 38)
        changed[:, :, 0] = 32000
        changed[:, :, -1] = 32001
        np.testing.assert_array_equal(complexity(x.astype(np.uint16), self.c),
                                      complexity(changed.reshape(2, -1).astype(np.int64), self.c))

    def test_shuffled_control_preserves_counts_and_marker_locations(self):
        x = encode_frames(torch.randint(10, (8, 3, 144)), 12, 10).numpy().astype(np.uint16)
        y = shuffled(x, self.c, 123)
        np.testing.assert_array_equal(np.sort(x, axis=1), np.sort(y, axis=1))
        np.testing.assert_array_equal(x >= 10000, y >= 10000)
        self.assertFalse(np.array_equal(x, y))

    def test_rule_split_generation_is_reproducible_and_disjoint(self):
        a, ra, sa = generate_split(self.c, 3, "train", "cpu")
        b, rb, sb = generate_split(self.c, 3, "train", "cpu")
        _, rv, sv = generate_split(self.c, 3, "validation", "cpu")
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ra["perception"], rb["perception"])
        self.assertEqual(sa["rule_ids"], sb["rule_ids"])
        self.assertFalse(set(sa["rule_ids"]) & set(sv["rule_ids"]))
        self.assertFalse(np.array_equal(ra["perception"], rv["perception"]))

    def test_filter_never_silently_relaxes_threshold(self):
        c = copy.deepcopy(self.c)
        c["nca"].update(filter_low=2., filter_high=3., max_candidates=8)
        with self.assertRaisesRegex(RuntimeError, "no relaxation"):
            generate_split(c, 1, "train", "cpu")

    def test_backbone_pairing_is_independent_of_vocabulary(self):
        a = LM(self.c["model"], 10002, 17, 50)
        b = LM(self.c["model"], 256, 17, 123)
        self.assertEqual(a.core_hash(), b.core_hash())

    def test_transfer_copies_backbone_and_reinitializes_common_head(self):
        ppt = LM(self.c["model"], 10002, 17, 50)
        with torch.no_grad():
            next(ppt.core.parameters()).add_(.1)
        base = language_model(self.c, 256, 17)
        transferred = language_model(self.c, 256, 17, ppt.core.state_dict())
        self.assertEqual(base.head_hash(), transferred.head_hash())
        self.assertNotEqual(base.core_hash(), transferred.core_hash())
        self.assertEqual(transferred.core_hash(), ppt.core_hash())
        self.assertTrue(all(p.requires_grad for p in transferred.parameters()))

    def test_language_optimizer_is_fresh_after_ppt(self):
        ppt = LM(self.c["model"], 10002, 17, 50)
        opt = optimizer_for(ppt, self.c["ppt"], True)
        logits = ppt(torch.zeros((2, 8), dtype=torch.long))
        F.cross_entropy(logits.reshape(-1, 10002), torch.zeros(16, dtype=torch.long)).backward()
        opt.step()
        self.assertTrue(opt.state)
        transferred = language_model(self.c, 256, 17, ppt.core.state_dict())
        language_opt = optimizer_for(transferred, self.c["language"], False)
        self.assertEqual(len(language_opt.state), 0)

    def test_causal_rope_model_cannot_see_future(self):
        model = language_model(self.c, 256, 17).eval()
        x = torch.randint(256, (2, 16))
        y = x.clone()
        y[:, 8:] = torch.randint(256, (2, 8))
        torch.testing.assert_close(model(x)[:, :8], model(y)[:, :8], rtol=0, atol=1e-6)

    def test_gate_requires_prediction_beyond_static_controls(self):
        controls = {"unigram_patch_nll": 7., "copy_mixture_patch_nll": 6.5}
        self.assertFalse(quality_gate({"patch_nll": 9.}, {"patch_nll": 7.}, controls, self.c)["passed"])
        self.assertTrue(quality_gate({"patch_nll": 9.}, {"patch_nll": 6.}, controls, self.c)["passed"])

    def test_unreached_language_reference_is_censored(self):
        curve = [{"validation_nll": 8., "language_step": 0}, {"validation_nll": 7., "language_step": 100}]
        self.assertIsNone(first_hit(curve, 6.))
        self.assertEqual(first_hit(curve, 7.)["language_step"], 100)

    def test_rewiring_preserves_degrees_slots_and_uniqueness(self):
        edges, stats = rewire(12, .25, 42)
        validate_topology(edges, 12)
        self.assertGreaterEqual(stats["actual_changed_fraction"], .25)
        np.testing.assert_array_equal(edges, rewire(12, .25, 42)[0])
        self.assertFalse(np.array_equal(edges, rewire(12, .25, 43)[0]))
        self.assertEqual(stats["actual_changed_fraction"], np.mean(edges[:, [0,1,2,3,5,6,7,8]] != lattice(12)[:, [0,1,2,3,5,6,7,8]]))

    def test_zero_rewiring_is_the_exact_grid(self):
        edges, stats = rewire(12, 0., 42)
        np.testing.assert_array_equal(edges, lattice(12))
        self.assertEqual(stats["actual_changed_fraction"], 0.)
        self.assertEqual(stats["sha256"], stats["grid_sha256"])

    def test_invalid_graph_is_rejected(self):
        edges = lattice(12)
        edges[0, 0] = edges[0, 1]
        with self.assertRaises(ValueError):
            validate_topology(edges, 12)

    def test_graph_message_update_respects_consistent_node_relabeling(self):
        edges, _ = rewire(12, .25, 42)
        perm = np.random.default_rng(17).permutation(144)
        inverse = np.argsort(perm)
        relabeled = inverse[edges[perm]]
        g = torch.Generator().manual_seed(17)
        rules = sample_rules(2, 10, g, "cpu")
        state = torch.randint(10, (2, 144), generator=g)
        original = message_logits(state, torch.tensor(edges), rules, 10)
        changed = message_logits(state[:, perm], torch.tensor(relabeled), rules, 10)
        torch.testing.assert_close(changed, original[:, perm], atol=1e-6, rtol=1e-5)

    def test_common_random_numbers_give_identical_zero_rewire_trajectories(self):
        c = copy.deepcopy(self.c)
        c["nca"]["temperature"] = .1
        generator = torch.Generator().manual_seed(71)
        rules = sample_rules(3, 10, generator, "cpu")
        grid, graph = paired_rollout(rules, c, 4, generator, "cpu", lattice(12))
        torch.testing.assert_close(grid, graph, rtol=0, atol=0)

    def test_rule_filter_uses_the_intersection_not_union(self):
        edges, _ = rewire(12, .25, 42)
        scores_grid = np.array([.6, .6, .2, .2, .2, .2, .2, .2])
        scores_graph = np.array([.6, .2, .6, .2, .2, .2, .2, .2])
        c = copy.deepcopy(self.c)
        c["nca"].update(filter_low=.5, filter_high=1., max_candidates=8)
        with patch("transfer_v3.graph.complexity", side_effect=[scores_grid, scores_graph, np.array([.6]), np.array([.6])]):
            arrays, rules, stats = generate_paired(c, 1, "train", 17, "cpu", edges)
        self.assertEqual(stats["accepted"], 1)
        self.assertTrue(stats["rule_ids"][0].endswith(":0"))
        self.assertEqual(arrays["nca"].shape, arrays["graph"].shape)
        g = torch.Generator().manual_seed(c["nca"]["data_seed"] + 100003*17)
        candidate = sample_rules(8, 10, g, "cpu")
        np.testing.assert_array_equal(rules["perception"][0], candidate["perception"][0].numpy())

    def test_replicates_change_rules_and_topology(self):
        a, ga = rewire(12, .25, 17)
        b, gb = rewire(12, .25, 29)
        _, rules_a, stats_a = generate_paired(self.c, 2, "train", 17, "cpu", a)
        _, rules_b, stats_b = generate_paired(self.c, 2, "train", 29, "cpu", b)
        self.assertNotEqual(ga["sha256"], gb["sha256"])
        self.assertFalse(np.array_equal(rules_a["perception"], rules_b["perception"]))
        self.assertFalse(set(stats_a["rule_ids"]) & set(stats_b["rule_ids"]))

    def test_graph_failure_alone_blocks_language_stage(self):
        from transfer_v3.run import ppt_stage, language_stage
        c = copy.deepcopy(self.c)
        c["gate"]["enforce"] = True
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as name:
            def fake(c, arm, seed, *args):
                return {"arm": arm, "seed": seed, "gate": {"passed": arm != "graph_nca"}}
            with patch("transfer_v3.run.train_ppt", side_effect=fake):
                self.assertFalse(ppt_stage(c, name, {"17": {}}, "manifest", "cpu"))
            status = json.loads((Path(name)/"status.json").read_text())
            self.assertEqual(status["structured_seeds_failing_gate"], [{"arm": "graph_nca", "seed": 17}])
            with self.assertRaises(ValueError):
                language_stage(c, name, {}, {}, "manifest", "cpu")

    def test_graph_vs_grid_failure_is_censored_in_primary_report(self):
        from transfer_v3.common import ARMS
        from transfer_v3.report import report
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as name:
            root = Path(name)
            (root/"status.json").write_text(json.dumps({"state": "complete"}))
            for arm, final in zip(ARMS, (5., 4.5, 4.6, 4.7, 4.6)):
                folder = root/f"{arm}_17"
                folder.mkdir()
                (folder/"language_summary.json").write_text(json.dumps({"arm": arm, "final_validation_nll": final}))
                rows = [{"validation_nll": nll, "language_step": step, "language_tokens": step*152,
                         "language_train_seconds": step, "language_seconds": step+1,
                         "total_training_tokens": step*152+608, "cumulative_seconds": step+5}
                        for step, nll in ((0, 10.), (6, final))]
                (folder/"language_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            with patch("transfer_v3.report.plot"), patch("transfer_v3.report.plot_ppt"):
                result = report(root, self.c)
            primary = result["treatment_comparisons"][0]
            self.assertEqual(primary["reference_arm"], "grid_nca")
            self.assertEqual(primary["seeds_reaching_reference"], 0)
            self.assertIsNone(primary["geomean_language_tokens_speedup_all_seeds"])
            self.assertAlmostEqual(primary["mean_final_validation_nll_delta"], .1)


if __name__ == "__main__":
    unittest.main()
