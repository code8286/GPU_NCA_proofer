"""Regression coverage for independent-stage scientific controls."""
import copy
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from transfer_v2.common import effective_tokens, load_config, validate_config
from transfer_v2.model import LM, language_model
from transfer_v2.nca import (complexity, convolution_logits, encode_frames, generate_split, lattice,
                             message_logits, sample_rules, shuffled, targets)
from transfer_v2.report import first_hit
from transfer_v2.training import optimizer_for, quality_gate


ROOT = Path(__file__).resolve().parents[1]


class TransferV2Controls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.c = load_config(ROOT/"configs/transfer_v2_smoke.json")

    def test_real_configuration_has_independent_budgets(self):
        c = load_config(ROOT/"configs/transfer_v2.json")
        self.assertEqual(c["ppt"]["steps"], 1000)
        self.assertEqual(c["language"]["steps"], 2000)
        self.assertEqual(effective_tokens(c), 4096)
        self.assertEqual(c["nca"]["states"], 10)
        self.assertTrue(c["gate"]["enforce"])
        self.assertFalse(c["text"]["evaluate_test"])

    def test_default_model_accepts_real_bpe_vocabulary_and_backpropagates(self):
        c = load_config(ROOT/"configs/transfer_v2.json")
        model = language_model(c, 50257, 17)
        logits = model(torch.zeros((1, 16), dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (1, 16, 50257))
        F.cross_entropy(logits.reshape(-1, 50257), torch.zeros(16, dtype=torch.long)).backward()
        self.assertTrue(torch.isfinite(model.core.blocks[0].qkv.weight.grad).all())
        self.assertGreater(sum(p.numel() for p in model.parameters()), 10_000_000)

    def test_development_cannot_bypass_synthetic_gate(self):
        c = load_config(ROOT/"configs/transfer_v2.json")
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
        with self.assertRaisesRegex(RuntimeError, "no silent threshold relaxation"):
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


if __name__ == "__main__":
    unittest.main()
