"""Scientific validity checks. Run: python -m unittest discover -s tests -v."""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

import experiment as e


class ScientificChecks(unittest.TestCase):
    def setUp(self):
        self.c = e.load_config("configs/smoke.json")
        e.seed_all(123)

    def test_degree_matched_graph_without_self_or_duplicate_edges(self):
        n = 64
        lattice = e.neighbors(8)
        graph = e.neighbors(8, np.random.default_rng(10), .25)
        self.assertGreater(np.mean(graph != lattice), .05)
        for relation in graph.T:
            np.testing.assert_array_equal(np.sort(relation), np.arange(n))
        for node, row in enumerate(graph):
            self.assertEqual(len(set(row)), 8)
            self.assertNotIn(node, row)
        np.testing.assert_array_equal(graph, e.neighbors(8, np.random.default_rng(10), .25))

    def test_lattice_message_passing_equals_convolution(self):
        # Proves the grid reference IS an NCA, and that writing it as a GNN
        # alone does not constitute a different scientific treatment.
        side, hidden = 8, 16
        x = torch.randint(2, (1, side*side))
        w1, b1 = torch.randn(1, 9, hidden), torch.randn(1, hidden)
        w2, b2 = torch.randn(1, hidden, 2), torch.randn(1, 2)
        edges = torch.tensor(e.neighbors(side))
        graph = e.graph_step(x, edges, w1, b1, w2, b2)
        # graph input order is self, then the 8 row-major offsets.
        slots = [1, 2, 3, 4, 0, 5, 6, 7, 8]
        kernel = w1[0, slots].T.reshape(hidden, 1, 3, 3)
        field = (x.float()*2-1).reshape(1, 1, side, side)
        z = F.relu(F.conv2d(F.pad(field, (1, 1, 1, 1), mode="circular"), kernel, b1[0]))
        logits = F.conv2d(z, w2[0].T.reshape(2, hidden, 1, 1), b2[0])
        conv = logits.argmax(1).reshape(1, -1)
        torch.testing.assert_close(graph, conv)

    def test_binary_patch_vocabulary_is_bijective(self):
        states = torch.tensor([[(i >> j) & 1 for j in range(4)] for i in range(16)])[:, None, :]
        encoded = e.patch_tokens(states, 2)
        torch.testing.assert_close(encoded[:, 1], torch.arange(256, 272))
        self.assertTrue(torch.all(encoded[:, 0] == 272))
        self.assertTrue(torch.all(encoded[:, 2] == 273))

    def test_shuffle_preserves_counts_and_boundaries_but_changes_order(self):
        a = np.tile(np.r_[272, np.arange(256, 272), 273], (8, 4)).astype(np.uint16)
        b = e.shuffle_trajectories(a, 42)
        np.testing.assert_array_equal(np.sort(a, axis=1), np.sort(b, axis=1))
        np.testing.assert_array_equal(a >= 272, b >= 272)
        self.assertFalse(np.array_equal(a, b))

    def test_batches_never_cross_trajectory_boundaries(self):
        c = dict(self.c, context=8, batch=64)
        a = np.stack([np.full(40, i) for i in range(5)])
        x, y = e.sample_batch(a, c, np.random.default_rng(1), "cpu")
        self.assertTrue(torch.all(x == x[:, :1]))
        self.assertTrue(torch.all(y == x))

    def test_causal_attention_cannot_see_future_tokens(self):
        m = e.LM(self.c).eval()
        x = torch.randint(256, (2, 16))
        y = x.clone()
        y[:, 8:] = torch.randint(256, (2, 8))
        torch.testing.assert_close(m(x)[:, :8], m(y)[:, :8], rtol=0, atol=1e-6)

    def test_identical_initialization_and_gradient_update(self):
        e.seed_all(17)
        a = e.LM(self.c)
        e.seed_all(17)
        b = e.LM(self.c)
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        before = a.blocks[0].qkv.weight.detach().clone()
        opt = torch.optim.AdamW(a.parameters(), lr=.01)
        logits = a(torch.randint(256, (2, 16)))
        F.cross_entropy(logits.reshape(-1, e.VOCAB), torch.randint(256, (32,))).backward()
        opt.step()
        self.assertFalse(torch.equal(before, a.blocks[0].qkv.weight))

    def test_embedding_reset_preserves_backbone_and_weight_tying(self):
        m = e.LM(self.c)
        emb = m.token.weight.detach().clone()
        with torch.no_grad():
            m.token.weight.add_(1)
            m.blocks[0].qkv.weight.add_(1)
        backbone = m.blocks[0].qkv.weight.detach().clone()
        with torch.no_grad():
            m.token.weight.copy_(emb)
        torch.testing.assert_close(m.blocks[0].qkv.weight, backbone)
        self.assertIs(m.head.weight, m.token.weight)

    def test_unreached_target_is_censored(self):
        rows = [{"stage": "pre", "val_nll": 1., "total_tokens": 10},
                {"stage": "target_start", "val_nll": 3., "total_tokens": 20},
                {"stage": "target", "val_nll": 2., "total_tokens": 30}]
        self.assertIsNone(e.first_hit(rows, 1.5, "total_tokens"))
        self.assertEqual(e.first_hit(rows, 2.5, "total_tokens"), 30)

    def test_text_baseline_can_reach_target_in_first_stage(self):
        rows = [{"arm": "scratch", "stage": "pre", "val_nll": 2., "total_tokens": 10},
                {"arm": "scratch", "stage": "target", "val_nll": 1.5, "total_tokens": 30}]
        self.assertEqual(e.first_hit(rows, 2.5, "total_tokens"), 10)

    def test_generation_reproducible_and_valid(self):
        a, stats, edges = e.generate(self.c, "graph", "cpu")
        b, _, edges2 = e.generate(self.c, "graph", "cpu")
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(edges, edges2)
        self.assertTrue(np.all((a >= 256) & (a < e.VOCAB)))
        self.assertGreater(stats["mean_cell_change"], 0)
        self.assertGreater(stats["generation_seconds"], 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
