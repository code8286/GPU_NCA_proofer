"""New mechanism/analysis regressions; full training paths have offline fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from transfer_v4.analysis import common_target_analysis
from transfer_v4.common import ARMS, frame_length, load_config, validate_config
from transfer_v4.controls import prefix_shuffled, validate_prefix_control
from transfer_v4.nca import encode_frames, targets, control_probabilities, shuffled

ROOT = Path(__file__).resolve().parents[1]


class MechanismChecks(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT/'configs/transfer_v4_smoke.json')
        rng = torch.Generator().manual_seed(392)
        self.x = encode_frames(torch.randint(10, (8, 5, 144), generator=rng), 12, 10).numpy().astype(np.uint16)

    def test_prefix_histogram_matches_actual_supervised_targets(self):
        y = prefix_shuffled(self.x, self.c, 42)
        validate_prefix_control(self.x, y, self.c)
        boundary = self.c['model']['context']+1
        a = targets(torch.tensor(self.x[:, :boundary].astype(np.int64)), self.c, patch_only=True).numpy()
        b = targets(torch.tensor(y[:, :boundary].astype(np.int64)), self.c, patch_only=True).numpy()
        np.testing.assert_array_equal(np.sort(a, axis=1), np.sort(b, axis=1))
        pa, _ = control_probabilities(self.x, self.c)
        pb, _ = control_probabilities(y, self.c)
        np.testing.assert_array_equal(pa, pb)
        self.assertFalse(np.array_equal(self.x[:, frame_length(self.c):boundary], y[:, frame_length(self.c):boundary]))

    def test_partial_final_frame_boundary_and_initial_input_counts(self):
        c = copy.deepcopy(self.c)
        c['model']['context'] = 101  # Boundary lies inside a patch frame.
        y = prefix_shuffled(self.x, c, 81)
        validate_prefix_control(self.x, y, c)
        np.testing.assert_array_equal(np.sort(self.x[:, :102], axis=1), np.sort(y[:, :102], axis=1))

    def test_control_is_reproducible_but_seed_changes_order(self):
        a = prefix_shuffled(self.x, self.c, 42)
        np.testing.assert_array_equal(a, prefix_shuffled(self.x, self.c, 42))
        self.assertFalse(np.array_equal(a, prefix_shuffled(self.x, self.c, 43)))
        self.assertEqual(a.dtype, np.uint16)

    def test_no_suffix_tokens_leak_into_training_prefix(self):
        c = self.c
        x = self.x.copy()
        boundary = c['model']['context']+1
        suffix = x[:, boundary:]
        suffix[suffix < 10000] = 9999
        y = prefix_shuffled(x, c, 42)
        self.assertEqual(int((y[:, :boundary] == 9999).sum()), int((x[:, :boundary] == 9999).sum()))

    def test_default_v3_gap_is_masked_conditioning_not_unused_suffix(self):
        c = load_config(ROOT/'configs/transfer_v4.json')
        g = torch.Generator().manual_seed(124)
        x = encode_frames(torch.randint(10, (4, 27, 144), generator=g), 12, 10).numpy().astype(np.uint16)
        old = shuffled(x, c, 123)
        boundary = c['model']['context']+1
        self.assertEqual(x.shape[1], boundary+1)
        self.assertTrue(np.all(x[:, boundary:] >= 10000))
        np.testing.assert_array_equal(np.sort(x[:, :boundary], axis=1), np.sort(old[:, :boundary], axis=1))
        a = targets(torch.tensor(x[:, :boundary].astype(np.int64)), c, patch_only=True).numpy()
        b = targets(torch.tensor(old[:, :boundary].astype(np.int64)), c, patch_only=True).numpy()
        self.assertFalse(np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1)))
        self.assertFalse(np.array_equal(x[:, :frame_length(c)], old[:, :frame_length(c)]))
        validate_prefix_control(x, prefix_shuffled(x, c, 123), c)

    def test_initial_frame_markers_and_histogram_corruption_rejected(self):
        good = prefix_shuffled(self.x, self.c, 42)
        for index in (1, 38, 39):
            with self.subTest(index=index):
                bad = good.copy()
                bad[0, index] = (int(bad[0, index])+1) % 10000
                with self.assertRaises(ValueError):
                    validate_prefix_control(self.x, bad, self.c)

    def test_short_sequence_rejected(self):
        with self.assertRaises(ValueError):
            prefix_shuffled(self.x[:, :30], self.c, 42)

    def test_default_is_six_fresh_cohorts_and_seven_arms(self):
        c = load_config(ROOT/'configs/transfer_v4.json')
        self.assertEqual(len(c['seeds']), 6)
        self.assertFalse(set(c['seeds']) & {17, 29, 43})
        self.assertEqual(len(ARMS), 7)
        for mutation in ('criterion', 'old_seed', 'few_seeds', 'test'):
            bad = copy.deepcopy(c)
            if mutation == 'criterion': bad['study']['alpha_per_comparison'] = .05
            elif mutation == 'old_seed': bad['seeds'][0] = 43
            elif mutation == 'few_seeds': bad['seeds'] = bad['seeds'][:3]
            else: bad['text']['evaluate_test'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_config(bad)

    def test_frozen_sources_and_mechanistic_identity(self):
        manifest = ROOT/'configs/historical_sources_pre_v4.json'
        if not (ROOT/'transfer_v3').exists():
            self.skipTest('Historical packages are not embedded in the standalone v4 notebook')
        for name, expected in json.loads(manifest.read_text()).items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)
        for name in ('model.py', 'nca.py', 'graph.py', 'training.py'):
            # Training only changes checkpoint schema; numerical updates are fixed.
            a = (ROOT/'transfer_v3'/name).read_text().replace('"schema_version": 3', '"schema_version": 4')
            self.assertEqual(a, (ROOT/'transfer_v4'/name).read_text(), name)
        old = json.loads((ROOT/'configs/transfer_v3.json').read_text())
        new = load_config(ROOT/'configs/transfer_v4.json')
        for key in old:
            if key not in ('schema_version', 'seeds'):
                self.assertEqual(old[key], new[key], key)


class AnalysisChecks(unittest.TestCase):
    def fixture(self):
        c = load_config(ROOT/'configs/transfer_v4.json')
        def row(step, nll):
            return {'language_step': step, 'validation_nll': nll, 'language_tokens': step*4096,
                    'language_seconds': step+1., 'cumulative_seconds': step+1000.}
        curves = {}
        for seed in c['seeds']:
            curves['scratch', seed] = [row(0, 10), row(2000, 5)]
            for arm in ('grid_nca', 'prefix_graph'):
                curves[arm, seed] = [row(0, 10), row(1800, 5), row(2000, 4.9)]
            curves['graph_nca', seed] = [row(0, 10), row(1600, 5), row(2000, 4.8)]
        return c, curves

    def test_six_wins_pass_both_fixed_tests_at_common_target(self):
        c, curves = self.fixture()
        a = common_target_analysis(curves, c)
        self.assertTrue(a['supports_graph_advantage_on_declared_target'])
        for item in a['comparisons']:
            self.assertEqual(item['one_sided_sign_p'], 1/64)
            self.assertAlmostEqual(item['geomean_language_tokens_speedup_all_cohorts'], 1800/1600)
            self.assertEqual({r['target_nll'] for r in item['rows']}, {5})

    def test_one_tie_is_not_dropped_and_prevents_claim(self):
        c, curves = self.fixture()
        curves['grid_nca', c['seeds'][-1]] = copy.deepcopy(curves['graph_nca', c['seeds'][-1]])
        a = common_target_analysis(curves, c)
        self.assertFalse(a['supports_graph_advantage_on_declared_target'])
        self.assertEqual(a['comparisons'][0]['one_sided_sign_p'], 7/64)
        self.assertEqual(len(a['comparisons'][0]['rows']), 6)

    def test_censoring_withholds_aggregate_and_significance(self):
        c, curves = self.fixture()
        for row in curves['graph_nca', c['seeds'][-1]]:
            row['validation_nll'] = 8.
        a = common_target_analysis(curves, c)
        self.assertFalse(a['supports_graph_advantage_on_declared_target'])
        for item in a['comparisons']:
            self.assertIsNone(item['one_sided_sign_p'])
            self.assertIsNone(item['geomean_language_tokens_speedup_all_cohorts'])

    def test_missing_cohort_and_zero_step_are_undefined(self):
        for kind in ('missing', 'zero', 'truncated_scratch'):
            c, curves = self.fixture()
            if kind == 'missing': del curves['scratch', c['seeds'][0]]
            elif kind == 'truncated_scratch': curves['scratch', c['seeds'][0]][-1]['language_step'] = 1900
            else: curves['graph_nca', c['seeds'][0]][0]['validation_nll'] = 4.
            a = common_target_analysis(curves, c)
            self.assertFalse(a['supports_graph_advantage_on_declared_target'])
            self.assertIsNone(a['comparisons'][0]['geomean_language_tokens_speedup_all_cohorts'])

    def test_fixture_cannot_support_research_claim(self):
        c, curves = self.fixture()
        c['purpose'] = 'execution_check'
        self.assertFalse(common_target_analysis(curves, c)['supports_graph_advantage_on_declared_target'])


if __name__ == '__main__':
    unittest.main()
