import copy
import math
import unittest
import numpy as np
import torch
from fit_v9.diagnostic import classify, conditions, score_cell, transition_score, comparisons
from transfer_v4.model import LM


class RecordingModel(torch.nn.Module):
    def __init__(self, confident=False):
        super().__init__()
        self.lengths = []
        self.confident = confident

    def forward(self, x):
        self.lengths.append(x.shape[1])
        logits = torch.zeros((*x.shape, 18))
        if self.confident:
            for pos in range(x.shape[1]):
                target = 16 if pos == 5 else 17 if pos == 10 else 0
                logits[:, pos, target] = 6 if pos in (5, 10) else 3
        return logits


class FitChecks(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[16, 1, 2, 3, 4, 17, 16, 0, 0, 0, 0, 17]], dtype=np.uint16)

    def test_frozen_categories_and_margin_boundaries(self):
        self.assertEqual(classify(.01, .01), 'fit_regression_on_train_and_assessment')
        self.assertEqual(classify(-.01, .01), 'training_gain_with_assessment_reversal')
        for pair in ((0, .1), (.1, 0), (-.1, -.1), (.009, .1)):
            self.assertEqual(classify(*pair), 'mixed_or_below_margin')

    def test_all_conditions_retain_initial_and_both_controls(self):
        rows = conditions({'seeds': [131, 149, 167]})
        self.assertEqual(len(set(rows)), 15)
        for s in (131, 149, 167):
            self.assertIn((s, 'initial', None), rows)
            for k in ('standard', 'weighted'):
                for a in ('coherent', 'prefix'):
                    self.assertIn((s, k, a), rows)

    def test_uniform_nll_and_prefix_lengths(self):
        model = RecordingModel()
        row = score_cell(model, self.array)
        self.assertAlmostEqual(row['first_patch_nll'], math.log(18), places=6)
        self.assertAlmostEqual(row['transition_all_token_nll'], math.log(18), places=6)
        self.assertEqual(model.lengths, [7, 11])
        self.assertEqual(row['transition_patch_accuracy'], 1.)

    def test_patch_marker_positions_and_six_target_mean(self):
        row = score_cell(RecordingModel(True), self.array)
        patch = math.log(math.exp(3)+17)-3
        marker = math.log(math.exp(6)+17)-6
        self.assertAlmostEqual(row['transition_patch_nll'], patch, places=6)
        self.assertAlmostEqual(row['transition_marker_nll'], marker, places=6)
        self.assertAlmostEqual(row['transition_all_token_nll'], (4*patch+2*marker)/6, places=6)
        self.assertEqual(row['per_query']['transition_patch_correct_count'], [4])

    def test_later_frames_are_never_supplied(self):
        extended = np.concatenate([self.array, np.full((1, 36), 65535, dtype=np.uint16)], axis=1)
        self.assertEqual(score_cell(RecordingModel(True), self.array), score_cell(RecordingModel(True), extended))

    def test_real_causal_backbone_short_long_agreement(self):
        row = score_cell(LM({'layers': 1, 'heads': 2, 'width': 16, 'context': 47}, 18, 11, 12), self.array)
        self.assertLessEqual(row['short_long_max_nll_difference'], 1e-5)

    def test_bad_transition_shapes_and_layout_rejected(self):
        for array in (self.array[:, :11], self.array[:0], self.array[0]):
            with self.assertRaises(ValueError):
                transition_score(RecordingModel(), array)
        a = self.array.copy()
        a[:, 7] = 17
        with self.assertRaises(ValueError):
            transition_score(RecordingModel(), a)

    def test_paired_deltas_and_generalization_gaps(self):
        c, cells = {'seeds': [131]}, []
        for seed, condition, arm in conditions(c):
            for data_arm in ('coherent', 'prefix'):
                for split, value in zip(('train', 'development', 'assessment'), (1., 1.2, 1.3)):
                    cells.append({'seed': seed, 'condition': condition, 'arm': arm, 'data_arm': data_arm, 'split': split,
                                  'first_patch_nll': value+(.2 if condition == 'weighted' else 0)})
        before = copy.deepcopy(cells)
        pairs, gaps = comparisons(cells, c)
        self.assertEqual(cells, before)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(len(gaps), 10)
        self.assertTrue(all(r['category'] == 'fit_regression_on_train_and_assessment' for r in pairs))
        self.assertAlmostEqual(gaps[0]['assessment_minus_train_nll'], .3)


if __name__ == '__main__':
    unittest.main()
