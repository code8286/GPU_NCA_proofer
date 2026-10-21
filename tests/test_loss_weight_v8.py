import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from loss_weight_v8.objective import weights, loss, update
from loss_weight_v8.plan import ROOT, read, validate
from loss_weight_v8.run import compute_assessment
from transfer_v4.nca import targets
from transfer_v4.model import LM
from transfer_v4.training import update as standard_update, optimizer_for
from interaction_v7.data import generate


class WeightingChecks(unittest.TestCase):
    def setUp(self):
        self.c = read(ROOT/'configs/interaction_v7.json')
        self.s = read(ROOT/'configs/loss_weight_v8.json')
        self.c['model'].update(layers=1, width=16, heads=2)
        self.c['ppt']['steps'] = 4
        self.c['splits'] = {'train': 32, 'development': 4, 'assessment': 4}
        self.tokens = generate(self.c)[0]['spatial_train']
        self.y = targets(torch.tensor(self.tokens[:2].astype(np.int64)), self.c)

    def test_first_transition_half_mass_and_marker_balance(self):
        w = weights(self.y, self.c, 6.)
        self.assertEqual(float(w.sum()), 144.)
        self.assertEqual(float(w[:, 5:11].sum()), 72.)
        self.assertEqual(int((w > 0).sum()), 84)
        self.assertEqual(float(w[self.y < 16].sum()), 96.)

    def test_factor_one_matches_original_loss_exactly(self):
        z = torch.randn(2, 47, 18)
        self.assertEqual(float(loss(z, self.y, self.c, 1.)), float(F.cross_entropy(z.reshape(-1, 18), self.y.reshape(-1), ignore_index=-100)))

    def test_masked_initial_frame_has_zero_gradient(self):
        z = torch.randn(2, 47, 18, requires_grad=True)
        value = loss(z, self.y, self.c, 6.)
        value.backward()
        self.assertTrue(torch.all(z.grad[:, :5] == 0))
        b = z.detach().clone()
        b[:, :5] = 1000
        self.assertEqual(float(value.detach()), float(loss(b, self.y, self.c, 6.)))

    def test_weighted_expression_is_normalized_not_global_scaling(self):
        z = torch.zeros(2, 47, 18)
        self.assertAlmostEqual(float(loss(z, self.y, self.c, 6.)), float(np.log(18)), places=6)
        per = F.cross_entropy(z.transpose(1, 2), self.y, reduction='none', ignore_index=-100)
        w = weights(self.y, self.c, 6.)
        self.assertAlmostEqual(float(loss(z, self.y, self.c, 6.)), float((per*w).sum()/w.sum()), places=6)

    def test_invalid_weights_and_empty_objective_rejected(self):
        for factor in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                weights(self.y, self.c, factor)
        with self.assertRaises(ValueError):
            loss(torch.zeros(2, 47, 18), torch.full_like(self.y, -100), self.c, 6.)

    def test_factor_one_update_matches_original_parameters_and_order(self):
        model = LM(self.c['model'], 18, 13, 14)
        other = copy.deepcopy(model)
        a, b = optimizer_for(model, self.c['ppt'], True), optimizer_for(other, self.c['ppt'], True)
        ha, hb = hashlib.sha256(), hashlib.sha256()
        x = update(model, a, self.tokens, self.c, self.c['ppt'], 0, np.random.default_rng(5), ha, 1.)
        y = standard_update(other, b, self.tokens, self.c, self.c['ppt'], 0, np.random.default_rng(5), 'cpu', True, hb)
        self.assertEqual(x, y)
        self.assertEqual(ha.hexdigest(), hb.hexdigest())
        for key, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, other.state_dict()[key]), key)

    def test_treatment_decision_and_language_settings_locked(self):
        validate(self.s)
        for key, value in [('first_transition_weight', 2.), ('language_allowed', True), ('require_all_seeds', False), ('minimum_first_patch_nll_improvement', 0.)]:
            s = dict(self.s)
            s[key] = value
            with self.assertRaises(ValueError):
                validate(s)

    def test_assessment_cannot_read_queries_before_all_weighted_models_finish(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            with patch('loss_weight_v8.run.verify_plan', return_value={'training_config': self.c, 'settings': self.s}), patch('loss_weight_v8.run.np.load', side_effect=AssertionError('query accessed')):
                with self.assertRaises(FileNotFoundError):
                    compute_assessment(Path(name))


if __name__ == '__main__':
    unittest.main()
