import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from trainability_v6.data import generate
from trainability_v6.plan import ROOT, freeze, verify_plan, validate
from trainability_v6.run import compute_assessment, verify_training
from transfer_v4.nca import targets
from transfer_v4.controls import validate_prefix_control


class TrainabilityChecks(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT/'configs/trainability_v6.json').read_text())
        (ROOT/'tmp').mkdir(exist_ok=True)

    def fixture(self):
        c = copy.deepcopy(self.c)
        c.update(purpose='execution_check', seeds=[1], splits={'train': 32, 'development': 4, 'assessment': 4})
        c['model'].update(layers=1, width=16, heads=2)
        c['ppt']['steps'] = 2
        return c

    def test_phase_equivalent_trajectories_never_cross_splits(self):
        arrays, ids = generate(self.c)
        groups = [set(v) for v in ids.values()]
        self.assertEqual(sum(map(len, groups)), 704)
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        for name, count in self.c['splits'].items():
            self.assertEqual(arrays[name].shape, (count, 48))
            self.assertEqual(arrays[name].dtype, np.uint16)

    def test_patch_transition_is_known_complement_with_original_mask(self):
        arrays, _ = generate(self.c)
        z = arrays['assessment'].reshape(-1, 8, 6)
        np.testing.assert_array_equal(z[:, 1:, 1:5], z[:, :-1, 1:5] ^ 15)
        y = targets(torch.tensor(arrays['assessment'].astype(np.int64)), self.c, patch_only=True)
        self.assertEqual(int((y != -100).sum()), 128*28)
        self.assertTrue(torch.all(y[:, :5] == -100))

    def test_matched_shuffles_are_reproducible_and_keep_conditioning(self):
        a, ids_a = generate(self.c)
        b, ids_b = generate(self.c)
        self.assertEqual(ids_a, ids_b)
        for name in self.c['splits']:
            np.testing.assert_array_equal(a[name], b[name])
            np.testing.assert_array_equal(a['prefix_'+name], b['prefix_'+name])
            validate_prefix_control(a[name], a['prefix_'+name], self.c)
            self.assertFalse(np.array_equal(a[name], a['prefix_'+name]))

    def test_development_backbone_budget_margins_and_no_language_are_enforced(self):
        validate(self.c)
        changes = [('language_allowed', True), ('selection', 'best_validation'), ('minimum_ordering_advantage_nats', 0.)]
        for key, value in changes:
            c = copy.deepcopy(self.c)
            c[key] = value
            with self.assertRaises(ValueError):
                validate(c)
        for section, key, value in [('model', 'width', 128), ('ppt', 'steps', 2000)]:
            c = copy.deepcopy(self.c)
            c[section][key] = value
            with self.assertRaises(ValueError):
                validate(c)

    def test_frozen_array_tampering_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            out = Path(name)/'run'
            freeze(out, self.fixture())
            verify_plan(out)
            with (out/'data/train.npy').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                verify_plan(out)

    def test_changed_plan_receipt_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            out = Path(name)/'run'
            freeze(out, self.fixture())
            (out/'plan.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Plan checksum'):
                verify_plan(out)

    def test_assessment_cannot_load_targets_before_training_is_complete(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            out = Path(name)/'run'
            freeze(out, self.fixture())
            with patch('trainability_v6.run.np.load', side_effect=AssertionError('assessment accessed')):
                with self.assertRaises(FileNotFoundError):
                    compute_assessment(out)

    def test_refuses_reusing_frozen_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            out = Path(name)/'run'
            freeze(out, self.fixture())
            with self.assertRaises(FileExistsError):
                freeze(out, self.fixture())


if __name__ == '__main__':
    unittest.main()
