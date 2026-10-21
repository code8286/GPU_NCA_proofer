import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from interaction_v7.data import step, unpack, pack, canonical, generate, oracle_nll
from interaction_v7.plan import ROOT, read, validate, freeze, verify_plan
from interaction_v7.run import compute_assessment
from transfer_v4.nca import targets
from transfer_v4.controls import validate_prefix_control


class InteractionChecks(unittest.TestCase):
    def setUp(self):
        self.c = read(ROOT/'configs/interaction_v7.json')
        (ROOT/'tmp').mkdir(exist_ok=True)

    def fixture(self):
        c = copy.deepcopy(self.c)
        c.update(purpose='execution_check', seeds=[1], splits={'train': 32, 'development': 4, 'assessment': 4})
        c['model'].update(layers=1, width=16, heads=2)
        c['ppt']['steps'] = 2
        return c

    def test_spatial_map_is_exhaustively_bijective_and_involutive(self):
        ids = np.arange(65536)
        x = unpack(ids)
        y = step(x, 'spatial')
        np.testing.assert_array_equal(step(y, 'spatial'), x)
        self.assertEqual(len(np.unique(pack(y))), 65536)

    def test_neighbor_outside_patch_causally_changes_its_next_patch(self):
        x = np.zeros((1, 4, 4), dtype=int)
        y = x.copy()
        y[0, 0, 3] = 1  # Outside the top-left 2x2 patch, adjacent to (0,0).
        np.testing.assert_array_equal(x[:, :2, :2], y[:, :2, :2])
        self.assertNotEqual(step(x, 'spatial')[0, 0, 0], step(y, 'spatial')[0, 0, 0])
        self.assertEqual(step(x, 'complement')[0, 0, 0], step(y, 'complement')[0, 0, 0])

    def test_joint_orbits_cover_both_maps_and_have_expected_count(self):
        ids = np.arange(65536)
        keys = canonical(ids)
        self.assertEqual(len(np.unique(keys)), 16512)
        np.testing.assert_array_equal(keys, canonical(ids ^ 65535))
        np.testing.assert_array_equal(keys, canonical(pack(step(unpack(ids), 'spatial'))))

    def test_matched_initials_and_joint_phase_exclusion(self):
        arrays, ids = generate(self.c)
        groups = [set(v) for v in ids['joint_orbit_ids'].values()]
        self.assertEqual(len(set.union(*groups)), 704)
        for split in self.c['splits']:
            np.testing.assert_array_equal(arrays[f'complement_{split}'][:, :6], arrays[f'spatial_{split}'][:, :6])
            self.assertEqual(arrays[f'spatial_{split}'].shape, (self.c['splits'][split], 48))

    def test_masks_controls_and_data_are_reproducible(self):
        a, ids = generate(self.c)
        b, ids_b = generate(self.c)
        self.assertEqual(ids, ids_b)
        for rule in ('complement', 'spatial'):
            for split in self.c['splits']:
                name = f'{rule}_{split}'
                np.testing.assert_array_equal(a[name], b[name])
                validate_prefix_control(a[name], a[f'{rule}_prefix_{split}'], self.c)
                labels = targets(torch.tensor(a[name].astype(np.int64)), self.c, patch_only=True)
                self.assertEqual(int((labels != -100).sum()), self.c['splits'][split]*28)

    def test_known_full_frame_oracle_is_causal_and_detects_shuffle(self):
        arrays, _ = generate(self.c)
        for rule in ('complement', 'spatial'):
            native = oracle_nll(arrays[f'{rule}_development'], rule, .99)
            shuffled = oracle_nll(arrays[f'{rule}_prefix_development'], rule, .99)
            np.testing.assert_allclose(native, -np.log(.99), atol=1e-12)
            self.assertGreater(float(np.mean(shuffled)), float(np.mean(native))+.01)

    def test_every_historical_setting_is_locked(self):
        validate(self.c)
        for key, value in [('batch', 4), ('data_seed', 9), ('language_allowed', True), ('minimum_local_table_advantage_nats', 0.)]:
            c = copy.deepcopy(self.c)
            c[key] = value
            with self.assertRaises(ValueError):
                validate(c)

    def test_frozen_input_tamper_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            out = Path(name)/'run'
            freeze(out, self.fixture())
            with (out/'data/spatial_train.npy').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                verify_plan(out)

    def test_assessment_waits_for_all_training_arms(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            out = Path(name)/'run'
            freeze(out, self.fixture())
            with patch('interaction_v7.run.np.load', side_effect=AssertionError('assessment accessed')):
                with self.assertRaises(FileNotFoundError):
                    compute_assessment(out)

    def test_frozen_plan_and_directory_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            out = Path(name)/'run'
            freeze(out, self.fixture())
            with self.assertRaises(FileExistsError):
                freeze(out, self.fixture())
            (out/'plan.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_plan(out)


if __name__ == '__main__':
    unittest.main()
