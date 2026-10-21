import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from diagnose_transfer_v4_frequency import conditional_patch_nll
from readiness_v5.frequency import counts, per_rule_nll
from readiness_v5.gate import decision
from readiness_v5.positive_control import positive_control
from readiness_v5.evaluate import score, mean_nll
from readiness_v5.plan import verify_plan
from readiness_v5.run import require_readiness
from transfer_v4.model import LM
from transfer_v4.nca import encode_frames
from transfer_v4.training import evaluate


class ReadinessChecks(unittest.TestCase):
    def setUp(self):
        self.s = json.loads((ROOT/'configs/readiness_v5.json').read_text())
        (ROOT/'tmp').mkdir(exist_ok=True)

    def test_both_advantages_required_with_finite_scores(self):
        self.assertTrue(decision(1., 1.1, 1.1, self.s)['passed'])
        self.assertFalse(decision(1., 1.1, .9, self.s)['passed'])
        self.assertFalse(decision(1., .9, 1.1, self.s)['passed'])
        self.assertFalse(decision(1., 1., 1., self.s)['passed'])
        with self.assertRaises(ValueError):
            decision(float('nan'), 1., 1., self.s)

    def test_frequency_rows_match_previously_audited_joint_score(self):
        initial = np.array([[1, 2, 0, 1], [0, 1, 3, 0]], dtype=float)
        target = np.array([[2, 1, 3, 2], [3, 2, 2, 1]], dtype=float)
        p = np.array([.1, .2, .3, .4])
        values = per_rule_nll(initial, target, p, 10.)
        self.assertAlmostEqual(float(np.mean(values)), conditional_patch_nll(initial, target, p, 10.), places=12)

    def test_known_rule_positive_and_frequency_negative_are_detected(self):
        result = positive_control(self.s)
        self.assertTrue(result['passed'])
        self.assertTrue(result['learned_gate']['frequency_passed'])
        self.assertTrue(result['learned_gate']['ordering_passed'])
        self.assertFalse(result['frequency_negative_gate']['passed'])
        self.assertTrue(result['complement_equivalence_classes_disjoint'])
        self.assertEqual(result['frequency_nll'], result['frequency_shuffled_nll'])

    def test_per_rule_patch_scores_reproduce_frozen_evaluator(self):
        c = json.loads((ROOT/'configs/transfer_v4_smoke.json').read_text())
        model = LM(c['model'], 10002, 17, 18)
        g = torch.Generator().manual_seed(123)
        tokens = encode_frames(torch.randint(10, (4, 3, 144), generator=g), 12, 10).numpy().astype(np.uint16)
        rows = score(model, tokens, c)
        old = evaluate(model, tokens, c, 'cpu', True)
        self.assertAlmostEqual(mean_nll(rows), old['patch_nll'], places=5)
        self.assertEqual(sum(r['patch_targets'] for r in rows), old['patch_targets'])

    def test_unused_suffix_does_not_change_frequency_score(self):
        c = {'nca': {'grid': 4, 'states': 2}, 'model': {'context': 11}}
        a = np.array([[16, 1, 2, 1, 4, 17, 16, 1, 3, 2, 1, 17, 16, 4, 3, 2, 2, 17]], dtype=np.uint16)
        b = a.copy()
        b[:, 12:] = 15
        ai, at = counts(a, c)
        bi, bt = counts(b, c)
        np.testing.assert_array_equal(ai, bi)
        np.testing.assert_array_equal(at, bt)

    def test_plan_tamper_rejected_before_input_loading(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            folder = Path(name)
            (folder/'plan.json').write_text('{}')
            (folder/'plan_receipt.json').write_text(json.dumps({'plan_sha256': 'wrong'}))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_plan('unused', folder)

    def guard_fixture(self, folder, passed=True):
        s = copy.deepcopy(self.s)
        s['families'] = [['grid_nca', 'prefix_grid', 'nca']]
        s['seeds'] = [59, 71]
        rows = []
        for seed in s['seeds']:
            coherent, shuffled, freq = 1., 1.1, 1.2 if passed or seed == 59 else .9
            rows.append({'arm': 'grid_nca', 'seed': seed, 'structured_coherent_nll': coherent,
                         'structured_shuffled_nll': shuffled, 'frequency_nll': freq,
                         'gate': decision(coherent, shuffled, freq, s)})
        plan = {'settings': s, 'input_files_sha256': {f'runs/development/grid_nca_{seed}/ppt_selected.pt': 'weight-sha' for seed in s['seeds']}}
        (folder/'plan.json').write_text('{}')
        from transfer_v4.common import dump, sha_file
        result = {'plan_sha256': sha_file(folder/'plan.json'), 'positive_control': {'passed': True}, 'cohorts': rows}
        dump(folder/'results.json', result)
        dump(folder/'results_receipt.json', {'results_sha256': sha_file(folder/'results.json')})
        return plan

    def test_one_failed_cohort_blocks_all_language_authorization(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            folder = Path(name)
            plan = self.guard_fixture(folder, passed=False)
            with patch('readiness_v5.run.verify_plan', return_value=plan), patch('readiness_v5.run.positive_control', return_value={'passed': True}):
                with self.assertRaisesRegex(ValueError, 'Language training blocked'):
                    require_readiness('unused', folder)

    def test_complete_passing_guard_returns_only_bound_checkpoint_ids(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            folder = Path(name)
            plan = self.guard_fixture(folder)
            with patch('readiness_v5.run.verify_plan', return_value=plan), patch('readiness_v5.run.positive_control', return_value={'passed': True}):
                result = require_readiness('unused', folder)
            self.assertEqual(len(result), 2)
            self.assertEqual({r['checkpoint_sha256'] for r in result}, {'weight-sha'})


if __name__ == '__main__':
    unittest.main()
