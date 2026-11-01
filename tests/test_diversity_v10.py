import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from interaction_v7.data import generate as original_data, canonical
from transfer_v4.common import effective_tokens
from transfer_v4.model import LM
from diversity_v10.data import expanded_ids, generate, draw_order, exposure_mask
from diversity_v10.plan import read, ROOT, validate
from diversity_v10.run import Trace, conditions, subset_metrics, decisions, train, verify_training, compute


class DiversityChecks(unittest.TestCase):
    def setUp(self):
        self.c = read(ROOT/'configs/interaction_v7.json')
        self.c['splits'] = {'train': 32, 'development': 4, 'assessment': 4}
        self.arrays, self.ids = original_data(self.c)
        self.s = read(ROOT/'configs/diversity_v10.json')

    def test_nested_exact_arrays_and_shuffles(self):
        s = self.s | {'training_conditions': 64}
        a, b, ids = generate(self.ids, self.arrays['spatial_train'], self.arrays['spatial_prefix_train'], self.c, s)
        np.testing.assert_array_equal(a[:32], self.arrays['spatial_train'])
        np.testing.assert_array_equal(b[:32], self.arrays['spatial_prefix_train'])
        self.assertEqual(ids['initial_ids'][:32], self.ids['initial_ids']['train'])
        self.assertEqual(len(set(ids['joint_orbit_ids'])), 64)

    def test_no_new_original_or_intervention_phase_leakage(self):
        ids = expanded_ids(self.ids, 256, self.s['expansion_seed'])
        extra = set(canonical(np.array(ids[32:])).tolist())
        forbidden = set(sum(self.ids['joint_orbit_ids'].values(), []))
        forbidden.update(canonical(np.array(self.ids['initial_ids']['assessment']) ^ 8).tolist())
        self.assertFalse(extra & forbidden)

    def test_expansion_deterministic_and_capacity_guarded(self):
        self.assertEqual(expanded_ids(self.ids, 64, 19), expanded_ids(self.ids, 64, 19))
        with self.assertRaises(ValueError):
            expanded_ids(self.ids, 32, 19)
        with self.assertRaises(ValueError):
            expanded_ids(self.ids, 17000, 19)

    def test_trace_matches_frozen_order_and_changes_cache_bound(self):
        c = copy.deepcopy(self.c)
        c['ppt']['steps'] = 3
        rng, trace = np.random.default_rng(131+101), Trace()
        for _ in range(c['ppt']['steps']*c['accumulate']):
            trace.update(rng.integers(0, 4096, size=c['batch']).astype('<i8').tobytes())
        self.assertEqual(trace.summary(), draw_order(131, c, 4096))
        self.assertNotEqual(draw_order(131, c, 512)['sha256'], trace.summary()['sha256'])

    def test_cached_and_actually_exposed_examples_are_distinct(self):
        orders = {'512_131': {'observed_indices': list(range(512))}, '4096_131': {'observed_indices': [0, 512, 514]}}
        self.assertFalse(exposure_mask('extension', 512, 'standard', 131, orders, 4).any())
        self.assertEqual(exposure_mask('extension', 4096, 'weighted', 131, orders, 4).tolist(), [True, False, True, False])
        self.assertFalse(exposure_mask('assessment', 4096, 'weighted', 131, orders, 4).any())

    def test_empty_subset_is_null_not_zero(self):
        self.assertEqual(subset_metrics([1., 2.], [True, False], [False, False]), {'count': 0, 'nll': None, 'accuracy': None})
        self.assertEqual(subset_metrics([1., 2.], [True, False], [True, False]), {'count': 1, 'nll': 1., 'accuracy': 1.})

    def test_both_objectives_and_every_seed_required(self):
        c = self.c | {'seeds': [131, 149, 167]}
        full, intervention = [], []
        for seed in c['seeds']:
            for objective in ('standard', 'weighted'):
                for data in ('coherent', 'prefix'):
                    full.append({'seed': seed, 'level': 4096, 'objective': objective, 'arm': 'coherent', 'data_arm': data, 'patch_nll': .5 if data == 'coherent' else 2.})
                for level in (512, 4096):
                    for arm in ('coherent', 'prefix'):
                        intervention.append({'seed': seed, 'level': level, 'objective': objective, 'arm': arm, 'native_nll': 2. if level == 512 else 1., 'changed_nll': 2. if level == 512 else 1.})
        baseline = [{'level': 4096, 'frequency_nll': 3.}]
        _, strata = decisions(full, intervention, baseline, c)
        self.assertEqual(strata, {'standard': True, 'weighted': True})
        next(r for r in intervention if (r['seed'], r['level'], r['objective'], r['arm']) == (167, 4096, 'weighted', 'coherent'))['changed_nll'] = 3.
        _, strata = decisions(full, intervention, baseline, c)
        self.assertEqual(strata, {'standard': True, 'weighted': False})

    def test_all_model_conditions_and_settings_locked(self):
        self.assertEqual(len(set(conditions(self.c))), 27)
        validate(self.s)
        for key, value in (('training_conditions', 2048), ('expansion_seed', 17), ('language_allowed', True), ('minimum_first_patch_improvement_nats', 0.)):
            with self.assertRaises(ValueError):
                validate(self.s | {key: value})

    def test_assessment_cannot_access_arrays_before_completion(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            with patch('diversity_v10.run.verify_plan', return_value={'training_config': self.c}), patch('diversity_v10.run.np.load', side_effect=AssertionError('likelihood inputs accessed early')):
                with self.assertRaises(FileNotFoundError):
                    compute(Path(name))

    def test_isolated_four_arm_two_update_training_execution(self):
        # Explicit tiny software fixture with mocked provenance, not scientific data.
        c = copy.deepcopy(self.c)
        c.update(seeds=[131], eval_every=1)
        c['model'].update(layers=1, width=16, heads=2)
        c['ppt']['steps'] = 2
        model = LM(c['model'], 18, 131, 50131)
        initial = {'model': model.state_dict(), 'core_sha256': model.core_hash(), 'head_sha256': model.head_hash()}
        base = {'steps': 2, 'input_tokens': 2*effective_tokens(c), 'prediction_targets': 336,
                'initial_core_sha256': model.core_hash(), 'initial_head_sha256': model.head_hash()}
        plan = {'training_config': c, 'settings': {'purpose': 'isolated_execution_fixture'},
                'expected_orders': {'4096_131': draw_order(131, c, 4096)},
                'standard_controls': [dict(seed=131, objective=o, arm=a, source_summary=base) for o in ('standard', 'weighted') for a in ('coherent', 'prefix')]}
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            out = Path(name)
            (out/'data').mkdir()
            (out/'plan.json').write_text('{}', encoding='utf-8')
            torch.save(initial, out/'initialization_131.pt')
            for arm, suffix in (('coherent', ''), ('prefix', 'prefix_')):
                data = np.tile(self.arrays['spatial_'+('prefix_' if arm == 'prefix' else '')+'train'], (128, 1))
                np.save(out/f'data/{suffix}train.npy', data)
                np.save(out/f'data/original_{suffix}development.npy', data[:4])
            with patch('diversity_v10.run.verify_plan', return_value=plan):
                train(out)
                rows = verify_training(out, plan)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r['steps'] == 2 and r['prediction_targets'] == 336 for r in rows))
            self.assertEqual(len({r['actual_order']['sha256'] for r in rows}), 1)


if __name__ == '__main__':
    unittest.main()
