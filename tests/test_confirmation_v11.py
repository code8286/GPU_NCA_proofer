import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from confirmation_v11.data import select_ids, generate, canonical, draw_order
from confirmation_v11.plan import ROOT, read, validate
from confirmation_v11.run import train, verify_training, compute, conditions, decisions
from transfer_v4.model import LM
from transfer_v4.controls import prefix_shuffled, validate_prefix_control


class ConfirmationChecks(unittest.TestCase):
    def setUp(self):
        self.c = read(ROOT/'configs/interaction_v7.json')
        self.c.update(data_seed=20261023, seeds=[179, 193, 211])
        self.c['splits'] = {'train': 32, 'development': 4, 'assessment': 8}
        self.forbidden = set(canonical(np.arange(1000)).tolist())

    def test_fresh_phase_and_intervention_reservation(self):
        ids = select_ids(self.forbidden, self.c['data_seed'], self.c['splits'])
        groups = [set(ids['joint_orbit_ids'][k]) for k in ('train', 'development', 'assessment')]
        changed = set(ids['changed_assessment_orbits'])
        for group in groups+[changed]:
            self.assertFalse(group & self.forbidden)
        self.assertTrue(groups[0].isdisjoint(groups[1] | groups[2] | changed))
        self.assertTrue(groups[1].isdisjoint(groups[2] | changed))
        self.assertEqual(len(groups[0]), 32)
        self.assertEqual(len(groups[2]), 8)

    def test_determinism_and_capacity_rejection(self):
        self.assertEqual(select_ids(self.forbidden, 19, self.c['splits']), select_ids(self.forbidden, 19, self.c['splits']))
        with self.assertRaises(ValueError):
            select_ids(set(canonical(np.arange(65536)).tolist()), 19, self.c['splits'])

    def test_exact_nested_shuffle_and_known_neighbor_target(self):
        arrays, ids, native, changed = generate(self.forbidden, self.c)
        np.testing.assert_array_equal(arrays['prefix_train'][:16], prefix_shuffled(arrays['train'][:16], self.c, self.c['data_seed']+1))
        validate_prefix_control(arrays['train'], arrays['prefix_train'], self.c)
        np.testing.assert_array_equal(native[:, 1], changed[:, 1])
        np.testing.assert_array_equal(native[:, 7] ^ changed[:, 7], np.ones(8, dtype=np.uint16))

    def test_confirmation_settings_and_all_conditions_locked(self):
        settings = read(ROOT/'configs/confirmation_v11.json')
        validate(settings)
        self.assertEqual(len(set(conditions(self.c))), 27)
        for k, v in [('primary_objective', 'standard'), ('seeds', [131, 149, 167]), ('data_seed', 19), ('language_allowed', True), ('minimum_intervention_improvement_nats', 0.)]:
            with self.assertRaises(ValueError):
                validate(settings | {k: v})

    def test_failed_standard_does_not_erase_weighted_stratum(self):
        full, intervention = [], []
        for s in self.c['seeds']:
            for o in ('standard', 'weighted'):
                for a in ('coherent', 'prefix'):
                    for n in (512, 4096):
                        loss = 2. if n == 512 else 1.
                        if o == 'standard' and n == 4096 and s == 179:
                            loss = 3.
                        intervention.append(dict(seed=s, objective=o, arm=a, level=n, native_nll=loss, changed_nll=loss))
                for a in ('coherent', 'prefix'):
                    full.append(dict(seed=s, objective=o, arm='coherent', level=4096, data_arm=a, patch_nll=.5 if a == 'coherent' else 2.))
        pairs, strata = decisions(full, intervention, [{'level': 4096, 'frequency_nll': 3.}], self.c)
        self.assertEqual(strata, {'standard': False, 'weighted': True})
        self.assertFalse(all(strata.values()))
        self.assertEqual(len(pairs), 6)

    def test_assessment_blocks_before_all_training(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            with patch('confirmation_v11.run.verify_plan', return_value={'training_config': self.c}), patch('confirmation_v11.run.np.load', side_effect=AssertionError('Early data access')):
                with self.assertRaises(FileNotFoundError):
                    compute(Path(name))

    def test_csv_retains_failed_comparison_when_readiness_passes(self):
        from scripts.report_confirmation_v11 import pair_csv_rows
        row = {'objective': 'standard', 'seed': 179, 'passed': False,
               'readiness': {'passed': True, 'frequency_passed': True}}
        flattened = pair_csv_rows([row])[0]
        self.assertFalse(flattened['passed'])
        self.assertTrue(flattened['readiness_passed'])
        self.assertTrue(flattened['readiness_frequency_passed'])
        self.assertEqual(row['readiness'], {'passed': True, 'frequency_passed': True})

    def test_recovery_preserves_numerical_training_ast(self):
        import ast
        import inspect
        from scripts.recover_confirmation_v11 import derived_source
        import confirmation_v11.run as original
        before = ast.parse(inspect.getsource(original.train))
        source, transformed = derived_source()
        self.assertEqual(source, inspect.getsource(original.train))
        after = ast.parse(transformed)
        removed = []
        class RemoveSkip(ast.NodeTransformer):
            def visit_If(self, node):
                if (isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                        and node.test.func.id == 'completed_or_archive'):
                    self_outer.assertEqual(len(node.body), 1)
                    self_outer.assertIsInstance(node.body[0], ast.Continue)
                    removed.append(node)
                    return None
                return self.generic_visit(node)
        self_outer = self
        after = RemoveSkip().visit(after)
        self.assertEqual(len(removed), 1)
        self.assertEqual(ast.dump(before), ast.dump(after))

    def test_eight_arm_tiny_execution_and_actual_sampling(self):
        # Isolated software fixture; provenance is mocked. Not scientific evidence.
        c = copy.deepcopy(self.c)
        c.update(seeds=[179], eval_every=1)
        c['model'].update(layers=1, width=16, heads=2)
        c['ppt']['steps'] = 2
        arrays, _, _, _ = generate(self.forbidden, c)
        model = LM(c['model'], 18, 179, 50179)
        initial = dict(model=model.state_dict(), core_sha256=model.core_hash(), head_sha256=model.head_hash())
        plan = {'training_config': c, 'settings': {'purpose': 'isolated_fixture'},
                'expected_orders': {f'{n}_179': draw_order(179, c, n) for n in (512, 4096)}}
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            out = Path(name)
            (out/'data').mkdir()
            (out/'plan.json').write_text('{}')
            torch.save(initial, out/'initialization_179.pt')
            for suffix in ('', 'prefix_'):
                np.save(out/f'data/{suffix}train.npy', np.tile(arrays[suffix+'train'], (128, 1)))
                np.save(out/f'data/{suffix}development.npy', arrays[suffix+'development'])
            with patch('confirmation_v11.run.verify_plan', return_value=plan):
                train(out)
                rows = verify_training(out, plan)
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r['steps'] == 2 and r['prediction_targets'] == 336 for r in rows))
            self.assertEqual(len({r['initial_core_sha256'] for r in rows}), 1)
            self.assertEqual(len({r['actual_order']['sha256'] for r in rows}), 2)
            with self.assertRaises(FileExistsError):
                with patch('confirmation_v11.run.verify_plan', return_value=plan):
                    train(out)


if __name__ == '__main__':
    unittest.main()
