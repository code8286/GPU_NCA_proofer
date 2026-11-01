import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from ood_v12.graphs import distances, floyd, sample_graph, topology_key, intervene
from ood_v12.data import make_examples, generate, public_input, answer
from ood_v12.scoring import score, parse_answer, score_official, accuracy_crossing, require_paired_transfer
from ood_v12.inference import greedy, last_logits, impossible_prefix
from transfer_v4.model import LM

ROOT = Path(__file__).resolve().parents[1]


class GraphOODChecks(unittest.TestCase):
    def rows(self):
        edges = [(0, 1, 9), (0, 2, 1), (1, 2, 1)]
        return make_examples(4, edges, 0, np.random.default_rng(7), {'regime': 'id', 'family': 'fixture', 'length': 4, 'cell': 'fixture'})

    def test_bfs_and_dijkstra_different_answers_and_unreachable(self):
        edges = [(0, 1, 9), (0, 2, 1), (1, 2, 1)]
        self.assertEqual(distances(4, edges, 0), [0, 1, 1, -1])
        self.assertEqual(distances(4, edges, 0, True), [0, 2, 1, -1])
        for weighted in (True, False):
            self.assertEqual(distances(4, edges, 0, weighted), floyd(4, edges, 0, weighted))

    def test_independent_oracles_across_families_and_sizes(self):
        rng = np.random.default_rng(19)
        for n in (8, 12, 20):
            for family in ('cyclic_er', 'branching_tree', 'path', 'two_components'):
                e = sample_graph(n, family, rng)
                for source in (0, n-1):
                    for weighted in (False, True):
                        self.assertEqual(distances(n, e, source, weighted), floyd(n, e, source, weighted))
                        self.assertNotEqual(distances(n, e, source, weighted), distances(n, intervene(n, e, source, weighted), source, weighted))

    def test_negative_or_duplicate_edges_rejected(self):
        for e in ([(0, 1, 0)], [(0, 1, -1)], [(1, 1, 2)], [(0, 1, 2), (0, 1, 3)]):
            with self.assertRaises(ValueError):
                distances(3, e, 0)

    def test_fingerprint_invariant_to_relabeling_and_weights(self):
        e = sample_graph(10, 'cyclic_er', np.random.default_rng(19))
        p = np.random.default_rng(23).permutation(10)
        other = [(*sorted((int(p[u]), int(p[v]))), 1) for u, v, _ in e]
        self.assertEqual(topology_key(10, e), topology_key(10, other))

    def test_invariance_and_causal_controls(self):
        rows = self.rows()
        oracle = [{'id':r['id'], 'text': r['references'][0]} for r in rows]
        value = score(rows, oracle)
        self.assertEqual(value['overall_exact_match'], 1)
        self.assertTrue(all(r['native_and_cut_correct'] for r in value['paired']))
        native = {r['task']:r['references'][0] for r in rows if r['view'] == 'native'}
        cached = score(rows, [{'id':r['id'], 'text':native[r['task']]} for r in rows])
        self.assertEqual(cached['overall_exact_match'], .75)
        self.assertFalse(any(r['native_and_cut_correct'] for r in cached['paired']))

    def test_wrong_consistency_does_not_count_as_rule_execution(self):
        rows = self.rows()
        value = score(rows, [{'id':r['id'], 'text':'[0,0,0,0]'} for r in rows])
        self.assertEqual(value['overall_exact_match'], 0)
        self.assertTrue(all(r['rename_consistent'] for r in value['paired']))
        self.assertFalse(any(r['native_and_renamed_correct'] for r in value['paired']))

    def test_no_partial_or_extra_explanation_accepted_as_correct(self):
        for text in ('[0,1]', 'answer: [0,1,-1,2]', '[0,1,-1,2] extra', '[0,1.0,-1,2]', '[0,-2,1,2]'):
            self.assertIsNone(parse_answer(text, 4))
        self.assertEqual(parse_answer(' [0, 1,-1,2]\n', 4), [0, 1, -1, 2])
        self.assertTrue(impossible_prefix('The answer is'))
        self.assertFalse(impossible_prefix(' [0,1'))

    def test_missing_duplicate_or_wrong_prediction_identity_fails(self):
        rows = self.rows()
        p = [{'id':r['id'], 'text':r['references'][0]} for r in rows]
        for values in (p[:-1], p+[p[0]], p[:-1]+[p[0]], p[:-1]+[dict(p[-1], id='other')]):
            with self.assertRaises(ValueError):
                score(rows, values)

    def test_split_disjoint_including_actual_interventions(self):
        c = json.loads((ROOT/'configs/ood_v12.json').read_text())
        c.update(train_graphs_per_cell=4, development_graphs_per_cell=2, assessment_graphs_per_cell=2)
        data, info = generate(c)
        again, _ = generate(c)
        self.assertEqual(data, again)
        keys = []
        for name in ('train', 'development', 'assessment'):
            current = {topology_key(r['graph']['n'], r['graph']['edges']) for r in data[name]}
            self.assertTrue(all(current.isdisjoint(k) for k in keys))
            keys.append(current)
        for r in data['assessment']:
            self.assertEqual(set(public_input(r)), {'id', 'prompt'})
        self.assertEqual(info['train']['base_graph_instances'], 8)

    def test_official_adapter_preserves_reference_semantics(self):
        rows = [{'id': 'x', 'prompt': 'bfs:\npi:', 'references': ['[0 0 1]\n\n']}]
        self.assertEqual(score_official(rows, [{'id':'x','text':'[0 0 1]'}])['exact_match'], 1)
        self.assertEqual(score_official(rows, [{'id':'x','text':'[0,0,1]'}])['exact_match'], 0)

    def test_crossings_are_censored_and_id_only(self):
        c = [dict(step=100, tokens=1000, seconds=2., exact_match=.5, selection_split='id_development')]
        self.assertFalse(accuracy_crossing(c, .8)['reached'])
        self.assertEqual(accuracy_crossing(c, .4)['step'], 100)
        with self.assertRaises(ValueError):
            accuracy_crossing(c+[dict(c[0], step=200, selection_split='ood')], .4)

    def test_future_transfer_rejects_unequal_stages_and_unready_ppt(self):
        arms = ('scratch', 'grid_nca', 'graph_nca', 'prefix_grid', 'prefix_graph')
        fields = ('seed', 'model_config_sha256', 'language_updates', 'language_order_sha256', 'language_schedule_sha256',
                  'language_head_initial_sha256', 'probe_updates', 'probe_order_sha256', 'probe_schedule_sha256',
                  'probe_head_initial_sha256', 'ood_plan_sha256')
        rows = [dict.fromkeys(fields, 'same') | dict(arm=a, synthetic_readiness_verified=True, readiness_evidence_sha256='a'*64) for a in arms]
        self.assertTrue(require_paired_transfer(rows))
        for key in ('language_updates', 'probe_order_sha256', 'language_head_initial_sha256'):
            bad = copy.deepcopy(rows)
            bad[2][key] = 'different'
            with self.assertRaises(ValueError):
                require_paired_transfer(bad)
        rows[2]['synthetic_readiness_verified'] = False
        with self.assertRaises(ValueError):
            require_paired_transfer(rows)

    def test_last_position_projection_matches_full_forward(self):
        model = LM({'layers':2,'width':16,'heads':2,'context':64}, 257, 7, 11)
        x = torch.tensor([[2,3,5,7,11]])
        with torch.no_grad():
            torch.testing.assert_close(last_logits(model,x), model(x)[:,-1], atol=1e-5, rtol=1e-5)

    def test_context_overflow_is_explicit_not_silently_truncated(self):
        class Codec:
            eot_token = 256
            def encode(self, x): return list(x.encode())
            def decode(self, x): return bytes(x).decode(errors='replace')
        model = LM({'layers':1,'width':16,'heads':2,'context':8}, 257, 7, 11)
        pred = greedy(model, Codec(), [{'id':'x','prompt':'long graph input'}], 4)
        self.assertEqual(pred[0]['status'], 'context_overflow')
        with self.assertRaises(ValueError):
            greedy(model, Codec(), [{'id':'x','prompt':'a','truth':[1]}], 2)


if __name__ == '__main__':
    unittest.main()
