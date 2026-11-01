"""Nested diversity treatment, reference scope and fixed decision boundaries."""
from collections import Counter
import unittest
from adaptation_v13.training import order
from id_audit_v14.metrics import select_training
from cache_v17.core import validate_nested,competent,reference_probe


def rows():
    return [{'id':f'{n}:{i}:{task}','base_id':f'{n}:{i}','task':task,'graph':{'n':n}}
            for n in (8,10) for i in range(80) for task in ('bfs_distances','dijkstra_distances')]


class CacheTests(unittest.TestCase):
    def test_same_ranking_nests_all_three_probes(self):
        c={'selection_salt':'id-audit-v14-20261031','training_graphs_per_size':4}
        original=rows();common=select_training(original,c)
        c['training_graphs_per_size']=16;train=select_training(original,c)
        c['training_graphs_per_size']=32;large=select_training(original,c)
        validate_nested(train,common,[v['id'] for v in large])
        self.assertEqual(len({r['base_id'] for r in train}),32)

    def test_reject_replaced_common_identity_and_dropped_control_cell(self):
        c={'selection_salt':'id-audit-v14-20261031','training_graphs_per_size':16}
        original=rows();train=select_training(original,c);c['training_graphs_per_size']=4;common=select_training(original,c)
        with self.assertRaises(ValueError):validate_nested(train,common[:-1],[v['id'] for v in original])
        bad=[dict(v) for v in common];bad[0]['id']='unmeasured'
        with self.assertRaises(ValueError):validate_nested(train,bad,[v['id'] for v in original])

    def test_fixed_sampler_gives_75_presentations_and_same_4800_total(self):
        indices=order(64,600,8,20261031)
        self.assertEqual(indices.shape,(600,8))
        self.assertEqual(set(Counter(indices.flatten()).values()),{75})
        self.assertTrue(all(len(set(indices.flatten()[i:i+64]))==64 for i in range(0,4800,64)))

    def test_cache64_and_development_integer_decision_boundaries(self):
        c={'id_exact_match_target':.8,'id_format_target':.95}
        g={'cells':[{'exact_match':13/16} for _ in range(4)],'valid_format':61/64,'coverage':1}
        self.assertTrue(competent(g,c));g['cells'][3]['exact_match']=12/16
        self.assertFalse(competent(g,c));g['cells'][3]['exact_match']=13/16;g['valid_format']=60/64
        self.assertFalse(competent(g,c))
        g['cells']=[{'exact_match':26/32} for _ in range(4)];g['valid_format']=122/128
        self.assertTrue(competent(g,c));g['cells'][0]['exact_match']=25/32
        self.assertFalse(competent(g,c))

    def test_small_reference_cannot_silently_cover_unmeasured_cases(self):
        with self.assertRaises(ValueError):
            reference_probe([{'id':'a'},{'id':'b'}],[{'id':'a','text':'[0]'}],{'cases':[]})

    def test_all_five_arms_and_original_numerical_configuration_are_declared(self):
        import json
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        c=json.loads((root/'configs/cache_v17.json').read_text())
        old=json.loads((root/'configs/overfit_v15.json').read_text())
        self.assertEqual(c['arms'],old['arms']);self.assertEqual(c['selection_salt'],old['selection_salt'])
        self.assertEqual(c['adaptation_config'],old['adaptation_config'])
        self.assertFalse(c['ood_evaluation_allowed']);self.assertFalse(c['confirmatory_graph_claim_allowed'])


if __name__=='__main__':unittest.main()
