"""Fixed optimization blocks, exposure and probe-specific readiness semantics."""
import copy
import unittest
from fit_audit_v19.core import trace_summary,decide,select_training


def trace():
    return [{'step':i,'sampled_ids':[str((8*(i-1)+j)%512) for j in range(8)],'loss':1-i/1000,
             'gradient_norm':.5 if i%2 else 2.,'lr':.0001,'input_tokens':800,'supervised_targets':160,'padded_positions':900}
            for i in range(1,601)]


def probe(accuracy):
    return {'semantic':{'cells':[{'task':t,'n':n,'accuracy':accuracy} for t in ('bfs_distances','dijkstra_distances') for n in (8,10)],
                        'valid_format':1.,'coverage':1.},
            'generated':{'cells':[{'exact_match':0.} for _ in range(4)],'valid_format':1.,'coverage':1.}}


class FitAuditTests(unittest.TestCase):
    def test_all_fixed_blocks_and_actual_exposures_are_preserved(self):
        result=trace_summary(trace(),{'updates':600},['0','511'])
        self.assertEqual(len(result['blocks']),6);self.assertEqual(result['presentations'],4800)
        self.assertEqual(result['unique_examples'],512);self.assertEqual(result['selected_exposures'],{'0':10,'511':9})
        self.assertEqual(result['blocks'][0]['fraction_preclip_norm_above_one'],.5)
        self.assertEqual(result['blocks'][-1]['last_update'],600)
        self.assertGreater(result['first50_mean_loss'],result['last50_mean_loss'])

    def test_missing_nonfinite_and_unseen_exposures_fail_closed(self):
        with self.assertRaises(ValueError):trace_summary(trace()[:-1],{'updates':600},['0'])
        values=trace();values[50]['gradient_norm']=float('nan')
        with self.assertRaises(ValueError):trace_summary(values,{'updates':600},['0'])
        with self.assertRaises(ValueError):trace_summary(trace(),{'updates':600},['unseen'])

    def test_each_probe_uses_same_predictor_on_its_own_cases(self):
        train=probe(.45);dev=probe(.45)
        base={'train':probe(.4)['semantic'],'development':probe(.2)['semantic']}
        d=decide(train,dev,base,{'baseline_margin':.1,'format_target':.95})
        self.assertFalse(d['training_semantic_ready']);self.assertTrue(d['development_semantic_ready'])
        self.assertEqual(d['interpretation'],'development_only_target_seen_probe_failure')

    def test_seen_failure_and_true_heldout_gap_are_distinct(self):
        base={s:probe(.2)['semantic'] for s in ('train','development')};c={'baseline_margin':.1,'format_target':.95}
        self.assertEqual(decide(probe(.25),probe(.1),base,c)['interpretation'],'seen_probe_semantic_fitting_not_established')
        self.assertEqual(decide(probe(.4),probe(.1),base,c)['interpretation'],'heldout_gap_after_seen_probe_target')

    def test_probe_selection_is_balanced_graph_paired_and_outcome_independent(self):
        rows=[{'id':f'{n}:{i}:{t}','base_id':f'{n}:{i}','task':t,'graph':{'n':n},'outcome':i%2}
              for n in (8,10) for i in range(128) for t in ('bfs_distances','dijkstra_distances')]
        c={'selection_salt':'fit-audit-v19-20261101','training_graphs_per_size':32}
        first=select_training(rows,c);other=copy.deepcopy(rows)
        for v in other:v['outcome']=1-v['outcome']
        self.assertEqual([v['id'] for v in first],[v['id'] for v in select_training(other,c)])
        self.assertEqual(len(first),128);self.assertEqual(len({v['base_id'] for v in first}),64)


if __name__=='__main__':unittest.main()
