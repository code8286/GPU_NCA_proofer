"""Seen/held-out selection and node/token error accounting invariants."""
import copy
from collections import Counter
import unittest
import torch
from torch.nn import functional as F
from transfer_v4.model import LM
from adaptation_v13.training import encode,collate
from id_audit_v14.metrics import select_training,generated_errors,teacher_forced,digit_bearing,interpret


class Codec:
    eot_token = 256
    def encode(self,text): return list(text.encode())
    def decode_single_token_bytes(self,token): return bytes([token])


def row(truth=None,task='bfs_distances',edges=None):
    return {'id':'x','base_id':'b','task':task,'view':'native','prompt':'q','references':['[0,1,2]'],
            'truth':truth or [0,1,2],'graph':{'n':3,'source':0,'edges':edges or [(0,1,1),(1,2,1)]},
            'auxiliary':{'regime':'id','family':'cyclic_er','length':3,'cell':'toy'}}


class IDAuditTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_balanced_graph_selection_is_order_and_outcome_blind(self):
        rows = [{'id':f'{n}:{b}:{task}','base_id':f'{n}:{b}','task':task,'truth':[b],
                 'graph':{'n':n}} for n in (8,10) for b in range(5) for task in ('bfs_distances','dijkstra_distances')]
        c = {'selection_salt':'frozen','training_graphs_per_size':2}
        selected = select_training(rows,c)
        changed = copy.deepcopy(rows[::-1])
        for r in changed: r['truth'] = ['different-label']
        self.assertEqual([r['id'] for r in selected],[r['id'] for r in select_training(changed,c)])
        self.assertEqual(Counter((r['task'],r['graph']['n']) for r in selected),
                         {(t,n):2 for t in ('bfs_distances','dijkstra_distances') for n in (8,10)})
        self.assertEqual(sorted(Counter(r['base_id'] for r in selected).values()),[2]*4)

    def test_invalid_vector_counts_wrong_in_all_hop_denominators(self):
        value = generated_errors([row()],[{'id':'x','text':'[0,1]'}])
        cell = value['cells'][0]
        self.assertEqual(value['node_accuracy'],0)
        self.assertEqual(cell['invalid_examples'],1)
        for role in ('source','one_hop','multi_hop'):
            self.assertEqual(cell[role+'_nodes'],1)
            self.assertEqual(cell[role+'_accuracy'],0)
        self.assertIsNone(value['errors'][0]['first_wrong_node'])

    def test_weighted_task_hop_categories_use_unweighted_connectivity(self):
        r = row([0,9,10],'dijkstra_distances',[(0,1,9),(1,2,1)])
        value = generated_errors([r],[{'id':'x','text':'[0,9,3]'}])
        cell = value['cells'][0]
        self.assertEqual(cell['one_hop_nodes'],1)
        self.assertEqual(cell['one_hop_accuracy'],1)
        self.assertEqual(cell['multi_hop_nodes'],1)
        self.assertEqual(cell['multi_hop_accuracy'],0)

    def test_signed_errors_and_first_wrong_position(self):
        value = generated_errors([row()],[{'id':'x','text':'[0,-1,4]'}])
        e = value['errors'][0]
        self.assertEqual((e['under'],e['equal'],e['over'],e['negative_reachable']),(1,1,1,1))
        self.assertEqual((e['first_wrong_node'],e['normalized_first_wrong']),(1,.5))

    def test_teacher_scores_equal_ordinary_full_forward(self):
        model = LM({'layers':1,'width':16,'heads':2,'context':32},257,7,11)
        r = row(); r['references'] = ['[1,2]']
        codec = Codec()
        x,y = collate([encode(r,codec,32)])
        with torch.no_grad():
            logits = model(x)[y != -100]
            labels = y[y != -100]
            expected_loss = float(F.cross_entropy(logits,labels))
            expected_correct = int((logits.argmax(-1)==labels).sum())
        value = teacher_forced(model,codec,[r])['cells'][0]
        self.assertAlmostEqual(value['nll'],expected_loss,places=6)
        self.assertEqual(value['correct'],expected_correct)
        self.assertEqual((value['targets'],value['digit_targets'],value['other_targets']),(6,2,4))
        self.assertEqual(value['targets'],value['digit_targets']+value['other_targets'])

    def test_digit_role_includes_mixed_tokens_but_excludes_eos(self):
        class Mixed:
            eot_token=9
            def decode_single_token_bytes(self,t): return {1:b',12',2:b'-',3:b']',9:b'123'}[t]
        self.assertTrue(digit_bearing(Mixed(),1))
        for token in (2,3,9): self.assertFalse(digit_bearing(Mixed(),token))

    def test_teacher_measurement_does_not_change_weights_or_gradients(self):
        model = LM({'layers':1,'width':16,'heads':2,'context':32},257,7,11)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        teacher_forced(model,Codec(),[row()])
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_interpretation_preserves_fit_failure_and_all_fixed_categories(self):
        c = {'exact_target':.8,'format_target':.95,'large_gap':.1}
        def value(exact,fmt=1):
            return {'exact_match':exact,'valid_format':fmt,'coverage':1,'cells':[{'exact_match':exact}]*4}
        self.assertEqual(interpret(value(.1),value(0),c)['interpretation'],'training_probe_competence_not_established')
        self.assertEqual(interpret(value(.9),value(.9),c)['interpretation'],'both_probes_competent')
        self.assertEqual(interpret(value(.9),value(.1),c)['interpretation'],'held_out_gap_after_training_probe_competence')
        self.assertEqual(interpret(value(.9),value(.9,.5),c)['interpretation'],'cell_or_format_failure_without_large_pooled_gap')


if __name__ == '__main__': unittest.main()
