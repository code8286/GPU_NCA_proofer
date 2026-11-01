"""One-factor training equivalence, nested selection and reference aggregation."""
import copy
from collections import Counter
import unittest
import torch
from transfer_v4.model import LM
from adaptation_v13.training import encode,order,collate,answer_loss,learning_rate
from id_audit_v14.metrics import select_training,teacher_forced
from overfit_v15.core import training_step,subset_teacher,reference_probe,competent,first_crossing


class Codec:
    eot_token=256
    def encode(self,text):return list(text.encode())
    def decode_single_token_bytes(self,t):return bytes([t])


class OverfitTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_small_cache_is_nested_in_original_outcome_blind_probe(self):
        rows=[{'id':f'{n}:{i}:{task}','base_id':f'{n}:{i}','task':task,'graph':{'n':n}}
              for n in (8,10) for i in range(40) for task in ('bfs_distances','dijkstra_distances')]
        c={'selection_salt':'id-audit-v14-20261031','training_graphs_per_size':32}
        big=select_training(rows,c);c['training_graphs_per_size']=4;small=select_training(rows,c)
        self.assertEqual(len(small),16)
        self.assertTrue({r['id'] for r in small}<={r['id'] for r in big})
        self.assertEqual(set(Counter((r['task'],r['graph']['n']) for r in small).values()),{4})

    def test_sampler_gives_exactly_300_presentations_without_changing_update_budget(self):
        batches=order(16,600,8,20261031)
        self.assertEqual(batches.shape,(600,8))
        self.assertEqual(set(Counter(batches.flatten()).values()),{300})
        self.assertTrue(all(len(set(batches.flatten()[i:i+16]))==16 for i in range(0,4800,16)))

    def test_training_update_matches_original_v13_numerical_procedure(self):
        a=LM({'layers':1,'width':16,'heads':2,'context':32},257,7,11);b=copy.deepcopy(a)
        c={'updates':600,'warmup_updates':30,'learning_rate':.0003,'weight_decay':.01,'gradient_clip':1.}
        oa=torch.optim.AdamW(a.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        ob=torch.optim.AdamW(b.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        examples=[encode({'id':'x','prompt':'abc','references':['[1,2]']},Codec(),32)]
        for step in (1,2):
            result=training_step(a,oa,examples,step,c)
            # Exact numerical operations from historical adaptation_v13.run.fit.
            b.train();x,y=collate(examples)
            for group in ob.param_groups:group['lr']=learning_rate(step,c)
            ob.zero_grad(set_to_none=True);loss=answer_loss(b,x,y);loss.backward()
            torch.nn.utils.clip_grad_norm_(b.parameters(),1.,error_if_nonfinite=True);ob.step()
            self.assertEqual(result['loss'],float(loss.detach()))
        for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
        self.assertEqual({int(v['step']) for v in oa.state.values()},{2})

    def test_reference_teacher_subset_matches_direct_subset_forward(self):
        model=LM({'layers':1,'width':16,'heads':2,'context':32},257,7,11)
        rows=[{'id':str(i),'task':'bfs_distances','graph':{'n':2},'prompt':'q'*i,'references':[s]}
              for i,s in enumerate(('[1,2]','[2,3]','[4,5]'),1)]
        full=teacher_forced(model,Codec(),rows)
        selected=subset_teacher(rows[::2],full);direct=teacher_forced(model,Codec(),rows[::2])
        self.assertEqual([r['id'] for r in selected['cases']],['1','3'])
        self.assertEqual(selected['cells'][0]['targets'],direct['cells'][0]['targets'])
        self.assertAlmostEqual(selected['cells'][0]['nll'],direct['cells'][0]['nll'],places=6)
        self.assertEqual(selected['cells'][0]['digit_accuracy'],direct['cells'][0]['digit_accuracy'])

    def test_reference_subset_rejects_missing_and_duplicate_case_identities(self):
        rows=[{'id':'x','task':'bfs_distances','graph':{'n':2}}]
        for cases in ([],[{'id':'x'},{'id':'x'}]):
            with self.assertRaises(ValueError):subset_teacher(rows,{'cases':cases})

    def test_four_example_cell_discretization_is_not_relaxed(self):
        c={'id_exact_match_target':.8,'id_format_target':.95}
        g={'cells':[{'exact_match':1.}]*4,'valid_format':1.,'coverage':1.}
        self.assertTrue(competent(g,c))
        g['cells']=[{'exact_match':1.}]*3+[{'exact_match':.75}]
        self.assertFalse(competent(g,c))

    def test_crossing_is_observed_censored_and_keeps_final_separate(self):
        c={'evaluation_updates':[0,100,300,600],'id_exact_match_target':.8}
        curve=[{'step':s,'metrics':{'generated':{'exact_match':0.}},'counts':{'input_tokens':s,'supervised_targets':s},'training_seconds':float(s)} for s in c['evaluation_updates']]
        self.assertFalse(first_crossing(curve,c)['reached'])
        curve[2]['metrics']['generated']['exact_match']=1.
        self.assertEqual(first_crossing(curve,c)['step'],300)
        self.assertEqual(curve[-1]['metrics']['generated']['exact_match'],0.)
        with self.assertRaises(ValueError):first_crossing(curve[:-1],c)

    def test_reference_prediction_mapping_does_not_reorder_gold_or_skip_missing(self):
        r={'id':'x','base_id':'b','task':'bfs_distances','view':'native','truth':[0,1],
           'graph':{'n':2,'source':0,'edges':[(0,1,1)]},'auxiliary':{'regime':'id','family':'cyclic_er','length':2,'cell':'toy'}}
        teacher={'cases':[{'id':'x','targets':2,'correct':2,'nll_sum':.1,'digit_targets':1,'digit_correct':1,'digit_nll_sum':.05,'other_targets':1,'other_correct':1,'other_nll_sum':.05}]}
        value,pred=reference_probe([r],[{'id':'unused','text':'[4,5]'},{'id':'x','text':'[0,1]'}],teacher)
        self.assertEqual(value['generated']['exact_match'],1)
        self.assertEqual([x['id'] for x in pred],['x'])
        with self.assertRaises(ValueError):reference_probe([r],[],teacher)


if __name__=='__main__':unittest.main()
