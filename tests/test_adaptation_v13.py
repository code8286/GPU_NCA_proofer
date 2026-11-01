"""Scientific pipeline invariants for answer-only adaptation and frozen decisions."""
import copy
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from transfer_v4.model import LM
from adaptation_v13.training import encode, collate, answer_loss, order, learning_rate, decision


class Codec:
    eot_token = 256
    def encode(self, text):
        return list(text.encode())


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_prompt_boundary_eos_and_padding_mask(self):
        a = encode({'id':'a', 'prompt':'abc', 'references':['[1]']}, Codec(), 32)
        b = encode({'id':'b', 'prompt':'q', 'references':['[10]']}, Codec(), 32)
        x, y = collate([a,b])
        self.assertEqual(a['target'], [-100,-100,91,49,93,256])
        self.assertEqual(a['input'], list(b'abc[1]'))
        self.assertEqual(y[1,-1].item(), -100)
        self.assertEqual(int((y != -100).sum()), 9)

    def test_masked_projection_matches_full_loss_and_all_gradients(self):
        model = LM({'layers':2,'width':16,'heads':2,'context':32},257,7,11)
        full = copy.deepcopy(model)
        rows = [encode({'id':s,'prompt':s,'references':['[1,2]']},Codec(),32) for s in ('abc','d')]
        x,y = collate(rows)
        value = answer_loss(model,x,y)
        expected = F.cross_entropy(full(x).flatten(0,1), y.flatten(), ignore_index=-100)
        torch.testing.assert_close(value,expected)
        value.backward()
        expected.backward()
        for a,b in zip(model.parameters(),full.parameters()):
            self.assertIsNotNone(a.grad)
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=1e-4)

    def test_causal_right_padding_does_not_change_answer_loss(self):
        model = LM({'layers':1,'width':16,'heads':2,'context':32},257,7,11)
        row = encode({'id':'a','prompt':'abc','references':['[1]']},Codec(),32)
        x,y = collate([row])
        longer_x = F.pad(x,(0,5),value=13)
        longer_y = F.pad(y,(0,5),value=-100)
        torch.testing.assert_close(answer_loss(model,x,y),answer_loss(model,longer_x,longer_y))

    def test_no_silent_context_truncation_or_empty_prompt(self):
        for prompt in ('x'*40,''):
            with self.assertRaises(ValueError):
                encode({'id':'a','prompt':prompt,'references':['[1]']},Codec(),32)

    def test_full_model_and_optimizer_receive_update(self):
        model = LM({'layers':1,'width':16,'heads':2,'context':32},257,7,11)
        initial = {k:v.clone() for k,v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        x,y = collate([encode({'id':'a','prompt':'abc','references':['[1]']},Codec(),32)])
        answer_loss(model,x,y).backward()
        optimizer.step()
        self.assertTrue(all(not torch.equal(v,initial[k]) for k,v in model.state_dict().items()))
        self.assertEqual({int(v['step']) for v in optimizer.state.values()},{1})

    def test_order_is_reproducible_and_visits_each_epoch(self):
        a = order(17,12,4,123)
        np.testing.assert_array_equal(a,order(17,12,4,123))
        self.assertEqual(set(a.flatten()[:17]),set(range(17)))
        self.assertEqual(set(a.flatten()[17:34]),set(range(17)))
        self.assertFalse(np.array_equal(a,order(17,12,4,124)))

    def test_schedule_warmup_peak_and_final_decay(self):
        c = {'updates':600,'warmup_updates':30,'learning_rate':.0003}
        self.assertAlmostEqual(learning_rate(1,c),.00001)
        self.assertEqual(learning_rate(30,c),.0003)
        self.assertAlmostEqual(learning_rate(600,c),.00003)
        with self.assertRaises(ValueError):
            learning_rate(601,c)

    def test_target_is_censored_and_every_cell_matters(self):
        c = {'updates':600,'id_exact_match_target':.8,'id_format_target':.95}
        r = dict(step=600,selection_split='id_development',tokens=100,seconds=2.,exact_match=.79,
                 valid_format=1.,coverage=1.,cells=[{'exact_match':1.}]*4)
        self.assertFalse(decision([r],c)['passed'])
        self.assertFalse(decision([r],c)['first_pooled_crossing']['reached'])
        r['exact_match'] = .9
        self.assertTrue(decision([r],c)['passed'])
        r['cells'] = [{'exact_match':1.}]*3+[{'exact_match':.7}]
        self.assertFalse(decision([r],c)['passed'])
        r['coverage'] = .99
        self.assertFalse(decision([r],c)['passed'])


if __name__ == '__main__':
    unittest.main()
