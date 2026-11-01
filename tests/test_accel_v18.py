"""Semantic target, censoring and stage-cost boundaries."""
import unittest
from accel_v18.core import marginal,baseline_predictions,semantic,target_met,crossing,ratios
from accel_v18.run import data_config,pipeline_at


def fixtures():
    return [{'id':f'{t}{n}','base_id':f'{t}{n}','task':t,'view':'native','graph':{'n':n,'source':1},
             'truth':[1,0]+[2]*(n-2),'auxiliary':{'regime':'id','family':'cyclic_er','length':n,'cell':f'{t}{n}'}}
            for t in ('bfs_distances','dijkstra_distances') for n in (8,10)]


def config():return {'baseline_margin':.1,'format_target':.95,'updates':600,'evaluation_updates':[0,100,200,300,400,500,600]}


def curve(values):
    return [{'step':s,'semantic':v,'counts':{'input_tokens':s*8},'training_seconds':s/10,'elapsed_seconds':s/9}
            for s,v in zip(config()['evaluation_updates'],values)]


class AccelTests(unittest.TestCase):
    def test_marginal_is_fit_to_training_non_source_labels(self):
        rows=fixtures();m=marginal(rows);pred=baseline_predictions(rows,m)
        self.assertTrue(all(v['distance']==2 for v in m))
        self.assertTrue(all(v['text'].split(',')[1]=='0' for v in pred))
        self.assertLess(semantic(rows,pred)['cells'][0]['accuracy'],1)

    def test_invalid_and_source_only_outputs_cannot_pass(self):
        rows=fixtures();base=semantic(rows,baseline_predictions(rows,marginal(rows)))
        for text in ('[0]','zeros'):
            pred=[{'id':r['id'],'text':text if text!='zeros' else '['+','.join('0' for _ in r['truth'])+']'} for r in rows]
            sem=semantic(rows,pred);self.assertFalse(target_met(sem,base,config()))
            self.assertTrue(all(v['correct']==0 for v in sem['cells']))

    def values(self):
        base={'cells':[{'task':r['task'],'n':r['graph']['n'],'accuracy':.2} for r in fixtures()]}
        good={'cells':[dict(v,accuracy=.4) for v in base['cells']],'valid_format':1,'coverage':1}
        bad={'cells':[dict(v,accuracy=.1) for v in base['cells']],'valid_format':1,'coverage':1}
        return base,good,bad

    def test_every_cell_and_format_are_required(self):
        b,g,_=self.values();self.assertTrue(target_met(g,b,config()))
        g['cells'][0]['accuracy']=.2;self.assertFalse(target_met(g,b,config()))
        g['cells'][0]['accuracy']=.4;g['valid_format']=.94;self.assertFalse(target_met(g,b,config()))

    def test_first_and_sustained_crossing_do_not_select_final(self):
        b,g,d=self.values();values=curve([d,g,d,g,g,d,d])
        self.assertEqual(crossing(values,b,config())['step'],100)
        self.assertEqual(crossing(values,b,config(),True)['step'],300)
        self.assertFalse(target_met(values[-1]['semantic'],b,config()))

    def test_censoring_is_not_attainment_or_finite_speedup(self):
        b,_,d=self.values();v=crossing(curve([d]*7),b,config())
        self.assertFalse(v['reached']);self.assertIsNone(v['step']);self.assertEqual(v['censoring_step'],600)
        self.assertIsNone(ratios(v,v)['updates_speedup']);self.assertIsNone(pipeline_at(v,{})['total_tokens'])

    def test_preexisting_target_has_no_infinite_ratio(self):
        v={'reached':True,'step':0}
        self.assertEqual(ratios(v,v)['reason'],'preexisting_target_no_finite_adaptation_ratio')

    def test_pipeline_keeps_ppt_language_and_mixed_hardware_costs(self):
        c={'ppt_tokens':40,'language_tokens':80,'ppt_updates':10,'language_updates':20,'original_cumulative_seconds':60.}
        v=pipeline_at({'reached':True,'step':5,'counts':{'input_tokens':8},'elapsed_seconds':9.},c)
        self.assertEqual(v['total_tokens'],128);self.assertEqual(v['total_updates'],35)
        self.assertEqual(v['accounted_mixed_hardware_seconds'],69.);self.assertIn('not controlled',v['reason'])

    def test_new_sample_generates_no_assessment_rows(self):
        c=data_config({'data_seed':2026103118})
        self.assertEqual(c['assessment_graphs_per_cell'],0);self.assertEqual(c['train_graphs_per_cell'],128)


if __name__=='__main__':unittest.main()
