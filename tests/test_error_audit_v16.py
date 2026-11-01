"""Failure attribution needs explicit invalid, empty-bin and paired denominators."""
import copy
import unittest
from error_audit_v16.core import annotate, aggregate, paired


def row(key='a'):
    return {'id':key,'base_id':key,'task':'dijkstra_distances',
            'graph':{'n':4,'source':0,'edges':[(0,1,5),(1,2,7)]},'truth':[0,5,12,-1]}


def teacher(key='a'):
    return {'id':key,'targets':8,'correct':6,'nll_sum':4.,'digit_targets':4,
            'digit_correct':2,'digit_nll_sum':3.,'other_targets':4,'other_correct':4,'other_nll_sum':1.}


class ErrorAuditTests(unittest.TestCase):
    def test_invalid_vector_is_wrong_on_all_nodes_and_has_no_first_error(self):
        a=annotate([row()],[{'id':'a','text':'[0,5]'}]);v=aggregate(a,[teacher()])
        self.assertEqual(v['correct_nodes'],0);self.assertEqual(v['valid_nodes'],0)
        self.assertIsNone(v['valid_node_accuracy']);self.assertEqual(v['source_accuracy'],0)
        self.assertEqual(v['first_wrong_vectors'],0);self.assertEqual(v['under'],0)

    def test_weighted_truth_uses_unweighted_hops_and_preserves_unreachable(self):
        a=annotate([row()],[{'id':'a','text':'[0,4,13,-1]'}]);v=aggregate(a,[teacher()])
        self.assertEqual(a[0]['roles'],['source','one_hop','multi_hop','unreachable'])
        self.assertEqual((v['under'],v['equal'],v['over']),(1,2,1))
        self.assertEqual(v['first_wrong_counts'],{'1':1});self.assertEqual(v['multi_hop_accuracy'],0)
        self.assertEqual(a,annotate([row()],[{'id':'a','text':'[0,4,13,-1]'}],True))

    def test_empty_bins_are_null_and_exact_vectors_have_no_first_wrong(self):
        r=row();r['graph']['edges'] = sorted(r['graph']['edges']+[(0,2,8),(0,3,2)]);r['truth']=[0,5,8,2]
        v=aggregate(annotate([r],[{'id':'a','text':'[0,5,8,2]'}]),[teacher()])
        self.assertIsNone(v['multi_hop_accuracy']);self.assertIsNone(v['unreachable_accuracy'])
        self.assertEqual(v['first_wrong_vectors'],0)

    def test_valid_conditional_accuracy_cannot_hide_invalid_outputs(self):
        a=annotate([row('a'),row('b')],[{'id':'a','text':'[0,5,12,-1]'},{'id':'b','text':'wrong'}])
        v=aggregate(a,[teacher('a'),teacher('b')])
        self.assertEqual(v['valid_node_accuracy'],1);self.assertEqual(v['node_micro_accuracy'],.5)
        self.assertEqual(v['source_nodes'],2);self.assertEqual(v['source_valid_nodes'],1)

    def test_paired_directions_are_small_minus_reference_and_partition(self):
        rows=[row('a'),row('b')]
        small=annotate(rows,[{'id':'a','text':'[0,5,12,-1]'},{'id':'b','text':'wrong'}])
        ref=annotate(rows,[{'id':'a','text':'wrong'},{'id':'b','text':'[0,0,0,0]'}])
        v=paired(small,ref)
        self.assertEqual(v['format_gained'],1);self.assertEqual(v['format_lost'],1)
        self.assertEqual(v['node_gained'],4);self.assertEqual(v['node_lost'],1)
        self.assertEqual(sum(v['node_'+k] for k in ('both','gained','lost','neither')),8)

    def test_reject_duplicate_missing_and_wrong_identity(self):
        for preds in ([],[{'id':'z','text':'[]'}],[{'id':'a','text':'[]'}]*2):
            with self.assertRaises(ValueError):annotate([row()],preds)
        a=annotate([row()],[{'id':'a','text':'[]'}])
        with self.assertRaises(ValueError):aggregate(a,[teacher(),teacher()])

    def test_teacher_micro_counts_and_partition_corruption(self):
        a=annotate([row()],[{'id':'a','text':'[]'}]);t=teacher();v=aggregate(a,[t])
        self.assertEqual(v['teacher']['digit_accuracy'],.5);self.assertEqual(v['teacher']['digit_nll'],.75)
        for key,value in (('targets',7),('nll_sum',5.),('digit_nll_sum',float('nan'))):
            bad=copy.deepcopy(t);bad[key]=value
            with self.assertRaises(ValueError):aggregate(a,[bad])

    def test_status_failure_overrides_a_well_formed_vector(self):
        pred=[{'id':'a','text':'[0,5,12,-1]','status':'error'}]
        v=aggregate(annotate([row()],pred),[teacher()])
        self.assertEqual(v['validity'],0);self.assertEqual(v['coverage'],0);self.assertEqual(v['exact_accuracy'],0)


if __name__=='__main__':unittest.main()
