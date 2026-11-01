"""Check explicit counter partitions, paired marginals and rendered evidence."""
from pathlib import Path
import math
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from error_audit_v16.run import PUBLIC,verify_plan
from error_audit_v16.core import ROLES
from ood_v12.plan import read
from scripts.report_error_audit_v16 import render
from scripts.check_release import selected_files
from transfer_v4.common import sha_file


def check(prod=False):
    p=verify_plan();r=read(PUBLIC/'results.json');v=read(PUBLIC/'verification.json')
    assert v['passed'] and v['results_sha256']==sha_file(PUBLIC/'results.json')
    assert r['new_optimizer_updates']==r['new_model_forwards']==r['new_predictions']==0
    assert not r['ood_allowed'] and r['graph_acceleration_claim'] is None
    assert len(r['rows'])==100 and len(r['paired'])==50
    index={(x['arm'],x['variant'],x['probe'],x['group']):x for x in r['rows']}
    assert len(index)==100
    for x in r['rows']:
        assert sum(x[role+'_nodes'] for role in ROLES)==x['nodes']
        assert sum(x[role+'_correct'] for role in ROLES)==x['correct_nodes']
        assert sum(x[role+'_valid_nodes'] for role in ROLES)==x['valid_nodes']
        assert x['under']+x['equal']+x['over']==x['valid_nodes']
        assert x['equal']==x['correct_nodes']
        assert sum(x['first_wrong_counts'].values())==x['first_wrong_vectors']==x['valid_examples']-x['exact_examples']
        assert x['source_nodes']==x['examples']
        t=x['teacher']
        for key in ('targets','correct','nll_sum'):
            assert math.isclose(t[key],t['digit_'+key]+t['other_'+key],rel_tol=1e-12,abs_tol=1e-12)
        if x['group']=='pooled':
            children=[index[x['arm'],x['variant'],x['probe'],g] for g in p['groups'][1:]]
            for key in ('examples','nodes','correct_nodes','valid_nodes','valid_examples','exact_examples','under','equal','over'):
                assert sum(c[key] for c in children)==x[key]
            assert math.isclose(sum(c['node_macro_accuracy']*c['examples'] for c in children)/x['examples'],x['node_macro_accuracy'])
    for q in r['paired']:
        a=index[q['arm'],'small16',q['probe'],q['group']];b=index[q['arm'],'reference512',q['probe'],q['group']]
        for kind,countkey,marginal in (('format','examples','valid_examples'),('node','nodes','correct_nodes')):
            assert sum(q[kind+'_'+s] for s in ('both','gained','lost','neither'))==a[countkey]==b[countkey]
            assert q[kind+'_both']+q[kind+'_gained']==a[marginal]
            assert q[kind+'_both']+q[kind+'_lost']==b[marginal]
    assert (PUBLIC/'REVIEW.md').read_text(encoding='utf-8')==render()
    selected=set(selected_files(ROOT))
    assert not any('/data/' in name or name.endswith(('.pt','.npy')) for name in selected if name.startswith('docs/results/error_audit_v16_run1/'))
    counts={}
    for directory in ('proofs/lean','docs/theory','docs/paper'):
        names=[n for n in selected if n.startswith(directory+'/')]
        assert names
        if prod:
            assert all(sha_file(ROOT/n)==sha_file(ROOT/'Prod'/n) for n in names)
        counts[directory]=len(names)
    print('PASS: all counters/role/teacher/first-error partitions, 50 paired marginals, exact review, privacy and recursive math inclusion',counts,'Prod parity',prod)


if __name__=='__main__':check('--prod' in sys.argv)
