"""Source-aware marginal target, semantic curves and explicit right censoring."""
from collections import Counter,defaultdict
from ood_v12.scoring import score


def marginal(train):
    counts=defaultdict(Counter)
    for r in train:
        counts[r['task'],r['graph']['n']].update(v for i,v in enumerate(r['truth']) if i!=r['graph']['source'])
    return [{'task':task,'n':n,'distance':min(c,key=lambda v:(-c[v],v))} for (task,n),c in sorted(counts.items())]


def baseline_predictions(rows,modes):
    table={(v['task'],v['n']):v['distance'] for v in modes}
    return [{'id':r['id'],'text':'['+','.join(str(0 if i==r['graph']['source'] else table[r['task'],r['graph']['n']]) for i in range(r['graph']['n']))+']','status':'ok'} for r in rows]


def semantic(rows,predictions):
    s=score(rows,predictions);index={r['id']:r for r in rows};counts=defaultdict(Counter)
    for case in s['cases']:
        r=index[case['id']];g=r['graph'];c=counts[r['task'],g['n']]
        for i,truth in enumerate(r['truth']):
            if i==g['source']:continue
            c['nodes']+=1;c['correct']+=case['parsed'] is not None and case['parsed'][i]==truth
    return {'cells':[{'task':task,'n':n,**dict(c),'accuracy':c['correct']/c['nodes']} for (task,n),c in sorted(counts.items())],
            'valid_format':s['overall_valid_format'],'coverage':s['coverage'],'exact_match':s['overall_exact_match']}


def target_met(value,baseline,c):
    targets={(v['task'],v['n']):v['accuracy']+c['baseline_margin'] for v in baseline['cells']}
    if len(targets)!=4 or any(x>1 for x in targets.values()):raise ValueError('Four feasible frozen cell targets required')
    return (len(value['cells'])==4 and {(v['task'],v['n']) for v in value['cells']}==set(targets)
            and all(v['accuracy']+1e-12>=targets[v['task'],v['n']] for v in value['cells'])
            and value['valid_format']>=c['format_target'] and value['coverage']==1)


def crossing(curve,baseline,c,sustained=False):
    if [v['step'] for v in curve]!=c['evaluation_updates']:raise ValueError('Retain every declared evaluation')
    for i,v in enumerate(curve):
        if target_met(v['semantic'],baseline,c) and (not sustained or i==len(curve)-1 or target_met(curve[i+1]['semantic'],baseline,c)):
            return {'reached':True,'step':v['step'],'counts':v['counts'],'training_seconds':v['training_seconds'],
                    'elapsed_seconds':v['elapsed_seconds'],'censoring_step':None,'preexisting':v['step']==0}
    return {'reached':False,'step':None,'counts':None,'training_seconds':None,'elapsed_seconds':None,
            'censoring_step':c['updates'],'censoring_counts':curve[-1]['counts'],'preexisting':False}


def ratios(treatment,reference):
    if not treatment['reached'] or not reference['reached']:
        return {'updates_speedup':None,'input_tokens_speedup':None,'train_seconds_speedup':None,'reason':'at_least_one_right_censored'}
    if treatment['step']==0 or reference['step']==0:
        return {'updates_speedup':None,'input_tokens_speedup':None,'train_seconds_speedup':None,'reason':'preexisting_target_no_finite_adaptation_ratio'}
    return {'updates_speedup':reference['step']/treatment['step'],
            'input_tokens_speedup':reference['counts']['input_tokens']/treatment['counts']['input_tokens'],
            'train_seconds_speedup':reference['training_seconds']/treatment['training_seconds'],'reason':None}
