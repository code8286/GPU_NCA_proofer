"""Outcome-independent probes, fixed trace blocks and descriptive decisions."""
from collections import Counter
from statistics import mean
import math
from id_audit_v14.metrics import select_training
from accel_v18.core import target_met
from overfit_v15.core import competent


def trace_summary(trace,config,selected_ids):
    if config['updates']!=600 or len(trace)!=config['updates'] or [v['step'] for v in trace]!=list(range(1,config['updates']+1)):
        raise ValueError('Full ordered original trace required')
    if any(not math.isfinite(v[k]) for v in trace for k in ('loss','gradient_norm','lr')):
        raise ValueError('Finite numerical trace required')
    exposure=Counter(key for v in trace for key in v['sampled_ids'])
    if not set(selected_ids)<=set(exposure) or any(exposure[key] not in (9,10) for key in selected_ids):
        raise ValueError('Every selected case must have actual prior training exposure')
    blocks=[]
    for start in range(0,600,100):
        values=trace[start:start+100];norms=[v['gradient_norm'] for v in values]
        blocks.append({'first_update':start+1,'last_update':start+100,'mean_loss':mean(v['loss'] for v in values),
          'mean_gradient_norm':mean(norms),'min_gradient_norm':min(norms),'max_gradient_norm':max(norms),
          'fraction_preclip_norm_above_one':mean(v>1 for v in norms),'mean_lr':mean(v['lr'] for v in values),
          **{key:sum(v[key] for v in values) for key in ('input_tokens','supervised_targets','padded_positions')}})
    return {'blocks':blocks,'first50_mean_loss':mean(v['loss'] for v in trace[:50]),
            'last50_mean_loss':mean(v['loss'] for v in trace[-50:]),'unique_examples':len(exposure),
            'presentations':sum(exposure.values()),'selected_exposures':{key:exposure[key] for key in selected_ids}}


def decide(train,dev,baselines,c):
    t=target_met(train['semantic'],baselines['train'],c);d=target_met(dev['semantic'],baselines['development'],c)
    label=('both_sampled_probes_meet_development_target' if t and d else
           'heldout_gap_after_seen_probe_target' if t else
           'development_only_target_seen_probe_failure' if d else 'seen_probe_semantic_fitting_not_established')
    old={'id_exact_match_target':.8,'id_format_target':.95}
    return {'training_semantic_ready':t,'development_semantic_ready':d,'interpretation':label,
            'training_exact_calibration':competent(train['generated'],old),
            'development_exact_calibration':competent(dev['generated'],old)}
