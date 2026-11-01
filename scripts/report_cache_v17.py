"""Export verified intermediate-cache evidence and render all fixed endpoints."""
import csv
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cache_v17.run import verify_plan
from ood_v12.plan import read
from transfer_v4.common import dump,sha_file
from scripts.report_overfit_v15 import point

RUN=ROOT/'runs/cache_v17_run1'
PUBLIC=ROOT/'docs/results/cache_v17_run1'


def groups(folder,model):
    arm=model['arm']
    result=[('new64','cache64',e['step'],e['metrics'],f'models/{arm}/cache_predictions_{e["step"]}.jsonl','train') for e in model['curve']]
    result += [('new64','common16',600,model['common'],f'models/{arm}/cache_predictions_600.jsonl','common'),
               ('new64','development',600,model['development'],f'models/{arm}/development_predictions.jsonl','development')]
    for directory,variant,splits in (('reference','reference512',('train','common','development')),
                                     ('reference16','reference16',('common','development'))):
        ref=read(folder/f'{directory}/{arm}/metrics.json')
        for split in splits:
            label={'train':'cache64','common':'common16','development':'development'}[split]
            result.append((variant,label,600,ref[split],f'{directory}/{arm}/{split}_predictions.jsonl',split))
    return result


def summary(p,r,v,folder):
    models=[]
    for m in r['models']:
        arm=m['arm'];ref=read(folder/f'reference/{arm}/metrics.json');small=read(folder/f'reference16/{arm}/metrics.json')
        models.append({'arm':arm,'new64_cache':point(m['curve'][-1]['metrics']),'new64_common':point(m['common']),
                       'new64_development':point(m['development']),'reference512_cache':point(ref['train']),
                       'reference512_common':point(ref['common']),'reference512_development':point(ref['development']),
                       'reference16_common':point(small['common']),'reference16_development':point(small['development']),
                       'cache_competent':m['cache_competent'],'development_competent':m['development_competent'],
                       'first_cache_crossing':m['first_cache_crossing']})
    return {'plan_sha256':r['plan_sha256'],'models':models,'cache64_control_passed':r['cache64_control_passed'],
            'new_scientific_optimizer_updates':3000,'imported_reference_optimizer_updates':6000,
            'new_language_or_ppt_updates':0,'new_ood_predictions':0,'original_random_rule_readiness':'still_failed',
            'ood_allowed':False,'graph_acceleration_claim':None,'new_training_seconds':sum(m['training_seconds'] for m in r['models']),
            'new_evaluation_seconds':sum(m['evaluation_seconds'] for m in r['models']),
            'new_predictions_replayed':v['new_predictions_replayed'],'checkpoints':v['checkpoints']}


def csv_rows(folder,r):
    output=[]
    for m in r['models']:
        for variant,probe,step,metrics,name,split in groups(folder,m):
            tf={(x['task'],x['n']):x for x in metrics['teacher_forced']['cells']}
            for cell in metrics['generated']['cells']:
                output.append({'arm':m['arm'],'variant':variant,'probe':probe,'step':step,**cell,
                               **{'teacher_'+k:tf[cell['task'],cell['n']][k] for k in ('nll','digit_accuracy','other_accuracy','targets')}})
    return output


def render(p,r,s):
    models=s['models'];lines=lambda rows:'\n'.join(rows)
    primary=[];common=[];development=[];continuous=[];crossings=[]
    for m,sm in zip(r['models'],models):
        g=m['curve'][-1]['metrics']['generated'];arm=m['arm']
        cells=', '.join(f"{v['task'].split('_')[0]}{v['n']}:{round(v['exact_match']*v['count'])}/{v['count']}" for v in g['cells'])
        primary.append(f"| {arm} | {cells} | {g['exact_match']:.2%} | {g['valid_format']:.2%} | {sm['reference512_cache']['exact_match']:.2%} | {'pass' if m['cache_competent'] else 'fail'} |")
        common.append(f"| {arm} | {sm['reference16_common']['exact_match']:.2%} | {sm['new64_common']['exact_match']:.2%} | {sm['reference512_common']['exact_match']:.2%} |")
        development.append(f"| {arm} | {sm['reference16_development']['exact_match']:.2%} | {sm['new64_development']['exact_match']:.2%} | {sm['reference512_development']['exact_match']:.2%} | {'pass' if m['development_competent'] else 'fail'} |")
        for key,label in (('reference16_development','16 imported'),('new64_development','64 new'),('reference512_development','512 imported')):
            x=sm[key];continuous.append(f"| {arm} | {label} | {x['valid_format']:.2%} | {x['node_accuracy']:.2%} | {x['teacher_digit_accuracy']:.2%} | {x['teacher_other_accuracy']:.2%} | {x['teacher_nll']:.4f} |")
        crossing=sm['first_cache_crossing'];crossings.append(f"| {arm} | {crossing['step'] if crossing['reached'] else 'unreached (censored)'} |")
    lead='PASSES for all five arms' if s['cache64_control_passed'] else 'FAILS the frozen all-arm criterion'
    heldout=('All five new models still have 0/128 exact held-out answers; none passes full-ID competence.'
             if all(m['new64_development']['exact_match']==0 for m in models) else
             'Full-ID competence and exact held-out results are retained separately for every arm below.')
    nll_note=('Answer/EOS NLL improves from imported16 to new64 in every arm. '
              if all(m['new64_development']['teacher_nll']<m['reference16_development']['teacher_nll'] for m in models) else
              'Answer/EOS NLL changes from imported16 to new64 are shown for every arm. ')
    nll_note+=('The imported512 reference still has lower NLL and higher node accuracy than new64 in every arm.'
               if all(m['reference512_development']['teacher_nll']<m['new64_development']['teacher_nll'] and
                      m['reference512_development']['node_accuracy']>m['new64_development']['node_accuracy'] for m in models)
               else 'Do not substitute the exact-vector floor for these continuous endpoint comparisons.')
    count=p['expected_counts_per_arm'];old=p['records'][0]['historical_reference_counts'];small=p['records'][0]['small_reference_counts']
    return f'''# Stage 17: intermediate 64-example cache at 600 updates

Completed local CPU development experiment, 31 October 2026.

**The primary cache-fitting criterion {lead}.** Held-out competence is
reported separately below. {heldout} Neither a cache-fitting success nor a mixed endpoint
establishes GraphNCA acceleration, unseen-rule learning, cost recovery or graph
equivalence. Original random-rule readiness and the categorical bridge remain
failed/required; OOD and further full language suites remain gated.

## Frozen treatment and identical starts

Plan `{s['plan_sha256']}` and [frozen protocol](FROZEN_PROTOCOL.md) precede the
first new update. The only declared training change is cache size: 64 examples
(32 graphs, 16 per size, both BFS/Dijkstra tasks). The SAME outcome-blind salted
ranking nests common16 inside cache64 inside the old training128/512. All are
inspected development evidence from one historical cohort, not fresh confirmation.

All five models restart from the SAME full stage13 step0 language endpoints,
including learned tied heads and fresh optimizers. Exact source/state identities
are retained. Architecture, tokenizer, prompt/answer/EOS format, context, answer-
only loss/masks, full-model fp32 CPU AdamW, clipping, two threads, determinism,
seed, batch8, warmup/cosine and 600 updates match the original procedure. Every
arm receives the same actual order. No model continues an older adapted fit.

| Per arm | Imported16 | New64 | Imported512 |
| --- | ---: | ---: | ---: |
| Examples / graphs | 16 /8 | 64 /32 | 512 /256 |
| Updates / presentations | 600 /4800 | 600 /4800 | 600 /4800 |
| Presentations per example | 300 | 75 | 9–10 |
| Unpadded input tokens | {small['input_tokens']:,} | {count['input_tokens']:,} | {old['input_tokens']:,} |
| Supervised answer/EOS targets | {small['supervised_targets']:,} | {count['supervised_targets']:,} | {old['supervised_targets']:,} |
| Padded input positions | {small['padded_positions']:,} | {count['padded_positions']:,} | {old['padded_positions']:,} |

Cache content/diversity/repetition and lengths change together. Fixed updates do
not mean equal tokens, FLOPs or time, or isolate repetition causally. Original
reference costs are imported, not zero; no reference fit/inference is newly run.

## Fixed endpoints and matched cases

The primary all-arm criterion requires >=80% exact in EVERY cache64 task/size
cell (>=13/16), >=95% pooled validity and full coverage at update600. Transient
pooled crossings do not replace it. No best checkpoint or omitted seed/control.
Each endpoint retains the four fixed cache evaluations plus dev only at600.

| Arm | Four cache64 cells | Cache exact | Valid vectors | Imported512, SAME64 exact | Final criterion |
| --- | --- | --- | --- | --- | --- |
{lines(primary)}

Three-way exact-vector comparison on the SAME common16 examples. This is a
training subset for ALL variants, not a held-out probe. New64 common scores are
derived by exact final64 case/counter subsetting without an extra model forward.

| Arm | Imported16 | New64 | Imported512 |
| --- | --- | --- | --- |
{lines(common)}

Three-way exact-vector comparison on the SAME full held-out128 examples:

| Arm | Imported16 | New64 | Imported512 | New64 full-ID criterion |
| --- | --- | --- | --- | --- |
{lines(development)}

The unchanged full-ID criterion requires >=26/32 exact in every cell, >=95%
pooled validity and full coverage. Cache-fit and full-ID decisions remain
separate, and neither alone satisfies the separately required synthetic gate.

All fixed development continuous metrics, using the original node-example mean
and teacher task/size-cell means (not token-micro rates):

| Arm | Cache variant | Valid vectors | Generated node accuracy | Teacher digit | Teacher other | Teacher answer/EOS NLL |
| --- | --- | --- | --- | --- | --- | --- |
{lines(continuous)}

{nll_note} Node-score changes relative to imported16 are not uniformly positive.
These are descriptive same-case changes in one cohort, not independent
replications or a sufficient full-ID competence result.

Invalid vectors count wrong on every unconditional node/source/hop metric.
Teacher scores supply gold earlier answers and digit tokens may span punctuation;
their accuracy is not free-generation semantic accuracy. [All 220 cell rows](cells.csv)
retain source/hop/signed and token counters, exact case coverage and all three
variants. [Complete results](results.json) retain per-case first-error details.
The small16 reference has NO full64 measurement on the additional48 cases:
those cases are absent, never silently scored zero. Imported512 alone supplies
the full64 same-case reference. References remain distinct from new measurements.

First observed pooled80% cache crossing at the frozen steps, without interpolation:

| Arm | First observed update |
| --- | --- |
{lines(crossings)}

![Observed cache64 fit curves and matched held-out node accuracy](cache_comparison.png)

Lines connect recorded points only; overlapping curves can coincide. The pooled
line is not the final per-cell decision. Bars show fixed held-out node accuracy.
One inherited cohort/two correlated tasks per graph do not support independent-
cohort confidence intervals, graph equivalence or confirmatory speedup ratios.

## Replay, provenance and costs

Separate same-code replay reloads all20 new model/optimizer checkpoints and
reproduces all1920 new cache/development generations and teacher scores,
common16 subsetting, original complete tensor/optimizer starts, step counts,
actual batch order/LR/token exposure, fixed decisions and both reference subsets.
It is not another fit or independently implemented neural inference. Independent
decimal parsing/cell accounting and public-only score reproduction complement it.

[Provenance](PROVENANCE.json) binds72 exact copies of the new evidence and
explicitly imported reference records. Private weights/optimizer states, order
arrays and raw caches stay excluded. Public replay reconstructs selected data/
labels/order, references and generated scores without private weights/data,
tokenizer cache or network. Imported teacher counters alone do not reproduce
neural logits; see [usage](../../CACHE_V17_USAGE.md).

New scientific CPU task-adaptation updates:3000. Imported reference updates:6000.
New PPT/language updates, GPU measurements and OOD predictions:ZERO. Summed new
CPU update timers:{s['new_training_seconds']:.3f}s; evaluations:{s['new_evaluation_seconds']:.3f}s.
Timers exclude provenance, IO and separate replay; command wall durations remain
in the append-only log. Local regression fixtures briefly overlap the last fit;
these CPU timers are observed costs, not matched throughput/cost-recovery claims.
Inherited provenance can rehash old assessment bytes. New fitting reads only
declared ID caches; public reconstruction can regenerate unused synthetic OOD
rows in memory, as declared prospectively. No official language test, paid/remote
job, email, push or external publication occurs.

## Next decision

Preserve this fixed endpoint and all controls. Interpret fitting and held-out
prediction separately from these tables; do not discard an arm, change the salt,
extend a fit, relax a target or select a checkpoint after observing outcomes.
Before any further treatment, inspect the saved per-case/conditional errors and
actual optimization traces under a separately declared diagnostic. A later
training-budget change must hold cache/data fixed; do not combine it with a
format/objective/topology/serialization change and attribute the result uniquely.

Any graph-specific transfer confirmation still needs eligible synthetic starts,
the categorical/random-rule bridge, fresh paired rule/data/topology cohorts,
identical full language/adaptation stages, all five controls, ID-only selection,
predeclared OOD criteria/censoring/multiplicity and total costs. No endpoint here
automatically authorizes a new full language/OOD suite.
'''


def main():
    p=verify_plan(RUN);r=read(RUN/'results.json');v=read(RUN/'verification.json')
    if not v['passed'] or v['results_sha256']!=sha_file(RUN/'results.json') or v['new_predictions_replayed']!=1920 or v['checkpoints']!=20:
        raise ValueError('Complete separate checkpoint replay required')
    names=['plan.json','plan_receipt.json','FROZEN_PROTOCOL.md','fit_started.json','results.json','results_receipt.json','verification.json']
    names += [n for n in p['inputs_sha256'] if n.startswith(('reference/','reference16/'))]
    for m in r['models']:
        names.append(f'models/{m["arm"]}/summary.json')
        names += [n for n in m['files_sha256'] if n.endswith('_predictions.jsonl') or '/cache_predictions_' in n]
    if len(names)!=72:raise ValueError('Retain all72 evidence copies')
    PUBLIC.mkdir(parents=True,exist_ok=False)
    for name in names:
        target=PUBLIC/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(RUN/name,target)
    dump(PUBLIC/'PROVENANCE.json',{'files':{n:sha_file(PUBLIC/n) for n in names},'scope':'New64 CPU task fits with imported16/512 same-case references'})
    flat=csv_rows(PUBLIC,r)
    if len(flat)!=220:raise ValueError('Retain all220 cell rows')
    with (PUBLIC/'cells.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for row in flat for k in row}));w.writeheader();w.writerows(flat)
    s=summary(p,r,v,PUBLIC)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.uv-cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for m,color in zip(r['models'],['#555555','#1672b8','#d2600a','#39854a','#8657a6']):
        axes[0].plot([e['step'] for e in m['curve']],[e['metrics']['generated']['exact_match'] for e in m['curve']],label=m['arm'],color=color,marker='o')
    axes[0].axhline(.8,color='black',linestyle=':',linewidth=1)
    axes[0].set(title='64-cache: observed fitting accuracy',xlabel='Adaptation updates',ylabel='Exact-vector accuracy',ylim=(-.03,1.03),xticks=[0,100,300,600]);axes[0].legend(frameon=False,fontsize=8);axes[0].grid(alpha=.2)
    for key,offset,color,label in (('reference16_development',-.25,'#ad693c','16 imported'),('new64_development',0,'#1672b8','64 new'),('reference512_development',.25,'#63756a','512 imported')):
        axes[1].bar([i+offset for i in range(5)],[m[key]['node_accuracy'] for m in s['models']],width=.24,color=color,label=label)
    axes[1].set(title='Same held-out128, fixed update600',ylabel='Node accuracy (mean per example)',ylim=(0,1),xticks=range(5),xticklabels=[m['arm'] for m in s['models']]);axes[1].tick_params(axis='x',labelrotation=20);axes[1].legend(frameon=False)
    fig.suptitle('Stage17: cache-size development ablation; one historical cohort');fig.tight_layout();fig.savefig(PUBLIC/'cache_comparison.png',dpi=160);plt.close(fig)
    s['plot_sha256']=sha_file(PUBLIC/'cache_comparison.png');dump(PUBLIC/'summary.json',s)
    (PUBLIC/'REVIEW.md').write_text(render(p,r,s),encoding='utf-8',newline='\n')
    print('EXPORTED72 exact copies,220 cells; cache64 criterion',r['cache64_control_passed'])


if __name__=='__main__':main()
