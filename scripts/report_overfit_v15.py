"""Export verified small-cache fitting evidence with unchanged reference/gates."""
import csv
import os
from pathlib import Path
import shutil
from statistics import mean
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from overfit_v15.run import verify_plan
from ood_v12.plan import read
from transfer_v4.common import dump,sha_file


def point(metrics):
    g=metrics['generated'];t=metrics['teacher_forced']['cells']
    return {k:g[k] for k in ('exact_match','valid_format','node_accuracy','coverage')} | {
        'teacher_'+k:mean(c[k] for c in t) for k in ('nll','digit_accuracy','other_accuracy')}


def main():
    run,public=ROOT/'runs/overfit_v15_run1',ROOT/'docs/results/overfit_v15_run1'
    p=verify_plan(run);r,v=read(run/'results.json'),read(run/'verification.json')
    if not v['passed'] or v['results_sha256']!=sha_file(run/'results.json'):raise ValueError('Exact checkpoint replay required')
    names=['plan.json','plan_receipt.json','FROZEN_PROTOCOL.md','fit_started.json','results.json','results_receipt.json','verification.json']
    names += [n for n in p['inputs_sha256'] if n.startswith('reference/')]
    for m in r['models']:
        names.append(f'models/{m["arm"]}/summary.json')
        names += [n for n in m['files_sha256'] if n.endswith('_predictions.jsonl') or '/cache_predictions_' in n]
    public.mkdir(parents=True,exist_ok=False)
    for n in names:
        path=public/n;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(run/n,path)
    dump(public/'PROVENANCE.json',{'files':{n:sha_file(public/n) for n in names},
                                 'scope':'New CPU small-cache task fits with explicitly imported historical 512-cache reference'})
    models=[];flat=[]
    for m in r['models']:
        arm=m['arm'];ref=read(run/f'reference/{arm}/metrics.json')
        models.append({'arm':arm,'small_cache':point(m['curve'][-1]['metrics']),'development':point(m['development']),
                       'reference_same_cache':point(ref['train']),'reference_development':point(ref['development']),
                       'cache_competent':m['cache_competent'],'development_competent':m['development_competent'],
                       'first_cache_crossing':m['first_cache_crossing']})
        groups=[('small16','cache',entry['step'],entry['metrics']) for entry in m['curve']]
        groups += [('small16','development',600,m['development']),('reference512','cache',600,ref['train']),('reference512','development',600,ref['development'])]
        for variant,probe,step,metrics in groups:
            tf={(c['task'],c['n']):c for c in metrics['teacher_forced']['cells']}
            for c in metrics['generated']['cells']:
                flat.append({'arm':arm,'variant':variant,'probe':probe,'step':step,**c,
                             **{'teacher_'+k:tf[c['task'],c['n']][k] for k in ('nll','digit_accuracy','other_accuracy','targets')}})
    with (public/'cells.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for row in flat for k in row}));w.writeheader();w.writerows(flat)
    summary={'plan_sha256':sha_file(run/'plan.json'),'models':models,'small_cache_control_passed':r['small_cache_control_passed'],
             'new_scientific_optimizer_updates':3000,'imported_reference_optimizer_updates':3000,
             'new_language_or_ppt_updates':0,'new_ood_predictions':0,'original_random_rule_readiness':'still_failed','graph_acceleration_claim':None,
             'new_training_seconds':sum(m['training_seconds'] for m in r['models']),
             'new_evaluation_seconds':sum(m['evaluation_seconds'] for m in r['models']),
             'new_predictions_replayed':v['new_predictions_replayed'],'checkpoint_replays':v['checkpoints']}
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.uv-cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    colors=['#555555','#1672b8','#d2600a','#39854a','#8657a6']
    for m,color in zip(r['models'],colors):
        axes[0].plot([e['step'] for e in m['curve']],[e['metrics']['generated']['exact_match'] for e in m['curve']],label=m['arm'],marker='o',color=color)
    axes[0].axhline(.8,color='black',linestyle=':',linewidth=1)
    axes[0].set(title='16-example cache: observed exact accuracy',xlabel='Adaptation updates',ylim=(-.03,1.03),xticks=[0,100,300,600]);axes[0].grid(alpha=.2)
    for variant,offset,color in (('small_cache',-.18,'#1672b8'),('development',.18,'#d2600a')):
        axes[1].bar([i+offset for i in range(5)],[m[variant]['exact_match'] for m in models],width=.35,color=color,label=variant)
    axes[1].set(title='Fixed endpoint: cache versus held-out ID',ylim=(0,1.03),xticks=range(5),xticklabels=[m['arm'] for m in models]);axes[1].tick_params(axis='x',labelrotation=20)
    axes[1].legend(frameon=False);axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Stage 15: 16-cache fitting control; 600 updates; one historical cohort')
    fig.tight_layout();fig.savefig(public/'cache_fit.png',dpi=160);plt.close(fig)
    summary['plot_sha256']=sha_file(public/'cache_fit.png');dump(public/'summary.json',summary)
    lead=('The small-cache fitting positive control PASSES for all five arms.' if r['small_cache_control_passed'] else
          'The small-cache fitting positive control FAILS its frozen all-arm final criterion.')
    heldout=('Every small-cache model still fails the original full-ID competence criterion.'
             if all(not m['development_competent'] for m in models) else
             'Full-ID competence is reported separately for every arm; the fitting decision does not replace it.')
    table='\n'.join(f"| {m['arm']} | {m['small_cache']['exact_match']:.2%} | {m['reference_same_cache']['exact_match']:.2%} | {m['development']['exact_match']:.2%} | {m['reference_development']['exact_match']:.2%} | {'pass' if m['cache_competent'] else 'fail'} |" for m in models)
    crossings='\n'.join(f"| {m['arm']} | {m['first_cache_crossing']['step'] if m['first_cache_crossing']['reached'] else 'not reached (censored)'} |" for m in models)
    c=p['adaptation_config'];counts=p['expected_counts_per_arm'];old=p['records'][0]['historical_reference_counts']
    details='\n'.join(f"| {m['arm']} | {m['small_cache']['valid_format']:.2%} / {m['development']['valid_format']:.2%} | {m['small_cache']['teacher_digit_accuracy']:.2%} / {m['development']['teacher_digit_accuracy']:.2%} | {m['small_cache']['node_accuracy']:.2%} / {m['development']['node_accuracy']:.2%} |" for m in models)
    dev_compare='\n'.join(f"| {m['arm']} | {m['development']['valid_format']:.2%} / {m['reference_development']['valid_format']:.2%} | {m['development']['node_accuracy']:.2%} / {m['reference_development']['node_accuracy']:.2%} | {m['development']['teacher_nll']:.4f} / {m['reference_development']['teacher_nll']:.4f} |" for m in models)
    next_text=('Reliable cache fitting is now demonstrated for these eight graphs. Inspect the retained cache-versus-held-out digit/source/hop errors before freezing one generalization treatment; do not treat memorization as algorithm learning or clear OOD/readiness. A later cache-diversity or token-budget change must be its own declared ablation.'
               if r['small_cache_control_passed'] else
               'Preserve the negative endpoint and all arms. Audit retained conditional versus generated fitting errors and optimization traces before choosing one new treatment. Do not extend this run, reduce the cache again by outcome, relax a margin, discard an arm or select an earlier checkpoint.')
    text=f'''# Stage 15: small-cache overfit control

Completed local CPU development study, 31 October 2026.

**{lead}** This tests fitting of a declared cache, not unseen-graph algorithm
execution or GraphNCA acceleration. The original random-rule readiness failure,
categorical bridge and stage 13 negative full-ID endpoint remain intact. OOD is
gated regardless of this positive-control result.

## Frozen one-factor treatment and reference

Plan: `{summary['plan_sha256']}`. The [frozen protocol](FROZEN_PROTOCOL.md)
precedes new fits. Only training cache size changes, 512 to 16 examples: four
graphs per size (8,10), both BFS/Dijkstra tasks, four examples per task/size cell.
Selection uses the SAME outcome-blind salt/ranking as stage 14, nested inside its
128-example training probe and the original 512 cache. These are inspected
development data, not fresh confirmation or independent graph cohorts.

ALL five arms restart from their original complete stage 13 step-0 language
models, including the learned tied heads, with fresh optimizer state. No model
continues its old step-600 fit. Architecture, GPT-2 vocabulary/context, prompt/
answer/EOS format, answer-only loss/masks, full-model fp32 CPU AdamW, clipping,
warmup/cosine LR, batch 8, seed and 600-update budget are unchanged. Numerical
update-equivalence and reference-subset tests pass before the freeze.

The verified historical 512-cache fits supply an IMPORTED reference. Their final
stage 14 per-case evidence is reaggregated onto the SAME selected 16 examples and
the SAME full 128 development examples, preserving raw reference outputs and
teacher counters. No reference fit or GPU measurement is rerun or relabeled.

| Per arm | Small-cache treatment | Historical reference |
| --- | ---: | ---: |
| Distinct training examples | 16 | 512 |
| Optimizer updates | 600 | 600 |
| Example presentations | 4800 | 4800 |
| Presentations per selected example | 300 | 9–10 |
| Unpadded input tokens | {counts['input_tokens']:,} | {old['input_tokens']:,} |
| Supervised answer/EOS targets | {counts['supervised_targets']:,} | {old['supervised_targets']:,} |
| Padded input positions | {counts['padded_positions']:,} | {old['padded_positions']:,} |

Cache diversity/content/repetition and lengths are coupled consequences of this
cache treatment; sample means/identities are in plan.json. Fixed updates and
presentations do NOT imply identical tokens, FLOPs, time or isolated repetition
causality. Same-cache evaluation controls the evaluation examples, not those
training-distribution consequences.

## Fixed endpoint results

| Arm | Small-cache fit, same 16 | 512-reference, same 16 | Small-cache model, ID-dev128 | 512-reference, ID-dev128 | Fit criterion |
| --- | ---: | ---: | ---: | ---: | --- |
{table}

{heldout} Exact-vector scores alone do not describe all regressions or gains;
format validity and node/token scores below retain the broader held-out behavior.

The frozen criterion requires EVERY arm to reach >=80% exact in EVERY four-
example cache cell, >=95% pooled valid vectors and complete coverage. Thus each
cell needs 4/4. Final per-cell scores, all malformed completions and teacher
counts remain in the [140-cell CSV](cells.csv) and complete results JSON. New
development is measured only at fixed update 600 and never selects an endpoint.
Its original full-ID criterion is separately reported, not replaced by cache fit.

First recorded pooled 80% CACHE crossing, without interpolation:

| Arm | Update |
| --- | --- |
{crossings}

![Cache fitting curves and fixed-endpoint held-out comparison](cache_fit.png)

Lines connect only the four recorded cache evaluations. Coincident curves can
overlap. The pooled crossing line is not the stricter final per-cell decision;
no independent-cohort confidence interval or learning-speed ratio is inferred.

| Arm | Valid vectors, cache /dev | Teacher-forced digit accuracy, cache /dev | Generated node accuracy, cache /dev |
| --- | ---: | ---: | ---: |
{details}

On the identical held-out data, SMALL16 / HISTORICAL512 continuous metrics are:

| Arm | Development valid vectors | Development node accuracy | Development answer/EOS NLL |
| --- | ---: | ---: | ---: |
{dev_compare}

These retain endpoint regressions/gains even when both exact-vector scores are
zero. They are single-cohort descriptive comparisons, not independent replications.

Teacher forcing supplies gold earlier answers; free generation sees prompt only.
Digit-bearing tokens can span punctuation, and conditional token accuracy is
not semantic per-node correctness. Source/one-hop/multi-hop and signed/first-node
errors remain in results; invalid vectors count wrong in unconditional node/hop
rates. No hints, logit constraints, context extension, truncation, new token or
head reset is used.

## Verification, costs and limits

All 20 new model/optimizer checkpoints and {v['new_predictions_replayed']} new
cache/development predictions replay exactly in a separate invocation, including
teacher scores, initial complete tensor hashes, optimizer steps, actual order/
token counters, final flags and cache crossings. Imported reference metrics are
reconstructed from unchanged stage 14 evidence. This is same-code replay, not
independent neural implementation or repeated fitting.

New scientific optimizer updates: 3000 CPU task-adaptation updates. Historical
reference updates: 3000 IMPORTED, not newly executed. New PPT/language updates,
OOD predictions, GPU measurements and official-language-test access: zero.
Summed new CPU update timers: {summary['new_training_seconds']:.3f}s; cache/development
evaluation timers: {summary['new_evaluation_seconds']:.3f}s. Those exclude initial
provenance preparation, IO and separate replay; command wall time is retained
in the append-only log. Historical reference timing/costs stay in stage 13 and
are not treated as zero or a newly timed matched throughput comparison.

[Provenance](PROVENANCE.json) binds {len(names)} exact evidence copies. Private
weights/optimizer states/raw caches stay excluded. Public code regenerates
selected data/order, reaggregates references and rescores all 960 new generated
predictions without private weights/data, tokenizer cache or network. Neural/
new teacher-forced scores require retained weights. Read the
[usage guide](../../OVERFIT_V15_USAGE.md). No remote/paid job, email, push or
external publication is launched. Inherited provenance can rehash old assessment
bytes; only the selected cache and ID-dev enter this new fit/evaluation.

## Next decision

{next_text}

Any future graph-specific claim still needs eligible synthetic initialization,
the separate categorical/random-rule bridge, fresh paired cohorts, identical
full language/adaptation procedures, all controls, ID-only selection, frozen
OOD criteria/censoring/multiplicity and total cost accounting. A cache-fitting
control cannot establish graph no-effect/equivalence, learning acceleration or
cost recovery.
'''
    (public/'REVIEW.md').write_text(text,encoding='utf-8',newline='\n')
    print('EXPORTED',len(names),'exact evidence copies, 140 cells; control pass',r['small_cache_control_passed'])


if __name__=='__main__':main()
