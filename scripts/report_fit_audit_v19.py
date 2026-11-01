"""Publish the fixed seen/held-out diagnostic without retuning any endpoint."""
import csv
import os
from pathlib import Path
import shutil
from statistics import mean
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fit_audit_v19.run import verify_plan
from ood_v12.plan import read
from transfer_v4.common import dump,sha_file

RUN=ROOT/'runs/fit_audit_v19_run1';PUBLIC=ROOT/'docs/results/fit_audit_v19_run1'


def point(m):
    return {key:m['generated'][key] for key in ('exact_match','valid_format','node_accuracy','coverage')} | {
      'non_source_macro_accuracy':mean(v['accuracy'] for v in m['semantic']['cells']),
      **{'teacher_'+key:mean(v[key] for v in m['teacher_forced']['cells']) for key in ('nll','digit_accuracy','other_accuracy')}}


def summary(r):
    return {'plan_sha256':r['plan_sha256'],'models':[{'arm':m['arm'],'train':point(m['train']),
      'development':point(m['development']),'decision':m['decision']} for m in r['models']],
      'new_optimizer_updates':0,'new_predictions':640,'imported_development_predictions':640,
      'new_ood_predictions':0,'confirmation_allowed':False,
      'new_evaluation_seconds':sum(m['evaluation_seconds'] for m in r['models'])}


def tables(r,baseline):
    cells=[];blocks=[]
    for m in r['models']:
        for split in ('train','development'):
            metrics=m[split];tf={(v['task'],v['n']):v for v in metrics['teacher_forced']['cells']}
            sem={(v['task'],v['n']):v for v in metrics['semantic']['cells']}
            base={(v['task'],v['n']):v['accuracy'] for v in baseline[split]['cells']}
            for v in metrics['generated']['cells']:
                key=v['task'],v['n'];value=sem[key]
                cells.append({'arm':m['arm'],'probe':split,**v,'non_source_correct':value['correct'],
                  'non_source_nodes':value['nodes'],'non_source_accuracy':value['accuracy'],'baseline_non_source_accuracy':base[key],
                  'baseline_advantage':value['accuracy']-base[key],
                  **{'teacher_'+k:tf[key][k] for k in ('nll','digit_accuracy','other_accuracy','targets')}})
        blocks += [{'arm':m['arm'],**b} for b in r['optimization'][m['arm']]['blocks']]
    return cells,blocks


def render(p,r,s):
    lines=lambda values:'\n'.join(values);baselines=read(PUBLIC/'baselines.json')['probes']
    headline=('No arm establishes the semantic target on its SEEN probe.' if all(not m['decision']['training_semantic_ready'] for m in r['models']) else
              'Seen-probe and held-out semantic decisions differ by arm; retain the full table.')
    rows=[];tokens=[];loss=[];controls=[]
    for split in ('train','development'):
        for v in baselines[split]['cells']:
            controls.append(f"| {split} | {v['task']} | {v['n']} | {v['accuracy']:.2%} | {v['accuracy']+.1:.2%} |")
    for m in s['models']:
        a,b=m['train'],m['development'];d=m['decision'];arm=m['arm']
        rows.append(f"| {arm} | {a['exact_match']:.2%} / {b['exact_match']:.2%} | {a['non_source_macro_accuracy']:.2%} / {b['non_source_macro_accuracy']:.2%} | {a['node_accuracy']:.2%} / {b['node_accuracy']:.2%} | {d['training_semantic_ready']} / {d['development_semantic_ready']} |")
        tokens.append(f"| {arm} | {a['valid_format']:.2%} / {b['valid_format']:.2%} | {a['teacher_digit_accuracy']:.2%} / {b['teacher_digit_accuracy']:.2%} | {a['teacher_other_accuracy']:.2%} / {b['teacher_other_accuracy']:.2%} | {a['teacher_nll']:.4f} / {b['teacher_nll']:.4f} |")
        opt=r['optimization'][arm];loss.append(f"| {arm} | {opt['first50_mean_loss']:.4f} | {opt['last50_mean_loss']:.4f} | {opt['unique_examples']} | {opt['presentations']} | {d['interpretation']} |")
    return f'''# Stage19: seen-probe fitting, held-out prediction and actual optimization

Completed local CPU checkpoint-only development diagnostic, 1 November 2026.
**{headline}** No fitting or additional update is performed. All original
stage18 outcomes, censoring, controls, costs and confirmation/OOD gates remain.

## Frozen question and evidence

Plan `{r['plan_sha256']}` and the [frozen protocol](FROZEN_PROTOCOL.md) precede
new seen-probe prediction. Evaluate ALL five fixed update600 endpoints only.
The balanced128 seen cases are an outcome-blind salted graph-ID subset of the
original512,32 cases per task/size cell,64 graphs retaining both tasks. Every
selected example has9 or10 verified prior presentations. No selected earlier
checkpoint, extra fit, different objective, decoding constraint or head reset.

The SAME full development128 predictions from stage18 are copied byte-for-byte;
metrics/teacher counters are exact-value extractions from its original results.
They are not newly predicted or independently neurally replayed here.
Train/dev have identical task/size counts, not identical covariates; full edge/
source-degree/target-distance and graph counts are in plan.json. Tasks on the
same graph are correlated; no independent-case hypothesis test or CI is inferred.

Keep the original marginal predictor fitted on ALL512 training labels. It knows
the source but not adjacency/path computation. The same predictor is scored on
each probe's OWN cases, so its empirical rates and +10pp targets can differ:

| Probe | Task | Size | Baseline non-source accuracy | Required accuracy |
| --- | --- | --- | --- | --- |
{lines(controls)}

Every cell must meet its own target together with >=95% pooled validity and
coverage1. The development baseline/targets remain EXACTLY stage18. The old80%
exact-vector calibration is separate and retained in each model's decision.
Invalid vectors count every semantic node wrong. No source-zero shortcut.

## Fixed endpoint: seen / held-out

| Arm | Exact vectors seen/dev | Non-source accuracy, cell mean seen/dev | Whole-node accuracy, example mean seen/dev | Semantic target seen/dev |
| --- | --- | --- | --- | --- |
{lines(rows)}

Teacher-forced token scores supply gold earlier answers; digit-bearing tokens
can include punctuation. They are not free-generation semantic node accuracies.
The table below uses per-task/size-cell means, preserving the original scorer:

| Arm | Valid vectors seen/dev | Teacher digit accuracy seen/dev | Teacher other accuracy seen/dev | Teacher answer/EOS NLL seen/dev |
| --- | --- | --- | --- | --- |
{lines(tokens)}

[All40 cell rows](cells.csv) and [complete results](results.json) retain exact
and semantic counts, baseline advantages, source/one-hop/multi-hop errors,
signed/first-error details, coverage and both exact/semantic decisions. The
observed train/dev difference is descriptive, not a uniquely identified cause.
This selected seen probe is not an exhaustive measure of all512 training cases.

## Actual optimization and exposure

Import all600 updates per arm unchanged, not reconstructed from configured CI.
All sampled IDs/LR/input/target/padding/cumulative counts are checked against the
original frozen order and actual encodings. Fixed100-update blocks retain mean
loss, mean/min/max pre-clip norm, fraction above1, mean LR and actual tokens.
There is no best-block or best-checkpoint choice. [All30 blocks](trace_blocks.csv)
retain every predeclared block; full exact trace copies are under optimization.

| Arm | First50 update mean loss | Last50 update mean loss | Unique training cases | Presentations | Predeclared interpretation |
| --- | --- | --- | --- | --- | --- |
{lines(loss)}

![Seen versus held-out scores and full optimization block means](fit_and_trace.png)

The left panel separates model scores and each probe's own baseline; it is a
cell-mean non-source score, not exact match or micro-token accuracy. The right
panel connects six fixed block means, not an interpolated learning-speed target.
Loss mixes semantic/format/EOS targets and changing example exposures/LR. A lower
loss alone does not prove semantic fitting, convergence, a unique optimization
defect or that additional updates will repair generalization.

## Verification, costs and next decision

Separate same-code replay reloads all five endpoints and reproduces all640 NEW
seen generations/teacher scores, full600-step optimizer/count receipts, decisions
and trace summaries. Stage18's640 held-out predictions and35/4480 earlier replay
receipt are IMPORTED, not relabeled as new neural measurements. Public-only
reconstruction regenerates ID data/probe/order/baselines and independently parses
all1280 new/imported outputs, trace and cell counts without private weights,
cached raw data, tokenizer cache or network. It reaggregates teacher counters;
neural logits and optimizer weights require private replay.

[Provenance](PROVENANCE.json) binds36 exact evidence copies. Private checkpoint
weights, arrays and raw caches remain excluded; explicitly selected imported
synthetic-ID update traces contain identities/numeric records, not corpora or
weights. [Usage](../../FIT_AUDIT_V19_USAGE.md) documents execution/reproduction.
New scientific optimizer/PPT/language/GPU/OOD/official-test updates:ZERO.
New seen evaluation timers sum{s['new_evaluation_seconds']:.3f}s; preparation,
IO/replay/export command-wall costs are retained in the append-only logger.
Original costs remain in stage18 and are not zero-cost or newly timed here.
No paid/remote job, email, push or external publication occurs.

If seen readiness fails, preserve that unresolved fitting prerequisite alongside
held-out weakness; do not attribute everything to generalization alone. If seen
passes while development fails, preserve that gap without diagnosing a unique
mechanism. These are probe-specific development results, not eligibility to
relabel the old all-censored acceleration endpoint. A longer-budget experiment,
if chosen later, must keep data/architecture/objective/controls fixed and freeze
its schedule, budget, target/censoring and costs separately before fitting.

No automatic OOD/full-language run follows. Original categorical/random-rule
readiness, eligible starts, fresh independent paired cohorts, identical full
language/adaptation stages, all controls, ID-only selection and frozen OOD/
censoring/multiplicity/cost criteria remain under the separate confirmation
protocol. This diagnostic is not graph equivalence or acceleration evidence.
'''


def main():
    p=verify_plan(RUN);r=read(RUN/'results.json');v=read(RUN/'verification.json')
    if not v['passed'] or v['checkpoints']!=5 or v['new_predictions_replayed']!=640 or v['results_sha256']!=sha_file(RUN/'results.json'):raise ValueError('Complete separate replay required')
    names=['plan.json','plan_receipt.json','FROZEN_PROTOCOL.md','evaluation_started.json','results.json','results_receipt.json','verification.json']
    names += [n for n in p['inputs_sha256'] if not n.startswith('data/')]
    for m in r['models']:names += [f'models/{m["arm"]}/summary.json',f'models/{m["arm"]}/train_predictions.jsonl']
    if len(names)!=36:raise ValueError('Keep all36 exact copies')
    PUBLIC.mkdir(parents=True,exist_ok=False)
    for name in names:
        path=PUBLIC/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(RUN/name,path)
    dump(PUBLIC/'PROVENANCE.json',{'files':{n:sha_file(PUBLIC/n) for n in names},'scope':'New640 seen predictions; imported640 held-out predictions and optimization traces;zero updates'})
    cells,blocks=tables(r,read(PUBLIC/'baselines.json')['probes'])
    for name,values,count in (('cells.csv',cells,40),('trace_blocks.csv',blocks,30)):
        if len(values)!=count:raise ValueError('Complete table required')
        with (PUBLIC/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for row in values for k in row}));w.writeheader();w.writerows(values)
    s=summary(r);os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.uv-cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));b=read(PUBLIC/'baselines.json')['probes']
    for split,offset,color,label in (('train',-.3,'#1672b8','seen model'),('train',-.1,'#9cb8c9','seen marginal'),
                                    ('development',.1,'#b76632','held-out model'),('development',.3,'#d7b99a','held-out marginal')):
        values=[m[split]['non_source_macro_accuracy'] for m in s['models']] if 'model' in label else [mean(v['accuracy'] for v in b[split]['cells'])]*5
        axes[0].bar([i+offset for i in range(5)],values,width=.19,color=color,label=label)
    axes[0].set(title='Fixed600: balanced128 probes',ylabel='Non-source accuracy (cell mean)',ylim=(0,1),xticks=range(5),xticklabels=[m['arm'] for m in s['models']]);axes[0].tick_params(axis='x',labelrotation=20);axes[0].legend(frameon=False,fontsize=8)
    for m,color in zip(r['models'],['#555555','#1672b8','#d2600a','#39854a','#8657a6']):
        values=r['optimization'][m['arm']]['blocks'];axes[1].plot([v['last_update'] for v in values],[v['mean_loss'] for v in values],color=color,marker='o',label=m['arm'])
    axes[1].set(title='All six fixed100-update blocks',xlabel='Last update in block',ylabel='Mean training answer/EOS loss');axes[1].legend(frameon=False,fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('Stage19: checkpoint-only fitting/generalization diagnostic; one historical cohort');fig.tight_layout();fig.savefig(PUBLIC/'fit_and_trace.png',dpi=160);plt.close(fig)
    s['plot_sha256']=sha_file(PUBLIC/'fit_and_trace.png');dump(PUBLIC/'summary.json',s)
    (PUBLIC/'REVIEW.md').write_text(render(p,r,s),encoding='utf-8',newline='\n')
    print('EXPORTED36 exact copies,40 cells,30 trace blocks;zero updates')


if __name__=='__main__':main()
