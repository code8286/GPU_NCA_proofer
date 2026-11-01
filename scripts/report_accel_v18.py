"""Export verified ID curves, honest censoring and component cost evidence."""
import csv
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from accel_v18.run import verify_plan,pipeline_at
from ood_v12.plan import read
from transfer_v4.common import sha_file,dump

RUN=ROOT/'runs/accel_v18_run1';PUBLIC=ROOT/'docs/results/accel_v18_run1'


def flatten(r,base):
    targets={(x['task'],x['n']):x['accuracy'] for x in base['cells']};rows=[]
    for m in r['models']:
        for entry in m['curve']:
            metrics=entry['metrics'];tf={(v['task'],v['n']):v for v in metrics['teacher_forced']['cells']}
            ns={(v['task'],v['n']):v for v in entry['semantic']['cells']}
            for cell in metrics['generated']['cells']:
                key=cell['task'],cell['n'];v=ns[key]
                rows.append({'arm':m['arm'],'step':entry['step'],**entry['counts'],
                 'training_seconds':entry['training_seconds'],'evaluation_seconds':entry['evaluation_seconds'],
                 'elapsed_seconds':entry['elapsed_seconds'],'whole_target_met':entry['target_met'],**cell,
                 'non_source_correct':v['correct'],'non_source_nodes':v['nodes'],'non_source_accuracy':v['accuracy'],
                 'marginal_baseline_accuracy':targets[key],
                 **{'teacher_'+k:tf[key][k] for k in ('nll','digit_accuracy','other_accuracy','targets')}})
    return rows


def budget_account(m):
    value=pipeline_at({'reached':True,'step':600,'counts':m['counts'],
                      'elapsed_seconds':m['curve'][-1]['elapsed_seconds']},m['cost'])
    value.pop('reached')
    value['scope']='full-budget spending; not an observed target crossing'
    value['target_met_at_final']=m['curve'][-1]['target_met']
    return value


def compact(r):
    return {'plan_sha256':r['plan_sha256'],'models':[{'arm':m['arm'],'crossing':m['crossing'],
      'sustained_crossing':m['sustained_crossing'],'pipeline_at_crossing':m['pipeline_at_crossing'],
      'final_non_source_cells':m['curve'][-1]['semantic']['cells'],'final_target_met':m['curve'][-1]['target_met'],
      'final_exact':m['curve'][-1]['semantic']['exact_match'],'counts':m['counts'],'cost':m['cost'],
      'full_budget_accounted_pipeline':budget_account(m),
      'training_seconds':m['training_seconds'],'evaluation_seconds':m['evaluation_seconds']} for m in r['models']],
      'contrasts':r['contrasts'],'new_scientific_updates':3000,'new_language_or_ppt_updates':0,
      'new_ood_predictions':0,'confirmation_allowed':False,'original_random_rule_readiness':'still_failed'}


def render(p,r,s):
    lines=lambda values:'\n'.join(values);baseline=read(PUBLIC/'baseline.json');base=baseline['semantic']
    targets=[f"| {v['task']} | {v['n']} | {v['correct']}/{v['nodes']} ({v['accuracy']:.2%}) | {v['accuracy']+p['config']['baseline_margin']:.2%} |" for v in base['cells']]
    table=[];costtable=[];full=[];contrasts=[]
    for m in s['models']:
        cross=m['crossing'];final=m['final_non_source_cells'];step=cross['step'] if cross['reached'] else 'not reached; censored600'
        ns=' / '.join(f'{v["accuracy"]:.2%}' for v in final)
        count=cross['counts'];tokens=f"{count['input_tokens']:,}" if count else 'unreached'
        elapsed=f"{cross['elapsed_seconds']:.3f}" if cross['reached'] else 'unreached'
        table.append(f"| {m['arm']} | {step} | {tokens} | {elapsed} | {ns} | {m['final_exact']:.2%} |")
        c=m['cost'];a=m['full_budget_accounted_pipeline']
        costtable.append(f"| {m['arm']} | {c['ppt_updates']} / {c['ppt_tokens']:,} | {c['ppt_train_seconds']:.3f} / {c['ppt_stage_seconds']:.3f} | {c['paired_generation_seconds']:.3f} / {c['paired_preparation_seconds']:.3f} | {c['language_stage_seconds']:.3f} | {c['original_cumulative_seconds']:.3f} |")
        full.append(f"| {m['arm']} | {m['training_seconds']:.3f} | {m['evaluation_seconds']:.3f} | {a['total_tokens']:,} | {a['accounted_mixed_hardware_seconds']:.3f} |")
    for v in r['contrasts']:
        contrasts.append(f"| {v['reference']} | {v['updates_speedup'] if v['updates_speedup'] is not None else 'undefined'} | {v['input_tokens_speedup'] if v['input_tokens_speedup'] is not None else 'undefined'} | {v['reason'] or 'both observed targets reached'} |")
    reached=sum(m['crossing']['reached'] for m in s['models'])
    targets_by_cell={(v['task'],v['n']):v['accuracy']+p['config']['baseline_margin'] for v in base['cells']}
    cell_passes=sum(v['accuracy']>=targets_by_cell[v['task'],v['n']] for m in r['models'] for e in m['curve'] for v in e['semantic']['cells'])
    conclusion=('All five arms are right-censored: this budget provides no measured time-to-target speedup. Undefined ratios are not evidence of equality, graph harm or impossibility.' if reached==0 else
                f'{reached}/5 arms attain the exploratory target. Only observed positive-denominator crossings support descriptive ratios; this single cohort does not confirm a graph-specific effect.')
    return f'''# Stage18: exploratory ID learning acceleration

Started 31 October 2026; completed 1 November 2026. New CPU task-adaptation fits, with imported T4
PPT/language costs. **{conclusion}** No earlier failed calibration, synthetic
readiness or OOD/confirmation eligibility is changed.

## Prospective target and matched experiment

Plan `{r['plan_sha256']}` and [frozen protocol](FROZEN_PROTOCOL.md) precede the
first new update. User authorized this exploratory test despite the separately
failed original readiness bridge. It measures downstream ID TASK ADAPTATION
after identical language stages, not language-stage convergence or GNN runtime.

Five original stage13 step0 starts retain full learned heads and fresh AdamW.
The shared architecture, tokenizer, masks/objective, data/order, batch, update
budget and LR schedule are fixed across graph/grid/scratch/both prefix controls.
A new ID resample uses train512/dev128, sizes8/10, BFS/Dijkstra, cyclic-ER family;
its assessment count isZERO. No OOD rows or official language test are generated
or evaluated. The source-aware marginal control is fitted to training labels
only; known source outputs zero, other nodes get that task/size's training mode.

The primary target requires >=10pp advantage on unconditional NON-SOURCE node
accuracy in EVERY cell, >=95% pooled vector validity and complete coverage:

| Task | Size | Frozen baseline correct/nodes | Required accuracy |
| --- | --- | --- | --- |
{lines(targets)}

This is a new exploratory semantic target, not retrospective success under the
old80% exact-vector calibration. Invalid vectors are wrong on every node; source
zeros cannot satisfy the non-source criterion. Teacher scores supply gold earlier
answers and token accuracy is not node accuracy. All140 cell rows preserve both.

## ID curves and observed/censored crossings

Every arm completes600 updates and all seven scheduled evaluations, even if it
attains a target earlier. First observed crossing is primary, with no interpolation
or checkpoint selection. Next-observation sustained crossing is a separately
reported sensitivity, including any later regression. Step0 success would be
pre-existing competence, not infinite acceleration. Unreached stays censored.

| Arm | First crossing update | Input tokens to target | CPU adaptation elapsed seconds to target | Final non-source cells: BFS8/BFS10/Dijkstra8/Dijkstra10 | Final exact vectors |
| --- | --- | --- | --- | --- | --- |
{lines(table)}

Individual semantic cell targets met: {cell_passes}/140 observed cell points.
This retains every declared checkpoint and cell, without selecting an endpoint.

![Held-out non-source accuracy curves and frozen cell targets](id_curves.png)

Lines connect recorded observations only; coincident curves can overlap. Gray
lines show the fixed baseline; dotted black lines show its +10pp target. All
four cells plus the validity requirement must pass at the SAME checkpoint.
See [complete results](results.json), [140 cell rows](cells.csv) and
[summary](summary.json) for exact/format/whole-node/source/hop/signed/first-error
scores, teacher NLL/digit/other counters, actual exposures and all timing axes.

Graph contrasts retain BOTH required controls, plus scratch and prefix_grid:

| Reference for graph | Update speedup | Input-token speedup | Interpretation |
| --- | --- | --- | --- |
{lines(contrasts)}

One inherited PPT/language initialization cohort and a new ID sample are not
independent paired rule/topology replicates. No p-values, CI, equivalence,
positive confirmation or graph no-effect conclusion is inferred. Full continuous
curves remain reported even if the primary time-to-target endpoint is censored.

## Imported PPT, generation and language costs

All arms inherit2000 language updates /8192000 text tokens with identical original
head initialization, data order and scheduler. Costs below are IMPORTED Tesla T4
seconds, not new GPU measurements. The entire paired preparation cost is charged
to each synthetic arm as in v4, not an isolated graph-generator throughput figure.

| Arm | PPT updates/tokens | PPT train/stage seconds | Paired generation/preparation seconds | Language stage seconds | Original cumulative seconds |
| --- | --- | --- | --- | --- | --- |
{lines(costtable)}

Language stage seconds include its validation; original cumulative totals also
include text/shared preparation and setup. Do not sum these overlapping columns
again. Scratch has no PPT. Original exclusions include queue/startup/calibration,
initial snapshot and final checkpoint verification/export/archives; unmeasured
costs are not assumed zero. Full input evidence and extracted [costs](costs.json)
are hash-bound before fitting. Token sums are stage ledgers, not equal FLOPs/$.
Historical PPT-gate flags in raw cost summaries are retained verbatim; they
refer to the weaker v4 gate, not the later stronger readiness check that failed.

Full600-update budget accounting, whether or not the target was reached:

| Arm | New CPU train seconds | New CPU eval seconds | PPT+language+adaptation input tokens | Accounted original GPU cumulative + new CPU elapsed seconds |
| --- | --- | --- | --- | --- |
{lines(full)}

The last column is a component execution-cost account across DIFFERENT hardware,
not controlled end-to-end throughput, deployment cost recovery or a target-crossing
time for censored runs. Adaptation elapsed includes evaluation/checkpoint IO;
arms run sequentially in a fixed order, without a controlled hardware crossover.
It starts after CPU model/optimizer construction. Train and eval timers are
separated. Setup/teardown and shared preparation remain in logged command wall
times, without a fabricated per-arm allocation. New preparation/replay/export times are
in the append-only log. No PPT cost is silently omitted or amortized over seeds.

## Verification and next decision

Separate same-code replay reloads35 model/optimizer checkpoints and exactly
reproduces4480 new generations/teacher scores, baseline/semantic decisions,
crossings/censoring/contrasts, initial identities, optimizer updates and actual
order/LR/token counts. Original runtime observations are retained, not expected
to repeat bit-for-bit. Replay is not a second fit or independent neural inference.
Independent parsing/count and public-only data/order/baseline/cost/score checks
complement it. [Usage](../../ACCEL_V18_USAGE.md) gives reproducible commands.

[Provenance](PROVENANCE.json) binds60 exact evidence copies. Private weights,
optimizer states, raw caches and order arrays remain excluded. The exact result
JSON alone has a bounded12MB publication allowance to retain all cases; other
artifact exclusions remain. New scientific updates:3000 CPU task-adaptation.
New PPT/language/GPU/OOD/official-test evaluations:ZERO. No paid/remote training,
email, push or external publication. No altered salt, relaxed margin, dropped arm,
extra update or selected checkpoint follows the observed result.

Preserve this exploratory endpoint. A later study needs its own frozen treatment,
target and budget; this run does not automatically open OOD. Original categorical/
random-rule readiness, eligible starts, fresh independent paired cohorts,
identical full language/adaptation stages, all controls, ID-only selection,
predeclared OOD targets/censoring/multiplicity and full costs remain necessary
under the separate confirmation protocol. No current result proves a universal
GraphNCA advantage, equivalence or impossibility.
'''


def main():
    p=verify_plan(RUN);r=read(RUN/'results.json');v=read(RUN/'verification.json')
    if not v['passed'] or v['checkpoints']!=35 or v['predictions']!=4480 or v['results_sha256']!=sha_file(RUN/'results.json'):raise ValueError('Full checkpoint replay required')
    names=['plan.json','plan_receipt.json','FROZEN_PROTOCOL.md','fit_started.json','results.json','results_receipt.json','verification.json']
    names += [n for n in p['inputs_sha256'] if not n.startswith('data/') and n!='order.npy']
    for m in r['models']:
        names.append(f'models/{m["arm"]}/summary.json');names += [n for n in m['files_sha256'] if '/development_' in n]
    if len(names)!=60:raise ValueError('Keep all60 exact copies')
    PUBLIC.mkdir(parents=True,exist_ok=False)
    for name in names:
        target=PUBLIC/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(RUN/name,target)
    dump(PUBLIC/'PROVENANCE.json',{'files':{n:sha_file(PUBLIC/n) for n in names},'scope':'New CPU exploratory ID fits; imported T4 costs, original gates unchanged'})
    rows=flatten(r,read(PUBLIC/'baseline.json')['semantic'])
    if len(rows)!=140:raise ValueError('Keep all140 cell rows')
    with (PUBLIC/'cells.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}));w.writeheader();w.writerows(rows)
    s=compact(r);os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.uv-cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    base=read(PUBLIC/'baseline.json')['semantic']
    for ax,cell in zip(axes.flat,base['cells']):
        for m,color in zip(r['models'],['#555555','#1672b8','#d2600a','#39854a','#8657a6']):
            values=[next(v['accuracy'] for v in e['semantic']['cells'] if (v['task'],v['n'])==(cell['task'],cell['n'])) for e in m['curve']]
            ax.plot([e['step'] for e in m['curve']],values,label=m['arm'],color=color,marker='o',markersize=3)
        ax.axhline(cell['accuracy'],color='gray',linewidth=1);ax.axhline(cell['accuracy']+.1,color='black',linestyle=':',linewidth=1)
        ax.set(title=f"{cell['task']} /n={cell['n']}",ylim=(0,1),xticks=p['config']['evaluation_updates']);ax.grid(alpha=.2)
    axes[1,0].set_xlabel('ID adaptation updates');axes[1,1].set_xlabel('ID adaptation updates')
    axes[0,0].set_ylabel('Non-source node accuracy');axes[1,0].set_ylabel('Non-source node accuracy')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,bbox_to_anchor=(.5,.95),frameon=False)
    fig.suptitle('Stage18: held-out ID learning curves; prospective baseline +10pp targets');fig.tight_layout(rect=(0,0,1,.9));fig.savefig(PUBLIC/'id_curves.png',dpi=160);plt.close(fig)
    s['plot_sha256']=sha_file(PUBLIC/'id_curves.png');dump(PUBLIC/'summary.json',s)
    (PUBLIC/'REVIEW.md').write_text(render(p,r,s),encoding='utf-8',newline='\n')
    print('EXPORTED60 exact copies/140 cells; reached',sum(m['crossing']['reached'] for m in r['models']),'of5')


if __name__=='__main__':main()
