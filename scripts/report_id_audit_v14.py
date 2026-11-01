"""Export the matched checkpoint audit with interpretation and descriptive plots."""
import csv
import os
from pathlib import Path
import shutil
from statistics import mean
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from id_audit_v14.run import verify_plan
from ood_v12.plan import read
from transfer_v4.common import dump,sha_file


def compact(metrics):
    result = {}
    for split in ('train','development'):
        g = metrics[split]['generated']; t = metrics[split]['teacher_forced']
        row = {k:g[k] for k in ('exact_match','valid_format','node_accuracy','coverage')}
        row |= {'teacher_'+k:mean(c[k] for c in t['cells']) for k in ('nll','accuracy','digit_accuracy','other_accuracy')}
        for hop in ('source','one_hop','multi_hop'):
            n = sum(c.get(hop+'_nodes',0) for c in g['cells'])
            correct = sum(c.get(hop+'_correct',0) for c in g['cells'])
            row[hop+'_accuracy'] = correct/n if n else None
        result[split] = row
    result['interpretation'] = metrics['interpretation']
    return result


def main():
    run,public = ROOT/'runs/id_audit_v14_run1',ROOT/'docs/results/id_audit_v14_run1'
    p = verify_plan(run); r,v = read(run/'results.json'),read(run/'verification.json')
    if not v['passed'] or v['results_sha256'] != sha_file(run/'results.json'):
        raise ValueError('Separate exact checkpoint replay required')
    names = ['plan.json','plan_receipt.json','FROZEN_PROTOCOL.md','evaluation_started.json','results.json','results_receipt.json','verification.json']
    names += [name for m in r['models'] for name in m['files_sha256']]
    public.mkdir(parents=True,exist_ok=False)
    for name in names:
        path = public/name; path.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(run/name,path)
    dump(public/'PROVENANCE.json',{'files':{n:sha_file(public/n) for n in names},'scope':'New CPU checkpoint-only diagnostic; zero new optimizer updates'})
    models = [{'arm':m['arm'],**compact(m['metrics'])} for m in r['models']]
    summary = {'plan_sha256':sha_file(run/'plan.json'),'models':models,'new_optimizer_updates':0,
               'inference_scoring_seconds':sum(m['inference_scoring_seconds'] for m in r['models']),
               'predictions_replayed':v['predictions_replayed'],'original_random_rule_readiness':'still_failed',
               'ood_evaluation_allowed':False,'graph_acceleration_claim':None}
    cells = []
    for m in r['models']:
        for split in ('train','development'):
            metrics = m['metrics'][split]
            tf = {(c['task'],c['n']):c for c in metrics['teacher_forced']['cells']}
            for cell in metrics['generated']['cells']:
                t = tf[cell['task'],cell['n']]
                cells.append({'arm':m['arm'],'split':split,**cell,
                              **{'teacher_'+k:t[k] for k in ('nll','accuracy','digit_accuracy','other_accuracy','targets','digit_targets','other_targets')}})
    fields = sorted({k for row in cells for k in row})
    with (public/'matched_cells.csv').open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(cells)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.uv-cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for ax,key,title in zip(axes,('node_accuracy','teacher_digit_accuracy'),('Free-generation node accuracy','Teacher-forced digit-token accuracy')):
        for split,delta,color in (('train',-.18,'#2079b5'),('development',.18,'#db7c22')):
            ax.bar([i+delta for i in range(5)],[m[split][key] for m in models],width=.35,color=color,label=split)
        ax.set(title=title,ylim=(0,1),xticks=range(5),xticklabels=[m['arm'] for m in models],ylabel='Accuracy')
        ax.tick_params(axis='x',labelrotation=20);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle('Stage 14: fixed endpoints; matched task/size probes; one historical cohort')
    fig.tight_layout();fig.savefig(public/'matched_errors.png',dpi=160);plt.close(fig)
    summary['plot_sha256'] = sha_file(public/'matched_errors.png')
    dump(public/'summary.json',summary)
    all_fit_fail = all(not m['interpretation']['training_probe_competent'] for m in models)
    lead = ('Every arm fails competence on the SEEN training probe as well as development. Held-out generalization alone cannot account for all observed failure.'
            if all_fit_fail else 'Training/development competence differs across arms; retain every fixed interpretation flag below.')
    table = '\n'.join(f"| {m['arm']} | {m['train']['exact_match']:.2%} / {m['development']['exact_match']:.2%} | {m['train']['node_accuracy']:.2%} / {m['development']['node_accuracy']:.2%} | {m['train']['teacher_digit_accuracy']:.2%} / {m['development']['teacher_digit_accuracy']:.2%} | {m['train']['teacher_nll']:.4f} / {m['development']['teacher_nll']:.4f} |" for m in models)
    hops = '\n'.join(f"| {m['arm']} | {split} | {m[split]['source_accuracy']:.2%} | {m[split]['one_hop_accuracy']:.2%} | {m[split]['multi_hop_accuracy']:.2%} |" for m in models for split in ('train','development'))
    flags = '\n'.join(f"- {m['arm']}: `{m['interpretation']['interpretation']}`; exact train-minus-dev {m['interpretation']['train_minus_development_exact']:.2%}." for m in models)
    exposures = list(p['original_training_exposures'].values())
    text = f'''# Stage 14: matched training/development error audit

Completed local CPU checkpoint-only diagnostic, 31 October 2026.

**{lead}** This establishes a limitation on the selected training examples;
it does not identify a unique optimization, capacity, data, representation or
rollout mechanism. No new fitting is performed, and OOD remains gated.

## Frozen scope and observations

Plan: `{summary['plan_sha256']}`. The [frozen protocol](FROZEN_PROTOCOL.md)
precedes new predictions. Retain ALL five stage 13 fixed final checkpoints at
600 adaptation updates. No best endpoint or arm is selected. The stage 13
failed criterion, historical language procedure, stronger synthetic-readiness
failure and earlier controls/results remain unchanged.

Both probes have 128 examples /64 base graphs: 32 per task/size cell for BFS
and Dijkstra at 8 and 10 nodes, native cyclic ER graphs. Training selection is
by frozen salted base-ID hash, including both tasks per graph; it does not
depend on labels or outputs. Each selected training example was actually used
{min(exposures)}–{max(exposures)} times in stage 13; development examples were
never in its optimizer batches. These are a quarter of the training cache, not
an exhaustive fitting census. Development was already inspected in stage 13;
no untouched-confirmation claim applies. No independent graph-cohort CI/p-value.

Each table entry is TRAIN / DEVELOPMENT. NLL and digit accuracy are macro means
of the four matched task/size cells; NLL is token-weighted within each cell.
The earlier eight-node-only training NLL composition mismatch is avoided.

| Arm | Exact vectors | Generated node accuracy | Teacher-forced digit tokens | Answer/EOS NLL |
| --- | ---: | ---: | ---: | ---: |
{table}

Teacher-forced OTHER-token accuracy is
{min(m['train']['teacher_other_accuracy'] for m in models):.2%}–{max(m['train']['teacher_other_accuracy'] for m in models):.2%}
on training and
{min(m['development']['teacher_other_accuracy'] for m in models):.2%}–{max(m['development']['teacher_other_accuracy'] for m in models):.2%}
on development. The digit-bearing subset remains
{min(m['train']['teacher_digit_accuracy'] for m in models):.2%}–{max(m['train']['teacher_digit_accuracy'] for m in models):.2%}
on training and
{min(m['development']['teacher_digit_accuracy'] for m in models):.2%}–{max(m['development']['teacher_digit_accuracy'] for m in models):.2%}
on development. This separates easy conditional formatting from weak conditional
value-token prediction. It does not identify a unique reason for that weakness.

![Matched seen/held-out node and digit-token errors](matched_errors.png)

Gold earlier answer tokens condition teacher-forced predictions; prompt-only
greedy generation receives no gold answer. Digit-bearing GPT-2 tokens can span
signs or punctuation; their accuracy is not semantic integer accuracy. NLL and
teacher-forced accuracy cannot replace full generated-vector correctness.
The [40-cell CSV](matched_cells.csv) and complete per-case result JSON retain
format validity, coverage, signed errors, first wrong positions, all token
counts and per-task/size differences. Invalid vectors contribute zero node
correctness and remain in every unconditional source/hop denominator.

| Arm | Probe | Source node | One-hop nodes | Multi-hop nodes |
| --- | --- | ---: | ---: | ---: |
{hops}

Hop categories use unweighted connectivity for BOTH tasks, including weighted
Dijkstra. Rates above use all relevant node counts, including malformed outputs;
they are descriptive micro rates within a probe. Task/size cells remain in the
CSV. Matching does not equalize degree, distances or token lengths: descriptive
covariates for each cell are preserved in `plan.json`. Across-split differences
are unpaired and do not prove a causal generalization mechanism.

Frozen interpretation flags (80% exact in every cell, 95% pooled format and
complete coverage are required for probe competence):

{flags}

## Verification and provenance

Independent Floyd-Warshall checks every selected label; original stage 12
structural/cut split exclusion receipts and stage 13 actual exposure identities
remain intact. Prior integrity verification can rehash old assessment bytes;
only ID probes enter new prediction/scoring. Original development generations
reproduce stage 13 final token IDs/text/stops exactly for all five models.

A separate process reopens five checkpoints and exactly reproduces all
{v['predictions_replayed']:,} new train/development generations, teacher-forced
scores, node errors and fixed interpretation flags. Model tensors and checkpoint
hashes stay identical before/after inference, and retained optimizer steps
remain 600. This is same-code replay, not independently implemented neural
inference or another fitting run. New scientific optimizer updates: **zero**.

[Provenance](PROVENANCE.json) binds {len(names)} exact evidence copies. Private
weights, datasets, caches and correspondence remain excluded. The public
`scripts/reproduce_id_audit_v14.py` regenerates exact matched probe bytes and
rescores all 1280 exported generated answers without private weights, data,
tokenizer cache or network. It cannot reproduce neural/teacher-forced scores
without weights. See the [usage guide](../../ID_AUDIT_V14_USAGE.md).

Summed new per-model inference/scoring timers: {summary['inference_scoring_seconds']:.3f}s,
including model construction/load, both generation probes and teacher-forced
measurements, excluding initial provenance preparation and separate replay.
Full command wall times remain logged. Original PPT/language/adaptation costs
are historical evidence, not zero or newly timed. No GPU/learning-speed ratio,
graph equivalence/no-effect, causal mechanism or cost recovery is established.

## Next decision

{lead}

Keep OOD and further full language suites gated. The next economical proposal
is a separately frozen small-cache overfit positive control: keep architecture,
tokenizer, output format, objective and update budget fixed, and change training
cache size alone. Retain the original cache reference and all five arms from
their same pre-adaptation starts. This can test whether the procedure can fit a
small declared set reliably before attempting broader generalization; it does
not promise that reducing or increasing data fixes the failure. It is a proposal,
not a new fit or an extension of stage 13. Do not relax its target, drop a control,
select a checkpoint or change multiple factors together. This audit identifies
seen-example competence as an unresolved prerequisite, not a unique cause.

The categorical/random-rule readiness bridge and eligible synthetic starts
remain separately necessary before a fresh graph-transfer study. Retain scratch,
grid and both prefix controls, identical language/adaptation stages, fresh paired
cohorts, ID-only selection, frozen OOD criteria and total costs. No remote/paid
job, official language test, email, push or external publication occurs here.
'''
    (public/'REVIEW.md').write_text(text,encoding='utf-8',newline='\n')
    print('EXPORTED',len(names),'exact copies; 40 matched cells; zero updates;',lead)


if __name__ == '__main__': main()
