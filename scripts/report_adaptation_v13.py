"""Publish compact verified calibration evidence, excluding private model weights."""
import csv
import os
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptation_v13.run import verify_plan
from ood_v12.plan import read
from transfer_v4.common import dump, sha_file


def plot_curves(out, result):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.uv-cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    colors = ['#555555','#1672b8','#d2600a','#39854a','#8657a6']
    for m, color in zip(result['models'], colors):
        for ax, metric in zip(axes, ('valid_format','exact_match','node_accuracy')):
            ax.plot([r['step'] for r in m['curve']], [r[metric] for r in m['curve']],
                    label=m['arm'], color=color, marker='o', linewidth=1.5, markersize=3)
    for ax, title in zip(axes, ('Valid distance vectors','Exact vector correctness','Node accuracy')):
        ax.set(title=title, xlabel='Adaptation updates', ylim=(-.03,1.03), xticks=[0,100,300,600])
        ax.grid(alpha=.2)
    axes[0].axhline(.95,color='black',linestyle=':',linewidth=1)
    axes[1].axhline(.8,color='black',linestyle=':',linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,fontsize=8)
    fig.suptitle('Stage 13: held-out ID development; one historical cohort')
    fig.tight_layout(rect=(0,.10,1,.95))
    fig.savefig(out/'learning_curves.png',dpi=160)
    plt.close(fig)


def main():
    run = ROOT/'runs/adaptation_v13_run1'
    out = ROOT/'docs/results/adaptation_v13_run1'
    p = verify_plan(run)
    result, verification = read(run/'results.json'), read(run/'verification.json')
    if not verification['passed'] or verification['results_sha256'] != sha_file(run/'results.json'):
        raise ValueError('Complete separate checkpoint replay required')
    names = ['plan.json','plan_receipt.json','FROZEN_PROTOCOL.md','fit_started.json','results.json','results_receipt.json','verification.json']
    for m in result['models']:
        names.append(f"models/{m['arm']}/summary.json")
        names += [n for n in m['files_sha256'] if '/predictions_' in n]
    out.mkdir(parents=True,exist_ok=False)
    for name in names:
        path = out/name
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(run/name,path)
    dump(out/'PROVENANCE.json', {'files': {n:sha_file(out/n) for n in names},
                              'scope':'New local CPU ID adaptation and replay; historical language/PPT costs remain separate'})
    flat = [dict(arm=m['arm'],step=r['step'],input_tokens=r['tokens'],supervised_targets=r['targets'],
                 training_seconds=r['seconds'],evaluation_seconds=r['evaluation_seconds'],
                 train_answer_eos_nll=r['train_nll']['answer_eos_nll'],dev_answer_eos_nll=r['development_nll']['answer_eos_nll'],
                 **cell) for m in result['models'] for r in m['curve'] for cell in r['cells']]
    with (out/'id_cells.csv').open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    summary = {'plan_sha256':sha_file(run/'plan.json'), 'passed':result['id_calibration_passed'],
               'models':[{'arm':m['arm'],'passed':m['decision']['passed'],'crossing':m['decision']['first_pooled_crossing'],
                          'final':m['curve'][-1]} for m in result['models']],
               'scientific_optimizer_updates':result['new_scientific_optimizer_updates'],
               'per_arm_exposure':p['expected_exposure_per_arm'],
               'training_seconds':sum(m['train_seconds'] for m in result['models']),
               'evaluation_seconds':sum(m['evaluation_seconds'] for m in result['models']),
               'checkpoint_replays':verification['checkpoint_replays'],
               'generated_answers_replayed':verification['generated_answers_replayed'],
               'ood_decision':result['ood_decision'],'new_ood_predictions':0,
               'new_language_updates':0,'new_gpu_measurements':0,'graph_acceleration_claim':None,
               'original_random_rule_readiness':'still_failed'}
    plot_curves(out,result)
    summary['learning_curves_sha256'] = sha_file(out/'learning_curves.png')
    dump(out/'summary.json',summary)
    table = '\n'.join(f"| {m['arm']} | {m['final']['exact_match']:.2%} | {m['final']['valid_format']:.2%} | {m['final']['node_accuracy']:.2%} | {m['final']['train_nll']['answer_eos_nll']:.4f} | {m['final']['development_nll']['answer_eos_nll']:.4f} | {'pass' if m['passed'] else 'fail'} |" for m in summary['models'])
    cells = '\n'.join(f"| {m['arm']} | {cell['task']} | {cell['n']} | {cell['exact_match']:.2%} | {cell['valid_format']:.2%} |" for m in summary['models'] for cell in m['final']['cells'])
    crossings = '\n'.join(f"| {m['arm']} | {m['crossing']['step'] if m['crossing']['reached'] else 'not reached by 600 (censored)'} |" for m in summary['models'])
    outcome = ('The fixed calibration criterion passes for every arm. Only a separately invoked fixed-endpoint development OOD evaluation is eligible; none was launched by this report.'
               if summary['passed'] else 'The fixed calibration criterion fails. No new OOD predictions are made; comparing OOD learning acceleration remains gated. Preserve this endpoint and all arms without extra updates or relaxed targets.')
    text = f'''# Stage 13: identical ID algorithmic adaptation

Completed local CPU development calibration, 30 October 2026.

**{outcome}** This tests a task-interface/trainability prerequisite, not a
confirmatory graph-specific PPT effect. It uses one historical cohort whose
original stronger synthetic readiness remains failed. No new language stage,
PPT stage, remote job or GPU measurement is performed.

## Frozen design and results

Plan SHA-256: `{summary['plan_sha256']}`.
The [frozen protocol](FROZEN_PROTOCOL.md) precedes fitting. Five v4 cohort-59 final
language models share their historical 2,000-update /8,192,000-token language
stage, original head initialization, actual data order and LR schedule. Their
complete learned states enter adaptation unchanged, including inherited tied
heads. Equal head policy preserves different post-language learned weights;
no head is reset or compared as if the trained heads were equal.

All arms receive 600 NEW full-model AdamW adaptation updates, batch 8, the same
512 ID training examples and actual sampled order, optimizer initialization,
warmup/cosine LR schedule and 128-example ID-development evaluation. New stage
cost is separate from historical PPT/language. No additional language budget
is substituted for PPT or algorithmic adaptation.

Each arm sees exactly {p['expected_exposure_per_arm']['input_tokens']:,} unpadded
input tokens, {p['expected_exposure_per_arm']['supervised_targets']:,} answer/EOS
targets and {p['expected_exposure_per_arm']['padded_positions']:,} padded input
positions. The full study performs {summary['scientific_optimizer_updates']:,}
scientific updates. Software fixture updates are separate. Prompt and padding
targets are masked, but all prompt tokens condition answer predictions. Gradient
checks verify the selective output projection against the ordinary full LM.

Final generated answers at update 600:

| Arm | ID exact match | Valid vectors | Node accuracy | Train-probe NLL | ID-dev NLL | Criterion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{table}

The criterion requires every arm to have at least 80% exact match pooled AND in
each task/size cell, 95% pooled valid vectors, and 100% coverage. Do not substitute
teacher-forced NLL or node accuracy for generated-vector correctness. Training
uses answer+EOS targets; inference is prompt-only, unconstrained greedy output
with the original tokenizer, vocabulary, context and decoder. Every ID prompt
plus generation budget fits; no task-specific logit mask or truncation is used.

| Arm | Task | Nodes | Exact match | Valid vectors |
| --- | --- | ---: | ---: | ---: |
{cells}

First observed pooled 80% ID-development exact-match crossing (no interpolation):

| Arm | Update |
| --- | --- |
{crossings}

The [complete curves and cells](id_cells.csv) retain updates 0,100,300,600.
No ratio or significance test is inferred from censored single-cohort crossings.
No checkpoint is selected using OOD. No additional OOD data is generated or parsed into examples by fitting; no OOD
labels/prompts condition training or selection. The inherited stage 12 integrity
verifier DOES rehash its prior assessment bytes. The frozen protocol's literal
"not read" wording overstates this boundary; preserve it and read the
[input-integrity clarification](../../ADAPTATION_V13_INTEGRITY_CLARIFICATION.md).
The reserved official language test remains untouched.

![ID learning curves: format, exact-vector correctness and node accuracy](learning_curves.png)

Lines use the four observed evaluations; coincident lines overlap. Dotted lines
are the frozen pooled format/exact targets. The per-cell exact criterion still
applies. There are no independent-cohort confidence intervals in this pilot.

## Interpretation, evidence and costs

Valid format and task correctness measure different things. A valid incorrect
vector demonstrates interface learning only. NLL includes formatting/EOS and
teacher forcing; it cannot establish algorithm execution or graph mechanism.
The fixed training probe is the FIRST 64 training examples: 32 eight-node base
graphs across both tasks. Development includes 8- AND 10-node graphs, so their
pooled NLL difference also changes composition and must NOT be described as a
matched generalization gap. Per-task/size generation metrics remain separate.

Replay reopens all {verification['checkpoint_replays']} initial/intermediate/final
checkpoints and exactly reproduces {verification['generated_answers_replayed']:,}
generated answers, token IDs, stops, scores and train/development NLL. Every final
optimizer tensor has 600 steps; initial checkpoints have fresh optimizer state.
This is separate same-code replay, not independent fitting or an independent
neural loss implementation. Independent Floyd-Warshall labels and inherited
structural split/cut exclusions are checked before fitting and during replay.

The [provenance](PROVENANCE.json) binds {len(names)} exact evidence copies.
Weights, optimizer states, raw graph datasets, private correspondence and caches
remain private; hashes identify retained private artifacts. Costs: summed CPU
update timers {summary['training_seconds']:.3f} seconds; development generation/
NLL timers {summary['evaluation_seconds']:.3f} seconds. These exclude source
hashing, model construction/checkpoint IO and verification; full command wall
times remain in the append-only log. Sequential order and machine load can affect
timing. Historical GPU PPT/language/generation costs remain imported evidence,
not newly measured or zero. No GPU, training-speedup or cost-recovery claim follows.

Readers can regenerate the exact ID train/development bytes and reproduce all
2,560 exported answer scores/cells from public files without weights, network,
email or tokenizer cache using `scripts/reproduce_adaptation_v13_scores.py`.
See the [usage guide](../../ADAPTATION_V13_USAGE.md). That independent invocation
reproduces scoring/data identities; it is not neural inference, NLL or training.

## Next decision

{outcome}

Before another fit, use the retained checkpoints for a separately frozen
training-versus-development generated-error audit, with matched task/size probes.
This distinguishes poor fitting from held-out generalization more directly than
the composition-mismatched NLL means. Freeze any next treatment separately and
keep all controls. A real
GraphNCA learning-acceleration study still requires the categorical/random-rule
readiness bridge, eligible initializations, fresh independent paired cohorts,
identical full language and adaptation stages, ID-only selection, predeclared
OOD targets/censoring/multiplicity and total costs. Graph must beat BOTH grid
and its matched prefix shuffle, with scratch and prefix-grid retained.
Stage 11 binary confirmation, stage 12 zero-shot floor and all historical negative
results are unchanged. No email, publication, push or remote/paid training occurs.
'''
    (out/'REVIEW.md').write_text(text,encoding='utf-8',newline='\n')
    print('EXPORTED',len(names),'exact copies; fixed ID decision',summary['passed'],'; OOD',summary['ood_decision'])


if __name__ == '__main__':
    main()
