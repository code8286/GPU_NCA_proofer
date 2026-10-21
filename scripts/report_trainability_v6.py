"""Export compact, verified CPU trainability evidence without weights or caches."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import dump, sha_file
from readiness_v5.gate import decision
from trainability_v6.plan import verify_plan, read
from trainability_v6.run import verify_training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'runs/trainability_v6_run1')
    p.add_argument('--out', type=Path, default=ROOT/'docs/results/trainability_v6_run1')
    a = p.parse_args()
    plan = verify_plan(a.run)
    training = verify_training(a.run, plan)
    result, verification = read(a.run/'results.json'), read(a.run/'verification.json')
    result_hash = sha_file(a.run/'results.json')
    if result_hash != read(a.run/'results_receipt.json')['results_sha256'] or result_hash != verification['results_sha256'] or not verification['passed']:
        raise ValueError('Verified result identity differs')
    c = plan['config']
    if len(result['cells']) != 6*len(c['seeds']) or result['language_allowed']:
        raise ValueError('Incomplete diagnostic or forbidden language state')
    a.out.mkdir(parents=True, exist_ok=False)
    copies = ['plan.json', 'plan_receipt.json', 'FROZEN_PROTOCOL.md', 'training_complete.json', 'training_receipt.json',
              'results.json', 'results_receipt.json', 'verification.json', 'status.json']
    copies += [f"{r['arm']}_{r['seed']}/{name}" for r in training for name in ('summary.json', 'metrics.jsonl')]
    provenance = {'scope': 'New CPU controlled-rule transformer training and replay; no language-transfer result', 'files': {}}
    for name in copies:
        dest = a.out/name
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(a.run/name, dest)
        provenance['files'][name] = {'sha256': sha_file(dest), 'origin': 'Frozen CPU trainability run: '+name}
    dump(a.out/'PROVENANCE.json', provenance)
    cells = []
    for r in result['cells']:
        rows = r['per_initial_condition']
        count = sum(v['patch_targets'] for v in rows)
        cells.append({k:r[k] for k in ('seed', 'checkpoint_arm', 'data_arm', 'patch_nll')} |
                     {'patch_targets': count, 'patch_accuracy': sum(v['patch_accuracy']*v['patch_targets'] for v in rows)/count})
    with (a.out/'assessment_cells.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)
    coherent = [r for r in cells if r['checkpoint_arm'] == r['data_arm'] == 'coherent']
    by_cell = {(r['seed'], r['checkpoint_arm'], r['data_arm']):r for r in cells}
    prefix_passes = sum(decision(by_cell[s, 'prefix', 'coherent']['patch_nll'], by_cell[s, 'prefix', 'prefix']['patch_nll'],
                                 result['frequency_nll'], c)['passed'] for s in c['seeds'])
    prefix_gaps = [r['prefix_model_ordering_gap'] for r in result['decisions']]
    coherent_gaps = [r['gate']['ordering_advantage_nats'] for r in result['decisions']]
    prefix_accuracy = statistics.mean(by_cell[s, 'prefix', 'coherent']['patch_accuracy'] for s in c['seeds'])
    summary = {'trainability_passed': result['trainability_passed'], 'language_allowed': False,
               'models_trained': len(training), 'replayed_cells': verification['replayed_cells'],
               'frequency_nll': result['frequency_nll'],
               'mean_coherent_nll': statistics.mean(r['patch_nll'] for r in coherent),
               'mean_coherent_accuracy': statistics.mean(r['patch_accuracy'] for r in coherent),
               'frequency_passes': sum(r['gate']['frequency_passed'] for r in result['decisions']),
               'ordering_passes': sum(r['gate']['ordering_passed'] for r in result['decisions']),
               'both_passes': sum(r['gate']['passed'] for r in result['decisions']),
               'mean_frequency_advantage_nats': statistics.mean(r['gate']['frequency_advantage_nats'] for r in result['decisions']),
               'mean_ordering_advantage_nats': statistics.mean(r['gate']['ordering_advantage_nats'] for r in result['decisions']),
               'recorded_training_stage_seconds': sum(r['stage_seconds'] for r in training),
               'total_updates': sum(r['steps'] for r in training), 'total_input_tokens': sum(r['input_tokens'] for r in training),
               'backbone_parameters': training[0]['backbone_parameters'], 'model_parameters': training[0]['parameter_count'],
               'plan_sha256': sha_file(a.run/'plan.json')}
    dump(a.out/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for i, seed in enumerate(c['seeds']):
        color = ('#2563eb', '#059669', '#b45309')[i%3]
        for arm, style in (('coherent', '-'), ('prefix', '--')):
            metrics = [json.loads(line) for line in (a.run/f'{arm}_{seed}/metrics.jsonl').read_text().splitlines()]
            ax.plot([m['step'] for m in metrics], [m['development']['patch_nll'] for m in metrics],
                    style, color=color, label=f'{arm}, seed {seed}')
    ax.set(title='Controlled complement rule: development loss during CPU training',
           xlabel='Optimizer updates (fixed endpoint: 1000)', ylabel='Native development patch NLL')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
    for extension in ('png', 'svg'):
        fig.savefig(a.out/f'learning_curves.{extension}', dpi=180)
    plt.close(fig)
    table = '\n'.join(f"| {r['seed']} | {r['coherent_nll']:.6f} | {r['shuffled_nll']:.6f} | {r['gate']['frequency_advantage_nats']:.6f} | {r['gate']['ordering_advantage_nats']:.6f} | {'Pass' if r['gate']['passed'] else 'Fail'} |" for r in result['decisions'])
    outcome = 'passes' if result['trainability_passed'] else 'fails'
    next_work = ('Declare a bridge to one interacting binary transition rule while keeping the alphabet, grid, context, serializer and training settings fixed. Change the transition rule alone. A later bridge back to the original random-rule task must separately clear readiness.'
                 if result['trainability_passed'] else
                 'Preserve this negative result. Inspect the prediction pipeline/capacity with one declared change at a time, using a fresh frozen diagnostic. Do not extend this budget or relax these margins after inspection.')
    report = f'''# Stage 6: controlled transformer trainability

Completed 20 October 2026: CPU development diagnostic. **The controlled task {outcome}.**
All {len(training)} models completed their full fixed budgets; every initial/final
assessment cell was replayed from the saved checkpoints. Language training stays
disabled regardless of this outcome. This implements the next objective in the
[stage 5 review](../readiness_v5_run1/REVIEW.md).

## What was held fixed and simplified

The unchanged v4 implementation supplies the full six-layer, width-192,
four-head RoPE/RMSNorm/SwiGLU backbone, tied embeddings, 2x2 patch encoder,
first-frame target mask, marker handling, minibatch/update function, Adam and
1000-update warmup/cosine schedule. The backbone has
{summary['backbone_parameters']:,} parameters; the 18-token model has
{summary['model_parameters']:,} parameters. No external pretrained weights are used.

This is the deliberately easier **binary cell-complement rule** on a 4x4 grid
over eight frames: context 47 instead of 1024, vocabulary 18 instead of 10002.
The rule has no spatial interaction or graph topology. Multiple task factors
were simplified for a basic capability check; this does not isolate why v4
random-rule readiness failed. Success can include learning the fixed patch
complement mapping. It does not establish unseen-rule induction, interacting
NCA dynamics, GraphNCA competence, language acceleration or cost recovery.

## Frozen data and endpoint

The [protocol](FROZEN_PROTOCOL.md), [plan](plan.json) and [receipt](plan_receipt.json)
were frozen before training. Plan SHA-256: `{summary['plan_sha256']}`.
Sources, all cached arrays and the three actual starting checkpoints are bound
to that plan. The data seed is {c['data_seed']}, with 512 training, 64 development
and 128 assessment initial conditions. Complement/phase classes are disjoint
across splits. All three model seeds share this one data split; they are model
initialization replicates, not independent data/topology cohorts.

For each seed (131, 149, 167), coherent and prefix-shuffled arms start from the
same full model state and use identical minibatch index order and budgets.
Shuffles preserve the initial frame, markers and supervised target histogram.
The training objective includes subsequent markers; reported readiness uses
patch targets only. Each trajectory contributes 28 scored patches.

Every arm receives exactly 1000 updates / 188,000 input tokens. The endpoint is
the final update, with no checkpoint selection, early stopping or retuning from
development curves. No assessment scores were computed until all six arms
finished. Development curves below are progress diagnostics, not selection data.
The new synthetic assessment is a development capability check; the official
language test and all original v4/v5 evidence remain untouched.

## Final assessment

Both inherited requirements remain **>=0.01 nats per patch**. Higher advantages
are better. All three coherent-trained final models must pass both requirements.
The causal frequency predictor uses training-only prior/concentration fitting.
Its held-out coherent and matched-shuffled joint scores are identical.

| Seed | Coherent NLL | Same checkpoint on shuffled NLL | Frequency advantage | Ordering advantage | Both |
| --- | ---: | ---: | ---: | ---: | --- |
{table}

The causal frequency baseline is **{result['frequency_nll']:.6f}** nats/patch.
Mean coherent-model NLL is **{summary['mean_coherent_nll']:.6f}**; mean patch
accuracy is **{summary['mean_coherent_accuracy']:.2%}**. Frequency passes:
{summary['frequency_passes']}/3; ordering passes: {summary['ordering_passes']}/3;
joint passes: {summary['both_passes']}/3. Full-precision per-initial-condition NLL,
accuracy and target counts are preserved in [results](results.json).

A transition table fitted only on this task's training examples recovers all
16 complement mappings. Its assessment NLL is
{result['table_control']['coherent_nll']:.6f} coherent versus
{result['table_control']['shuffled_nll']:.6f} shuffled, and its two-part gate
{'passes' if result['table_control']['gate']['passed'] else 'fails'}.
The frequency-only negative control correctly fails its zero self/ordering
advantages. This table validates the measurement separately from transformer
learning. No language-transfer significance test is inferred from initial-condition
rows or the three shared-data model seeds.

All 18 cells compare each initial, coherent-trained and shuffle-trained checkpoint
on both data versions, with **3584 patch targets per cell**. Read the
[flat assessment table](assessment_cells.csv), [summary](summary.json),
[replay verification](verification.json) and [copy provenance](PROVENANCE.json).
The descriptive shuffle-trained ordering gaps and differences in gaps are retained
for every seed; they are not additional gate criteria or proof of a mechanism.

As a descriptive control check, {prefix_passes}/3 shuffle-trained checkpoints
also clear the absolute readiness margins when rescored on coherent versus
shuffled assessment data. Their ordering gaps ({min(prefix_gaps):.3f}-{max(prefix_gaps):.3f} nats)
and mean coherent accuracy ({prefix_accuracy:.1%}) compare with the coherent-trained
models' {min(coherent_gaps):.3f}-{max(coherent_gaps):.3f}-nat gaps and
{summary['mean_coherent_accuracy']:.1%} accuracy. This reinforces that the two margins are
basic predictive readiness checks; passing alone does not attribute competence
to training on coherent dynamics. This control inspection does not change the
frozen primary decision or add a post-hoc requirement.

![Native development loss during the fixed CPU budget](learning_curves.png)

Recorded training stage timers total
**{summary['recorded_training_stage_seconds']/60:.3f} minutes** across six CPU models,
including development evaluation and checkpoint writing. These are new local CPU
measurements, not GPU speedups or cloud costs. Total updates/input tokens are
{summary['total_updates']:,}/{summary['total_input_tokens']:,}. Assessment/export
and verification are outside those per-model timers.

## Integrity and next decision

Eight task-specific regression tests cover split-phase exclusion, exact known
transitions/masks, matched control determinism, frozen configuration restrictions,
plan/array tampering, no assessment before completed training and overwrite refusal.
The complete local software suite passes 94 tests.
A retained tiny two-update CPU fixture checks execution only, separately from
this full-backbone scientific diagnostic. Verification reloads three initial and
six final checkpoints, checks final optimizer update counts, source/input/receipt
identities and reproduces all 18 cells and decisions exactly with the same scorer.
It is a separate replay invocation, not an independently implemented loss function
or a rerun of training from scratch. Private weights, optimizer states and arrays
remain preserved outside the publication.

{next_work}

There is no language-training command in this package. No outcome here changes
the failed readiness of the original random-rule checkpoints. Any future language
suite needs a separately frozen configuration that clears readiness, fresh paired
cohorts and an identical language stage. Graph-local serialization/topology and
shorter-PPT cost-recovery experiments remain separate. No remote training, paid
job or official-language-test evaluation was launched during this stage.
'''
    (a.out/'REVIEW.md').write_text(report, encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
