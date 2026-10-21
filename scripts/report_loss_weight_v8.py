"""Export fixed-budget objective evidence, separating imported and new CPU work."""
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
from loss_weight_v8.plan import PARENT, read, verify_plan
from loss_weight_v8.run import verify_training


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'runs/loss_weight_v8_run1')
    p.add_argument('--out', type=Path, default=ROOT/'docs/results/loss_weight_v8_run1')
    a = p.parse_args()
    plan = verify_plan(a.run)
    models, c = verify_training(a.run, plan), plan['training_config']
    result, verification = read(a.run/'results.json'), read(a.run/'verification.json')
    checksum = sha_file(a.run/'results.json')
    if checksum != read(a.run/'results_receipt.json')['results_sha256'] or checksum != verification['results_sha256'] or not verification['passed']:
        raise ValueError('Verified result identity differs')
    if plan['settings']['purpose'] != 'development_loss_weighting' or len(result['full_cells']) != 30 or len(result['first_transition_cells']) != 15 or result['language_allowed']:
        raise ValueError('Complete declared development run required')
    a.out.mkdir(parents=True, exist_ok=False)
    copies = ['plan.json', 'plan_receipt.json', 'FROZEN_PROTOCOL.md', 'training_complete.json', 'training_receipt.json',
              'results.json', 'results_receipt.json', 'verification.json', 'status.json']
    copies += [f"weighted_{r['arm']}_{r['seed']}/{name}" for r in models for name in ('metrics.jsonl', 'summary.json')]
    provenance = {'scope': result['scope'], 'files': {}}
    for name in copies:
        dest = a.out/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.run/name, dest)
        provenance['files'][name] = {'sha256': sha_file(dest), 'origin': 'Frozen objective ablation: '+name}
    dump(a.out/'PROVENANCE.json', provenance)
    full = []
    for row in result['full_cells']:
        values = row['per_initial_condition']
        n = sum(v['patch_targets'] for v in values)
        full.append({k: row[k] for k in ('seed', 'condition', 'arm', 'data_arm', 'patch_nll')} |
                    {'patch_targets': n, 'patch_accuracy': sum(v['patch_accuracy']*v['patch_targets'] for v in values)/n})
    first = [{k: v for k, v in row.items() if '_per_query_' not in k} for row in result['first_transition_cells']]
    write_csv(a.out/'full_cells.csv', full)
    write_csv(a.out/'first_transition_cells.csv', first)
    groups = []
    for condition, arm in [('initial', None)]+[(k, ar) for k in ('standard', 'weighted') for ar in ('coherent', 'prefix')]:
        rows = [r for r in first if (r['condition'], r['arm']) == (condition, arm)]
        trajectories = [r for r in full if (r['condition'], r['arm'], r['data_arm']) == (condition, arm, 'coherent')]
        checks = [decision(next(r['patch_nll'] for r in full if (r['seed'], r['condition'], r['arm'], r['data_arm']) == (s, condition, arm, 'coherent')),
                           next(r['patch_nll'] for r in full if (r['seed'], r['condition'], r['arm'], r['data_arm']) == (s, condition, arm, 'prefix')),
                           result['frequency_nll'], c) for s in c['seeds']]
        groups.append({'condition': condition, 'arm': arm, 'cohorts': len(rows),
                       **{'mean_'+key: statistics.mean(r[key] for r in rows) for key in
                          ('native_nll', 'native_accuracy', 'changed_nll', 'changed_accuracy', 'paired_both_correct', 'argmax_changes_as_required')},
                       'mean_full_patch_nll': statistics.mean(r['patch_nll'] for r in trajectories),
                       'mean_full_patch_accuracy': statistics.mean(r['patch_accuracy'] for r in trajectories),
                       'readiness_passes': sum(r['passed'] for r in checks)})
    summary = {'plan_sha256': sha_file(a.run/'plan.json'), 'all_seed_primary_improvement': result['all_seed_primary_improvement'],
               'language_allowed': False, 'groups': groups, 'paired_comparisons': result['paired_comparisons'],
               'original_queries': result['original_queries'], 'eligible_pairs': result['eligible_pairs'],
               'new_models': len(models), 'imported_standard_models': len(plan['standard_controls']),
               'new_updates': sum(r['steps'] for r in models), 'new_input_tokens': sum(r['input_tokens'] for r in models),
               'new_stage_seconds': sum(r['stage_seconds'] for r in models),
               'historical_control_stage_seconds': sum(r['stage_seconds'] for r in plan['standard_controls']),
               'full_replayed_cases': verification['full_cases'], 'first_transition_replayed_cases': verification['first_transition_cases']}
    dump(a.out/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    labels = ['Standard\ncoherent', 'Weighted\ncoherent', 'Standard\nshuffled', 'Weighted\nshuffled']
    ordered = [next(r for r in groups if (r['condition'], r['arm']) == pair) for pair in
               [('standard', 'coherent'), ('weighted', 'coherent'), ('standard', 'prefix'), ('weighted', 'prefix')]]
    for ax, key, title in zip(axes, ('native_accuracy', 'changed_accuracy', 'paired_both_correct'),
                              ('Original first patch', 'Neighbor intervention', 'Both correct in pair')):
        ax.bar(range(4), [100*r['mean_'+key] for r in ordered], color=['#64748b', '#2563eb', '#a8a29e', '#d97706'])
        for index, group in enumerate(ordered):
            rows = [r for r in first if (r['condition'], r['arm']) == (group['condition'], group['arm'])]
            ax.scatter([index-.12, index, index+.12], [100*r[key] for r in rows], color='#111827', s=16, zorder=3)
        ax.set(title=title, ylabel='Correct predictions (%)', ylim=(0, 100), xticks=range(4), xticklabels=labels)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Fixed 1000 updates: first-transition loss weighting (CPU development)')
    for extension in ('png', 'svg'):
        fig.savefig(a.out/f'first_transition_accuracy.{extension}', dpi=180)
    plt.close(fig)
    table = '\n'.join(f"| {r['seed']} | {r['first_patch_nll_improvement']:.6f} | {r['intervention_nll_improvement']:.6f} | {'Pass' if r['weighted_readiness']['passed'] else 'Fail'} | {'Pass' if r['first_patch_improved'] and r['intervention_improved'] else 'Fail'} |" for r in result['paired_comparisons'])
    control = '\n'.join(f"| {r['condition']} | {r['arm'] or 'initial'} | {r['mean_native_nll']:.6f} | {r['mean_native_accuracy']:.2%} | {r['mean_changed_nll']:.6f} | {r['mean_changed_accuracy']:.2%} | {r['mean_paired_both_correct']:.2%} | {r['mean_full_patch_nll']:.6f} | {r['mean_full_patch_accuracy']:.2%} | {r['readiness_passes']}/3 |" for r in groups)
    gains = '\n'.join(f"| {r['seed']} | {r['shuffled_first_patch_improvement']:.6f} | {r['difference_in_objective_gains']:.6f} |" for r in result['paired_comparisons'])
    standard = next(r for r in groups if (r['condition'], r['arm']) == ('standard', 'coherent'))
    weighted = next(r for r in groups if (r['condition'], r['arm']) == ('weighted', 'coherent'))
    next_step = ('The declared objective comparison passes on all three initialization seeds. Before broadening the rule family, freeze a fresh joint-phase-disjoint confirmation of this exact treatment and retain both shuffled controls. Inspect the remaining original/intervention errors; no perfect-execution threshold was part of this study.'
                 if result['all_seed_primary_improvement'] else
                 'The declared all-seed objective comparison fails. Every coherent seed has worse original AND intervention NLL; the trajectory readiness check alone misses this regression. Preserve the sixfold factor and negative endpoint. The next development diagnostic should compare first-transition train/development/assessment errors using these existing checkpoints, before choosing another treatment. This would separate a failure to fit the weighted prediction from a held-out generalization gap; it would not identify a unique cause. That additional diagnostic has not been executed here. Do not extend training, select a checkpoint, relax the margins or discard a seed in this study.')
    report = f'''# Stage 8: first-transition loss-weighting ablation

Completed 20 October 2026 on local CPU. Frozen all-seed primary improvement:
**{result['all_seed_primary_improvement']}**. Language training remains disabled.
Six new weighted models completed 1000 updates each, compared with six preserved,
verified standard-loss spatial controls from [stage 7](../interaction_v7_run1/REVIEW.md).
The [protocol](FROZEN_PROTOCOL.md), [plan](plan.json) and [receipt](plan_receipt.json)
were frozen before new scientific fitting. Plan SHA-256:
`{summary['plan_sha256']}`.

## What changed and what remained matched

Only the supervised training objective changes: all six targets of the first
transition have weight 6; the remaining 36 valid targets have weight 1; masked
initial-context targets have weight 0. Divide summed weighted cross-entropy by
the weight sum (72 per sequence). The first transition contributes 50% of the
objective instead of 1/7. Patch/marker total weight remains 2/3 versus 1/3.
This weights the entire first transition, not solely the queried first patch.
Evaluation always uses the original unweighted predictive likelihood.

The rule remains x' = 1 XOR x XOR left XOR right on a periodic 4x4 binary grid.
The same cached train/development/assessment arrays and query mask are copied
byte-for-byte. Every seed reuses its actual initial weights and minibatch order.
Six layers, width 192, four heads, 47-token context, 2x2 patches, eight frames,
512/64/128 split, Adam, LR schedule, accumulation, 1000 updates, 188,000 input
tokens and 168,000 supervised targets per model remain matched. No additional
examples or optimizer updates are introduced. Weighting adds pointwise arithmetic;
this is equal update/token exposure, not an assertion of identical FLOPs or time.

The standard controls are imported historical CPU runs, not freshly retrained
controls. Their complete saved parameters, input/source identities, optimizer
counts, initial states and original assessments are verified and replayed.
Their metrics and hashes remain in the stage 7 evidence; the six selected
identities are additionally bound in this [plan](plan.json). New weighted models
are trained from initialization, never continued from a standard endpoint.
Coherent and matched prefix-shuffled training arms are retained for all seeds.

## Frozen decision and full results

For EACH coherent seed, require standard NLL minus weighted NLL >=0.01 nats on
both the 128 original first-patch queries and 122 eligible neighbor interventions.
Also retain both inherited full-trajectory frequency and ordering readiness
advantages >=0.01 nats. All three must pass; final update 1000 is the only endpoint.
The first query conditions on the initial frame and next start marker, before
any repeated frame or next-frame patch. The intervention flips initial cell (0,3)
outside the target patch; the correct spatial output flips bit zero. Six changed
joint phases overlapping training/development remain excluded under the original
mask; no original query or seed is dropped. All data are inspected development
data. These seeds share one split/rule and are initialization replicates.

| Seed | Original NLL improvement | Intervention NLL improvement | Weighted trajectory readiness | Both first-patch improvements |
| --- | ---: | ---: | --- | --- |
{table}

Positive improvements favor weighting. The primary concerns predictive NLL;
accuracy and paired correctness have no newly invented admission threshold.

| Objective | Training data | Original NLL | Original accuracy | Changed NLL | Changed accuracy | Both correct | Full NLL | Full accuracy | Trajectory readiness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{control}

Means span three seeds. Coherent first-patch accuracy changes from
**{standard['mean_native_accuracy']:.2%} to {weighted['mean_native_accuracy']:.2%}**;
intervention accuracy from **{standard['mean_changed_accuracy']:.2%} to {weighted['mean_changed_accuracy']:.2%}**;
paired-both correctness from **{standard['mean_paired_both_correct']:.2%} to {weighted['mean_paired_both_correct']:.2%}**.
Full-trajectory accuracy changes from {standard['mean_full_patch_accuracy']:.2%}
to {weighted['mean_full_patch_accuracy']:.2%}; preserve this whole-trajectory change alongside the
first-transition result. Initial-model scores, all control outcomes and target
counts remain in [full results](results.json), the [30-cell CSV](full_cells.csv)
and [15-case first-transition CSV](first_transition_cells.csv).

![Original, intervened and paired correctness](first_transition_accuracy.png)

Bars show means; dots show the three initialization seeds.

## Shuffled controls and interpretation

| Seed | Shuffled-model original NLL improvement | Coherent gain minus shuffled gain |
| --- | ---: | ---: |
{gains}

These differences are descriptive. They are not an added primary criterion,
confidence interval or causal proof of a unique learned mechanism. A model
can improve likelihood without complete rule execution. Matching this single
neighbor perturbation does not establish every spatial dependency, unseen-rule
generalization or language transfer. Period-two recall can still help later
trajectory predictions; first-transition scoring limits that particular shortcut.

## Verification, costs and next decision

All 30 full-trajectory cases and 15 original/intervention checkpoint cases replay
exactly in a separate invocation after reloading saved checkpoints. Each full
case scores 128 initial conditions x 28 patch targets. Standard scores agree
exactly with their original stage 7 assessments. Replay uses the same frozen
scorers; it is not independent loss code or retraining from scratch.
[Verification](verification.json), [summary](summary.json) and the 21-copy
[provenance manifest](PROVENANCE.json) retain the compact evidence. Raw arrays,
weights and optimizer states remain private and preserved.

Eight new objective/matching tests and a retained two-update execution fixture
are separate from scientific evidence. The complete local software suite passes
112 tests. The fixture uses shorter weighted training against historical controls
and supports execution only. It is not included in the scientific comparison.
The factor-one regression reproduces the original training update exactly,
including all parameter values and sampled index order.

New weighted training: {summary['new_updates']:,} updates,
{summary['new_input_tokens']:,} input tokens; recorded stage timers sum to
{summary['new_stage_seconds']/60:.3f} CPU minutes. Imported control stage timers
sum to {summary['historical_control_stage_seconds']/60:.3f} historical CPU minutes.
The comparison accounts for twelve model budgets (12,000 updates / 2,256,000
input tokens), with six newly fitted and six reused. Assessment, replay, tests
and export add work. Concurrent local checks and different invocation conditions
prevent interpreting these stage timers as a controlled throughput comparison.
No GPU acceleration, reduced-token convergence or cost recovery is measured here.

{next_step}

The next bridge to interacting categorical/random rules must separately clear
synthetic readiness. Original random-rule checkpoints retain failed stage 5
readiness. Any language-transfer study needs a separately frozen configuration,
fresh paired cohorts and an identical full language stage. This study changes no
topology, serializer or budget. No remote/paid training, official language test,
language stage or publication was launched.
'''
    (a.out/'REVIEW.md').write_text(report, encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
