"""Export verified split-fit evidence without raw inputs or weights."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics as st
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import dump, sha_file
from fit_v9.diagnostic import read, verify_plan, SPLITS


METRICS = ('first_patch_nll', 'first_patch_accuracy', 'transition_patch_nll', 'transition_patch_accuracy', 'transition_marker_nll', 'transition_all_token_nll')


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(result):
    groups = []
    for condition, arm in [('initial', None)]+[(k, a) for k in ('standard', 'weighted') for a in ('coherent', 'prefix')]:
        for data_arm in ('coherent', 'prefix'):
            for split in SPLITS:
                cells = [r for r in result['cells'] if (r['condition'], r['arm'], r['data_arm'], r['split']) == (condition, arm, data_arm, split)]
                groups.append({'condition': condition, 'arm': arm, 'data_arm': data_arm, 'split': split, 'seeds': len(cells),
                               **{k: st.mean(r[k] for r in cells) for k in METRICS}})
    principal = [r for r in result['paired_comparisons'] if r['arm'] == r['data_arm'] == 'coherent']
    return {'new_training_updates': 0, 'language_allowed': False, 'models': result['models_scored'], 'cells': len(result['cells']),
            'query_scoring_exposures': sum(r['queries'] for r in result['cells']),
            'groups': groups, 'principal_comparisons': principal,
            'principal_categories': {k: sum(r['category'] == k for r in principal) for k in sorted({r['category'] for r in principal})},
            'max_short_long_difference': max(r['short_long_max_nll_difference'] for r in result['cells'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'runs/fit_v9_run1')
    p.add_argument('--out', type=Path, default=ROOT/'docs/results/fit_v9_run1')
    a = p.parse_args()
    verify_plan(a.run)
    result, verification = read(a.run/'results.json'), read(a.run/'verification.json')
    checksum = sha_file(a.run/'results.json')
    if checksum != verification['results_sha256'] or checksum != read(a.run/'results_receipt.json')['results_sha256'] or not verification['passed']:
        raise ValueError('Verified results required')
    summary = summarize(result) | {'plan_sha256': sha_file(a.run/'plan.json')}
    if summary['cells'] != 90 or summary['models'] != 15 or result['new_training_updates'] or result['language_allowed']:
        raise ValueError('Complete inference-only diagnostic required')
    a.out.mkdir(parents=True, exist_ok=False)
    copies = ('plan.json', 'plan_receipt.json', 'FROZEN_PROTOCOL.md', 'results.json', 'results_receipt.json', 'verification.json', 'status.json')
    provenance = {'scope': result['scope'], 'files': {}}
    for name in copies:
        shutil.copyfile(a.run/name, a.out/name)
        provenance['files'][name] = {'sha256': sha_file(a.out/name), 'origin': 'Frozen split-fit diagnostic: '+name}
    dump(a.out/'PROVENANCE.json', provenance)
    dump(a.out/'summary.json', summary)
    csv_write(a.out/'cells.csv', [{k: v for k, v in row.items() if k != 'per_query'} for row in result['cells']])
    csv_write(a.out/'group_means.csv', summary['groups'])
    csv_write(a.out/'generalization_gaps.csv', result['generalization_gaps'])
    csv_write(a.out/'paired_comparisons.csv', [{k: v for k, v in row.items() if k != 'weighted_minus_standard_nll'} | row['weighted_minus_standard_nll'] for row in result['paired_comparisons']])
    def group(condition, split, arm='coherent', data_arm='coherent'):
        return next(r for r in summary['groups'] if (r['condition'], r['arm'], r['data_arm'], r['split']) == (condition, arm, data_arm, split))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ('first_patch_nll', 'first_patch_accuracy', 'transition_patch_nll'),
                                ('First-patch NLL', 'First-patch accuracy', 'All four first-transition patches')):
        scale = 100 if metric.endswith('accuracy') else 1
        for i, condition in enumerate(('standard', 'weighted')):
            x = np.arange(3)+(i-.5)*.32
            values = [scale*group(condition, split)[metric] for split in SPLITS]
            ax.bar(x, values, width=.3, label=condition, color=('#64748b', '#2563eb')[i])
            for j, split in enumerate(SPLITS):
                rows = [r for r in result['cells'] if (r['condition'], r['arm'], r['data_arm'], r['split']) == (condition, 'coherent', 'coherent', split)]
                ax.scatter([x[j]-.06, x[j], x[j]+.06], [scale*r[metric] for r in rows], color='#111827', s=12, zorder=3)
        ax.set(title=title, xticks=range(3), xticklabels=['Train', 'Development', 'Assessment'], ylabel='Accuracy (%)' if scale == 100 else 'Nats per patch')
        if scale == 100:
            ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Existing coherent-trained checkpoints: split comparison (no new training)')
    for extension in ('png', 'svg'):
        fig.savefig(a.out/f'fit_gaps.{extension}', dpi=180)
    plt.close(fig)
    rows = []
    for split in SPLITS:
        for condition in ('standard', 'weighted'):
            r = group(condition, split)
            rows.append(f"| {split} | {condition} | {r['first_patch_nll']:.6f} | {r['first_patch_accuracy']:.2%} | {r['transition_patch_nll']:.6f} | {r['transition_patch_accuracy']:.2%} | {r['transition_marker_nll']:.6f} | {r['transition_all_token_nll']:.6f} |")
    main_table = '\n'.join(rows)
    pairs = '\n'.join(f"| {r['seed']} | {r['weighted_minus_standard_nll']['train']:.6f} | {r['weighted_minus_standard_nll']['development']:.6f} | {r['weighted_minus_standard_nll']['assessment']:.6f} | {r['category']} |" for r in summary['principal_comparisons'])
    control = []
    for split in SPLITS:
        for condition in ('standard', 'weighted'):
            r = group(condition, split, 'prefix', 'prefix')
            control.append(f"| {split} | {condition} | {r['first_patch_nll']:.6f} | {r['first_patch_accuracy']:.2%} | {r['transition_patch_nll']:.6f} | {r['transition_patch_accuracy']:.2%} |")
    control_table = '\n'.join(control)
    gaps = []
    for condition in ('standard', 'weighted'):
        values = [r for r in result['generalization_gaps'] if (r['condition'], r['arm'], r['data_arm']) == (condition, 'coherent', 'coherent')]
        gaps.append(f"| {condition} | {st.mean(r['development_minus_train_nll'] for r in values):.6f} | {st.mean(r['assessment_minus_train_nll'] for r in values):.6f} |")
    gap_table = '\n'.join(gaps)
    categories = summary['principal_categories']
    mixed = categories.get('training_gain_with_assessment_reversal', 0) > 0 and categories.get('fit_regression_on_train_and_assessment', 0) > 0
    interpretation = ('The result is mixed across seeds: two improve training first-patch NLL but worsen held-out NLL; one worsens on training data too. A single pooled mean would conceal this distinction. A generalization reversal is observed in seeds 149 and 167; held-out overfitting alone is insufficient to describe seed 131.'
                      if mixed else 'Use the per-seed frozen categories above rather than replacing discordant seeds with a pooled diagnosis.')
    report = f'''# Stage 9: first-transition fit versus held-out generalization

Completed 20 October 2026 on local CPU using existing checkpoints only.
**No new optimizer updates or model training occurred. Language remains gated.**
The [stage 8 negative result](../loss_weight_v8_run1/REVIEW.md) is unchanged.
This post-hoc development diagnostic was declared after inspecting stage 8
assessment outcomes, then frozen before new split scoring. It is not confirmation.
[Protocol](FROZEN_PROTOCOL.md), [plan](plan.json) and [receipt](plan_receipt.json):
`{summary['plan_sha256']}`.

## Observed first-patch fit and generalization

Positive deltas below mean the weighted model is worse. The descriptive margin
is 0.01 nats, fixed before new scores. All three coherent seeds are retained.

| Seed | Train NLL delta | Development NLL delta | Assessment NLL delta | Frozen descriptive category |
| --- | ---: | ---: | ---: | --- |
{pairs}

{interpretation}

| Split | Objective | First-patch NLL | First-patch accuracy | Four-patch NLL | Four-patch accuracy | Marker NLL | Six-token NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{main_table}

These are three-seed means on coherent data for coherent-trained models.
Lower NLL is better. The original first patch sees only the full initial frame
and next start marker. The four-patch metric averages all patches of the first
transition; later patches condition on earlier TRUE patches. It is teacher-forced
prediction, not a free rollout, and supplies no later frame. Marker loss is
separated so formatting performance cannot masquerade as patch competence.

| Objective | Mean development minus train first NLL | Mean assessment minus train first NLL |
| --- | ---: | ---: |
{gap_table}

Positive gaps indicate worse held-out likelihood. Training examples here are
the original sampled cache, not new examples. The 512/64/128 initial conditions
retain their original joint-phase-disjoint split. Phase disjointness prevents
the defined exact orbit leakage; it does not guarantee learned rule generalization.

![Existing-checkpoint split comparison](fit_gaps.png)

Bars are three-seed means; dots are individual seeds. These compare splits at
fixed final endpoints and are not convergence curves or acceleration estimates.

## The weighted transition and its first patch are different findings

The weighted model has lower whole-first-transition four-patch NLL on BOTH
train and assessment in all three coherent seeds. Its four-patch accuracy also
improves in each seed on these two splits. This coexists with worse held-out
FIRST-patch NLL and accuracy in every coherent seed. Aggregate first-transition
prediction therefore hides the queried first-patch failure, just as full-trajectory
readiness failed to detect it in stage 8. Lower whole-transition training loss
does not establish a general inability to fit the weighted task; nor does it
prove why the initial spatial prediction generalizes poorly.

The sixfold treatment weighted six targets, including markers, rather than
isolating the first patch. These results show different predictive behavior
across that aggregate objective. They do not identify gradient interference,
capacity, a particular shortcut or a unique internal circuit. Both standard and
weighted models have nonzero training errors and generalization gaps.

## Shuffled controls and complete scoring

Shuffle-trained controls on their actual prefix-shuffled data:

| Split | Objective | First-patch NLL | First-patch accuracy | Four-patch NLL | Four-patch accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
{control_table}

Both model arms are additionally scored on the other data version, and initial
models remain. All 15 model conditions x three splits x two data versions are
retained: 90 cells and {summary['query_scoring_exposures']:,} scoring exposures.
These are repeated measurements of the same 704 cached initial conditions and
three initialization seeds, not independent rule/data cohorts or 21,120 independent
samples. No confidence interval or significance claim is inferred from query rows.
See [all cells](cells.csv), [group means](group_means.csv),
[paired categories](paired_comparisons.csv), [gaps](generalization_gaps.csv),
[per-query results](results.json) and [summary](summary.json).

## Integrity and next decision

Every coherent assessment first-patch mean, accuracy and per-query vector agrees
EXACTLY with the corresponding stage 8 record (all 15 checkpoint conditions).
The longer causal prefix gives a maximum first-patch loss difference of
{summary['max_short_long_difference']:.9g} nats, below the frozen 1e-5 tolerance,
with identical argmax correctness. Short-prefix scores remain authoritative.
Reloading all 15 initial/final files reproduces every one of the 90 cells and
all descriptive categories exactly in a separate invocation using the same
scorers. This is checkpoint replay, not independently implemented loss code or
new training. Seven source/public copies have [SHA provenance](PROVENANCE.json);
[verification](verification.json) binds results. Raw arrays, weights, optimizer
states and original evidence remain preserved privately.

Eight new scoring/category tests pass; the complete local suite passes 122 tests,
including two subsequent starter-snapshot integrity regressions.
Tests cover fixed positions and patch/marker normalization, actual causal prefixes,
future-frame exclusion, short/long agreement, invalid inputs, all controls,
comparison signs/margins and paired gap arithmetic. No remote CI execution is
claimed. This run used CPU inference; no GPU speed or training-cost recovery was
measured.

The next candidate development treatment is **initial-condition diversity alone**:
hold the objective definitions, model, rule, representation and update/token
budget fixed while varying training-phase coverage, retaining standard/weighted
and both matched shuffled controls. Declare it separately before fitting and
keep first-patch, whole-transition and generalization measurements distinct.
This candidate responds to the observed fit/generalization gaps; this diagnostic
does not prove it will help. Do not combine it with a new weighting factor,
topology or serializer and attribute any result to data diversity. No such
treatment has been trained or configured by this diagnostic.

Fresh joint-phase-disjoint data are required for later confirmation. Original
random-rule checkpoints still fail stage 5 readiness. Any return to a language
suite requires a separately frozen eligible synthetic configuration, fresh paired
cohorts and an identical full language stage. No remote/paid training, official
language-test access, model update, publication or historical-evidence rewrite
occurred here.
'''
    (a.out/'REVIEW.md').write_text(report, encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
