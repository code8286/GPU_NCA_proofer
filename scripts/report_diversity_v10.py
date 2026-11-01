"""Export verified nested-diversity evidence and preserve every objective/control."""
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
from readiness_v5.gate import decision
from diversity_v10.plan import read, verify_plan, OBJECTIVES, ARMS
from diversity_v10.run import verify_training


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_evidence(run, result):
    raw = {k: [] for k in ('full_cells', 'fit_cells', 'intervention_cells')}
    for name, checksum in result['evidence_sha256'].items():
        if sha_file(run/name) != checksum:
            raise ValueError('Raw evidence changed')
        key = name.split('/')[-1].rsplit('_', 1)[0]
        raw[key].extend(read(run/name))
    return raw


def summarize(plan, result, models, raw):
    full = []
    for row in raw['full_cells']:
        values = row['per_initial_condition']
        n = sum(v['patch_targets'] for v in values)
        full.append({k: v for k, v in row.items() if k != 'per_initial_condition'} |
                    {'patch_targets': n, 'patch_accuracy': sum(v['patch_accuracy']*v['patch_targets'] for v in values)/n})
    groups = []
    for level, objective, arm in [(0, None, None)]+[(n, o, a) for n in (512, 4096) for o in OBJECTIVES for a in ARMS]:
        rows = [r for r in result['intervention_cells'] if (r['level'], r['objective'], r['arm']) == (level, objective, arm)]
        trajectories = [r for r in full if (r['level'], r['objective'], r['arm'], r['data_arm']) == (level, objective, arm, 'coherent')]
        gates = []
        for seed in plan['training_config']['seeds']:
            scores = {r['data_arm']: r['patch_nll'] for r in full if (r['seed'], r['level'], r['objective'], r['arm']) == (seed, level, objective, arm)}
            frequency = next(r['frequency_nll'] for r in result['baselines'] if r['level'] == (level or 512))
            gates.append(decision(scores['coherent'], scores['prefix'], frequency, plan['training_config']))
        groups.append({'level': level, 'objective': objective, 'arm': arm,
                       **{'mean_'+k: st.mean(r[k] for r in rows) for k in ('native_nll', 'native_accuracy', 'changed_nll', 'changed_accuracy', 'paired_both_correct')},
                       'mean_full_patch_nll': st.mean(r['patch_nll'] for r in trajectories),
                       'mean_full_patch_accuracy': st.mean(r['patch_accuracy'] for r in trajectories),
                       'readiness_passes': sum(g['passed'] for g in gates)})
    fits = []
    for level in (512, 4096):
        for objective in OBJECTIVES:
            for probe in ('anchor', 'extension', 'development', 'assessment'):
                rows = [r for r in result['fit_cells'] if (r['level'], r['objective'], r['arm'], r['probe']) == (level, objective, 'coherent', probe)]
                exposed = [r['drawn_subset']['nll'] for r in rows if r['drawn_subset']['count']]
                fits.append({'level': level, 'objective': objective, 'probe': probe,
                             'mean_first_patch_nll': st.mean(r['first_patch_nll'] for r in rows),
                             'mean_first_patch_accuracy': st.mean(r['first_patch_accuracy'] for r in rows),
                             'mean_transition_patch_nll': st.mean(r['transition_patch_nll'] for r in rows),
                             'mean_transition_patch_accuracy': st.mean(r['transition_patch_accuracy'] for r in rows),
                             'mean_drawn_queries': st.mean(r['drawn_subset']['count'] for r in rows),
                             'mean_drawn_nll_nonempty_seeds': st.mean(exposed) if exposed else None})
    summary = {'plan_sha256': result['plan_sha256'], 'objective_strata_passed': result['objective_strata_passed'],
               'study_primary_passed': result['study_primary_passed'], 'language_allowed': False,
               'new_models': len(models), 'imported_controls': len(plan['standard_controls']),
               'new_updates': sum(r['steps'] for r in models), 'new_input_tokens': sum(r['input_tokens'] for r in models),
               'new_stage_seconds': sum(r['stage_seconds'] for r in models),
               'historical_stage_seconds': sum(r['source_summary']['stage_seconds'] for r in plan['standard_controls']),
               'groups': groups, 'coherent_fit_probes': fits, 'paired_comparisons': result['paired_comparisons'],
               'unique_exposures': {k: v['unique_rows'] for k, v in plan['expected_orders'].items()},
               'replayed_full_cells': len(result['full_cells']), 'replayed_fit_cells': len(result['fit_cells']),
               'replayed_intervention_cells': len(result['intervention_cells'])}
    return summary, full


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'runs/diversity_v10_run1')
    p.add_argument('--out', type=Path, default=ROOT/'docs/results/diversity_v10_run1')
    a = p.parse_args()
    plan = verify_plan(a.run)
    models = verify_training(a.run, plan)
    result, verification = read(a.run/'results.json'), read(a.run/'verification.json')
    checksum = sha_file(a.run/'results.json')
    if checksum != verification['results_sha256'] or checksum != read(a.run/'results_receipt.json')['results_sha256'] or not verification['passed']:
        raise ValueError('Complete verified results required')
    raw = raw_evidence(a.run, result)
    summary, full = summarize(plan, result, models, raw)
    a.out.mkdir(parents=True, exist_ok=False)
    copies = ['plan.json', 'plan_receipt.json', 'FROZEN_PROTOCOL.md', 'training_complete.json', 'training_receipt.json',
              'results.json', 'results_receipt.json', 'verification.json', 'status.json']
    copies += list(result['evidence_sha256'])
    copies += [f"{r['objective']}_{r['arm']}_{r['seed']}/{name}" for r in models for name in ('metrics.jsonl', 'summary.json')]
    provenance = {'scope': result['scope'], 'files': {}}
    for name in copies:
        target = a.out/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.run/name, target)
        provenance['files'][name] = {'sha256': sha_file(target), 'origin': 'Frozen nested-diversity study: '+name}
    dump(a.out/'PROVENANCE.json', provenance)
    dump(a.out/'summary.json', summary)
    csv_write(a.out/'full_cells.csv', full)
    fit_rows = []
    for row in result['fit_cells']:
        fit_rows.append({k: v for k, v in row.items() if k not in ('drawn_subset', 'undrawn_subset')} |
                        {'drawn_'+k: v for k, v in row['drawn_subset'].items()} | {'undrawn_'+k: v for k, v in row['undrawn_subset'].items()})
    csv_write(a.out/'fit_cells.csv', fit_rows)
    csv_write(a.out/'intervention_cells.csv', result['intervention_cells'])
    csv_write(a.out/'paired_comparisons.csv', [{k: v for k, v in r.items() if k != 'readiness'} | r['readiness'] for r in result['paired_comparisons']])
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    def group(level, objective, arm='coherent'):
        return next(r for r in summary['groups'] if (r['level'], r['objective'], r['arm']) == (level, objective, arm))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for col, objective in enumerate(OBJECTIVES):
        for j, level in enumerate((512, 4096)):
            x = np.arange(2)+(j-.5)*.32
            g = group(level, objective)
            axes[0, col].bar(x, [g['mean_native_nll'], g['mean_changed_nll']], width=.3, color=('#64748b', '#2563eb')[j], label=f'{level} available')
            for i, key in enumerate(('native_nll', 'changed_nll')):
                rows = [r for r in result['intervention_cells'] if (r['level'], r['objective'], r['arm']) == (level, objective, 'coherent')]
                axes[0, col].scatter([x[i]-.06, x[i], x[i]+.06], [r[key] for r in rows], color='#111827', s=14, zorder=3)
            x = np.arange(3)+(j-.5)*.32
            axes[1, col].bar(x, [100*g[k] for k in ('mean_native_accuracy', 'mean_changed_accuracy', 'mean_full_patch_accuracy')],
                             width=.3, color=('#64748b', '#2563eb')[j], label=f'{level} available')
        axes[0, col].set(title=objective+' objective', ylabel='First-patch NLL (lower is better)', xticks=range(2), xticklabels=['Original', 'Neighbor intervention'])
        axes[0, col].set_ylim(0, axes[0, col].get_ylim()[1]*1.18)
        axes[1, col].set(ylabel='Accuracy (%)', ylim=(0, 100), xticks=range(3), xticklabels=['First patch', 'Intervention', 'Full trajectory'])
        for ax in axes[:, col]:
            ax.grid(axis='y', alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Nested initial-condition diversity: fixed 1000-update CPU endpoints')
    for extension in ('png', 'svg'):
        fig.savefig(a.out/f'diversity_comparison.{extension}', dpi=180)
    plt.close(fig)
    primary = '\n'.join(f"| {r['objective']} | {r['seed']} | {r['first_patch_improvement_nats']:.6f} | {r['intervention_improvement_nats']:.6f} | {'Pass' if r['readiness']['passed'] else 'Fail'} | {'Pass' if r['passed'] else 'Fail'} |" for r in result['paired_comparisons'])
    def group_table(arm):
        return '\n'.join(f"| {r['objective']} | {r['level']} | {r['mean_native_nll']:.6f} | {r['mean_native_accuracy']:.2%} | {r['mean_changed_nll']:.6f} | {r['mean_changed_accuracy']:.2%} | {r['mean_paired_both_correct']:.2%} | {r['mean_full_patch_nll']:.6f} | {r['mean_full_patch_accuracy']:.2%} | {r['readiness_passes']}/3 |" for r in summary['groups'] if r['arm'] == arm)
    coherent_table, shuffled_table = group_table('coherent'), group_table('prefix')
    probe_table = '\n'.join(f"| {r['objective']} | {r['level']} | {r['probe']} | {r['mean_first_patch_nll']:.6f} | {r['mean_first_patch_accuracy']:.2%} | {r['mean_transition_patch_nll']:.6f} | {r['mean_transition_patch_accuracy']:.2%} | {r['mean_drawn_queries']:.2f} |" for r in summary['coherent_fit_probes'])
    coverage = '\n'.join(f"| {s} | {plan['expected_orders'][f'512_{s}']['unique_rows']} | {plan['expected_orders'][f'4096_{s}']['unique_rows']} |" for s in plan['training_config']['seeds'])
    controls = '\n'.join(f"| {r['objective']} | {r['seed']} | {r['shuffled_first_patch_improvement_nats']:.6f} | {r['coherent_minus_shuffled_diversity_gain']:.6f} |" for r in result['paired_comparisons'])
    baseline = '\n'.join(f"| {r['level']} | {r['frequency_nll']:.6f} | {r['concentration']} |" for r in result['baselines'])
    if result['study_primary_passed']:
        next_step = 'Both predeclared objective strata pass on all seeds. Freeze an exact-treatment confirmation on fresh joint-phase-disjoint data before broadening the rule family. This does not authorize a language stage or establish an unseen-rule mechanism.'
    elif any(result['objective_strata_passed'].values()):
        next_step = 'Only a subset of objective strata passes; the study-wide primary fails. Preserve this objective-specific outcome, inspect the controls and first-transition probes, and freeze a separate confirmation before claiming a general diversity benefit. Do not drop the failing objective or seed.'
    else:
        next_step = 'Neither objective stratum clears the frozen all-seed comparison. Preserve the negative primary and examine the drawn/undrawn probes, control gains and first-transition versus trajectory differences before declaring another single development change. Do not extend training, change the cache size, alter a weighting factor or relax the gate in this run.'
    report = f'''# Stage 10: initial-condition diversity at fixed token exposure

Completed 21 October 2026 on local CPU. **Study-wide primary:
{result['study_primary_passed']}**. Standard stratum:
**{result['objective_strata_passed']['standard']}**; weighted stratum:
**{result['objective_strata_passed']['weighted']}**. Language remains gated.
Twelve new 4096-condition models are compared with twelve verified imported
512-condition endpoints. No imported control was retrained or relabeled.

[Protocol](FROZEN_PROTOCOL.md), [plan](plan.json) and [receipt](plan_receipt.json)
froze before fitting, with SHA-256:
`{summary['plan_sha256']}`. This is development on previously inspected held-out
data, not fresh confirmation. Earlier [stage 8](../loss_weight_v8_run1/REVIEW.md)
and [stage 9](../fit_v9_run1/REVIEW.md) results remain unchanged.

## The only treatment and actual exposure

Increase available training initial conditions from 512 to 4096. The original
512 coherent trajectories AND prefix shuffles are exact nested rows. New rows
use the first eligible distinct joint-phase representatives from the frozen
permutation. No new row shares an orbit with original train/development/
assessment or any neighbor-flipped assessment state. All 122 eligible neighbor
interventions remain excluded from the larger training set, with the old mask.
The known spatial rule, serializer, alphabet, 4x4 grid, eight frames, context 47,
six-layer/192-wide/four-head model and actual initial weights remain fixed.

Retain standard (factor one) and normalized complete-first-transition sixfold
objectives, each with coherent and matched shuffled training. Adam/reset state,
LR schedule, batch/accumulation and 1000 updates / 188000 input tokens / 168000
supervised targets per model match. All four new arms within each seed use the
same sampled indices. Changing the sampling bound changes actual examples across
cache sizes; only RNG seed and draw count are paired, not identical row identities.

| Seed | Distinct rows drawn from 512 | Distinct rows drawn from 4096 |
| --- | ---: | ---: |
{coverage}

Every model draws 4000 sequences with replacement. Available cache size and
actual observed diversity differ. Increasing cache size changes repetition at
fixed exposure. New coverage is checked against actual training-order digests,
and original 512 digests reproduce historical controls. Do not infer full-cache
fitting from a score on rows that were never drawn.

## Frozen primary comparisons

Require >=0.01 nats improvement in BOTH original first-patch and eligible
intervention NLL for every coherent seed, retaining both unchanged trajectory
readiness margins. Report standard and weighted strata separately; the study-wide
primary requires BOTH strata. These are fixed development criteria, not p-values
or independent rule-cohort evidence. Positive improvements favor the larger cache.

| Objective | Seed | First-patch NLL improvement | Intervention NLL improvement | Larger-cache trajectory readiness | All criteria |
| --- | ---: | ---: | ---: | --- | --- |
{primary}

Coherent-trained model means across three initialization seeds:

| Objective | Cache | First NLL | First accuracy | Intervention NLL | Intervention accuracy | Both correct | Full NLL | Full accuracy | Readiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{coherent_table}

The larger-cache weighted arm's first accuracy is
{group(4096, 'weighted')['mean_native_accuracy']:.2%}, compared with
{group(512, 'standard')['mean_native_accuracy']:.2%} for the original smaller-cache
standard arm; full-trajectory accuracy is
{group(4096, 'weighted')['mean_full_patch_accuracy']:.2%} versus
{group(512, 'standard')['mean_full_patch_accuracy']:.2%}. These cross-objective
comparisons are descriptive, not the primary test. Improving the failed narrow-
cache weighted branch does not establish a generally better synthetic objective
or improved language pre-pre-training.

![Fixed-budget diversity comparison](diversity_comparison.png)

Bars are three-seed means; dots in the NLL panels show individual seeds. These
are final-endpoint comparisons, not convergence curves or time-to-target speedups.
First-patch scoring sees the full initial frame and next start marker only.
The intervention flips a known adjacent cell outside the target patch. Paired
correctness requires both predictions correct; likelihood gain is not perfect
rule execution. Full-trajectory accuracy can conceal first-transition failures.

## Controls and calibrated readiness

Shuffle-trained models, scored on coherent assessment trajectories:

| Objective | Cache | First NLL | First accuracy | Intervention NLL | Intervention accuracy | Both correct | Full NLL | Full accuracy | Readiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{shuffled_table}

Larger-cache shuffle-trained controls pass the trajectory readiness margins in
{group(4096, 'standard', 'prefix')['readiness_passes']}/3 standard and
{group(4096, 'weighted', 'prefix')['readiness_passes']}/3 weighted seeds despite weak
first-patch accuracy. Readiness alone remains insufficient to establish reliable
spatial-rule prediction or a dynamics-specific mechanism.

| Objective | Seed | Shuffled-model first NLL diversity gain | Coherent gain minus shuffled gain |
| --- | ---: | ---: | ---: |
{controls}

These control gains are descriptive, not new primary gates or causal mechanism
proofs. Each cache has a train-only frequency predictor with the unchanged
calibration policy: its prior uses all training rows and its concentration is
selected on the first 256. This deterministic recalibration accounts for the
changed training distribution; no model hyperparameter is retuned.

| Training cache | Assessment causal frequency NLL | Concentration |
| ---: | ---: | ---: |
{baseline}

Exact priors, trial losses and both full assessment data versions are retained
in [results](results.json) and the [full-cell CSV](full_cells.csv). Initial models
and every imported/new control remain in the 54 full-trajectory cases.

## First-transition fit probes

Coherent-trained model means on common coherent probes:

| Objective | Cache | Probe | First NLL | First accuracy | Four-patch NLL | Four-patch accuracy | Mean actually drawn probe rows |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
{probe_table}

Anchor and extension probes each contain 512 rows; development has 64 and
assessment 128. Anchor is the original cache. Extension is the first 512 added
rows and is out of the smaller cache. Neither eligible cache membership nor a
probe score means a row was actually used in fitting. Per-seed drawn masks and
drawn/undrawn subset losses/counts are retained; empty subsets are null. The table
reports whole-probe means, not drawn-only means. This is not a complete 4096-row
training-cache assessment. Initial and held-out probes have no training exposure.
The four-patch metric is teacher-forced; later patches see earlier true patches.
Marker NLL is separately preserved. A smaller generalization gap alone is not
evidence of improved held-out prediction.

## Integrity, costs and next decision

All 54 full-trajectory, 108 fit-probe and 27 intervention cases replay exactly
after reopening 27 initial/final files. Imported full, first-patch, intervention
and common fit vectors agree exactly with stages 8/9. The scorer functions are
unchanged; replay is a separate invocation with the same scorers, not independent
loss code or a rerun of fitting. Nested array/shuffle reconstruction, disjoint
phase classes, initial states, sampling exposures and final optimizer counts are
checked. [Verification](verification.json), [summary](summary.json), four CSVs
and nine per-seed raw evidence files preserve every score vector. The
[provenance manifest](PROVENANCE.json) binds 42 exact source/public copies.
Raw arrays, parameters and optimizer moments remain private and preserved.
Checkpoint replay requires the private retained run folders; the compact public
files alone support score-accounting audits, not a weights-free neural replay.

Ten new data/protocol/exposure regressions, including an isolated tiny four-arm
execution fixture with mocked provenance, pass. The full local suite passes
134 tests, including two subsequent publication-policy integrity regressions.
The tiny fixture is software execution evidence, not a scientific
comparison. No remote CI run is implied.

New fitting totals {summary['new_updates']:,} updates / {summary['new_input_tokens']:,}
input tokens. Recorded new per-model CPU timers sum to
{summary['new_stage_seconds']/60:.3f} minutes; the twelve imported controls account
for {summary['historical_stage_seconds']/60:.3f} historical CPU minutes separately.
The comparison totals 24000 updates / 4512000 input tokens across 24 model
budgets, half newly fitted. Generation, source checks, assessment, replay and
software verification add costs. These timings span different invocations and
concurrent local checks; they are not a controlled throughput comparison or GPU
speedup. No reduced-token crossing or language-cost recovery is measured here.

{next_step}

This tests training diversity for one known interacting binary rule and one
inspected split. It does not establish GraphNCA-specific acceleration or
unseen-rule generalization. Confirmation needs fresh joint-phase-disjoint data.
Original random-rule checkpoints remain unready. Any language suite requires a
separately frozen eligible synthetic configuration, fresh paired cohorts and an
identical full language stage. No official-language-test access, language stage,
topology/serializer/budget retuning, paid job, remote GPU run, push or publication
occurred in this study.
'''
    (a.out/'REVIEW.md').write_text(report, encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
