"""Export fresh-phase confirmation with both objective strata and every control."""
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
from confirmation_v11.plan import read, verify_plan, OBJECTIVES, ARMS
from confirmation_v11.run import verify_training


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_csv_rows(pairs):
    """Keep the all-criteria decision separate from the readiness-only gate."""
    return [{k: v for k, v in r.items() if k != 'readiness'} |
            {'readiness_'+k: v for k, v in r['readiness'].items()} for r in pairs]


def raw_evidence(run, result):
    raw = {k: [] for k in ('full_cells', 'fit_cells', 'intervention_cells')}
    for name, checksum in result['evidence_sha256'].items():
        if sha_file(run/name) != checksum:
            raise ValueError('Raw evidence changed')
        key = name.split('/')[-1].rsplit('_', 1)[0]
        raw[key].extend(read(run/name))
    return raw


def summarize(plan, result, models, raw, recovery):
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
               'study_primary_passed': result['study_primary_passed'], 'both_objectives_passed': result['both_objectives_passed'], 'language_allowed': False,
               'new_models': len(models), 'imported_controls': 0,
               'new_updates': sum(r['steps'] for r in models), 'new_input_tokens': sum(r['input_tokens'] for r in models),
               'new_stage_seconds': sum(r['stage_seconds'] for r in models),
               'historical_stage_seconds': 0.,
               'groups': groups, 'coherent_fit_probes': fits, 'paired_comparisons': result['paired_comparisons'],
               'unique_exposures': {k: v['unique_rows'] for k, v in plan['expected_orders'].items()},
               'replayed_full_cells': len(result['full_cells']), 'replayed_fit_cells': len(result['fit_cells']),
               'replayed_intervention_cells': len(result['intervention_cells'])}
    summary.update(extra_aborted_updates=recovery['extra_aborted_updates'],
                   extra_aborted_input_tokens=recovery['extra_aborted_input_tokens'],
                   extra_aborted_prediction_targets=recovery['extra_aborted_prediction_targets'],
                   actual_updates_including_aborted=summary['new_updates']+recovery['extra_aborted_updates'],
                   actual_input_tokens_including_aborted=summary['new_input_tokens']+recovery['extra_aborted_input_tokens'],
                   final_endpoint_prediction_targets=sum(r['prediction_targets'] for r in models),
                   actual_prediction_targets_including_aborted=sum(r['prediction_targets'] for r in models)+recovery['extra_aborted_prediction_targets'],
                   extra_aborted_recorded_stage_seconds_lower_bound=sum(e['metrics'][-1]['stage_seconds'] for e in recovery['interrupted_models'].values()),
                   failed_training_command_seconds=recovery['failed_event']['duration_seconds'])
    return summary, full


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'runs/confirmation_v11_run1')
    p.add_argument('--out', type=Path, default=ROOT/'docs/results/confirmation_v11_run1')
    a = p.parse_args()
    plan = verify_plan(a.run)
    models = verify_training(a.run, plan)
    result, verification = read(a.run/'results.json'), read(a.run/'verification.json')
    checksum = sha_file(a.run/'results.json')
    if checksum != verification['results_sha256'] or checksum != read(a.run/'results_receipt.json')['results_sha256'] or not verification['passed']:
        raise ValueError('Complete verified results required')
    raw = raw_evidence(a.run, result)
    recovery = read(a.run/'execution_recovery.json')
    recovery_check = read(a.run/'execution_recovery_verification_6950.json')
    if not recovery_check['passed'] or recovery_check['training_sha256'] != sha_file(a.run/'training_complete.json'):
        raise ValueError('Completed recovery verification required')
    if sha_file(a.run/'execution_recovery.json') != read(a.run/'execution_recovery_receipt.json')['sha256'] or recovery_check['recovery_sha256'] != sha_file(a.run/'execution_recovery.json'):
        raise ValueError('Recovery identity changed')
    summary, full = summarize(plan, result, models, raw, recovery)
    a.out.mkdir(parents=True, exist_ok=False)
    copies = ['plan.json', 'plan_receipt.json', 'FROZEN_PROTOCOL.md', 'training_complete.json', 'training_receipt.json',
              'results.json', 'results_receipt.json', 'verification.json', 'status.json']
    copies += list(result['evidence_sha256'])
    copies += [f"{r['level']}_{r['objective']}_{r['arm']}_{r['seed']}/{name}" for r in models for name in ('metrics.jsonl', 'summary.json')]
    copies += ['execution_recovery.json', 'execution_recovery_receipt.json', 'execution_recovery_verification_6950.json',
               'execution_interruptions/4096_standard_prefix_211/metrics.jsonl', 'execution_interruptions/status_at_failure.json.tmp']
    provenance = {'scope': result['scope'], 'files': {}}
    for name in copies:
        target = a.out/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.run/name, target)
        provenance['files'][name] = {'sha256': sha_file(target), 'origin': 'Frozen fresh-phase confirmation: '+name}
    dump(a.out/'PROVENANCE.json', provenance)
    dump(a.out/'summary.json', summary)
    csv_write(a.out/'full_cells.csv', full)
    fit_rows = []
    for row in result['fit_cells']:
        fit_rows.append({k: v for k, v in row.items() if k not in ('drawn_subset', 'undrawn_subset')} |
                        {'drawn_'+k: v for k, v in row['drawn_subset'].items()} | {'undrawn_'+k: v for k, v in row['undrawn_subset'].items()})
    csv_write(a.out/'fit_cells.csv', fit_rows)
    csv_write(a.out/'intervention_cells.csv', result['intervention_cells'])
    csv_write(a.out/'paired_comparisons.csv', pair_csv_rows(result['paired_comparisons']))
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
        axes[0, col].set(title=objective+' objective', ylabel='First-patch NLL (lower is better)', xticks=range(2), xticklabels=['Native', 'Neighbor intervention'])
        axes[0, col].set_ylim(0, axes[0, col].get_ylim()[1]*1.18)
        axes[1, col].set(ylabel='Accuracy (%)', ylim=(0, 100), xticks=range(3), xticklabels=['First patch', 'Intervention', 'Full trajectory'])
        for ax in axes[:, col]:
            ax.grid(axis='y', alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Fresh-phase confirmation: fixed 1000-update CPU endpoints')
    for extension in ('png', 'svg'):
        fig.savefig(a.out/f'confirmation_comparison.{extension}', dpi=180)
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
        next_step = 'The frozen weighted confirmation passes. Plan a separately declared bridge toward interacting categorical rules, changing one task property at a time and retaining both objectives and shuffled controls. This fixed-rule success does not authorize a language suite.'
    else:
        next_step = 'The frozen weighted confirmation fails. Preserve every seed and endpoint. Inspect the already retained training/held-out first-transition probes and coherent/shuffled contrasts before proposing another development treatment; do not extend this study or loosen its criterion.'
    report = f'''# Stage 11: fresh-phase confirmation

Completed on local CPU. **Weighted primary confirmation:
{result['study_primary_passed']}**. Standard stratum:
**{result['objective_strata_passed']['standard']}**; broader both-objective
criterion: **{result['both_objectives_passed']}**. Language remains gated.
All 24 endpoints were fitted anew; no historical endpoint enters this comparison.

The [protocol](FROZEN_PROTOCOL.md), [plan](plan.json) and
[receipt](plan_receipt.json) froze before fitting. Plan SHA-256:
`{summary['plan_sha256']}`. This is a prospective confirmation of the weighted
hypothesis generated by [stage 10](../diversity_v10_run1/REVIEW.md), whose
both-objective primary remains failed. Making weighted confirmation primary in
this separate declaration does not replace that earlier outcome.

## Fresh data, unchanged treatment

Exclude {plan['historical_forbidden_phase_count']} historical joint phase classes
from controlled binary stages 6–10, including every old intervention partner.
Using the frozen seed, reserve 128 assessment states and their 128 changed
partners first, then select 64 development and 4096 training states. All joint
classes are disjoint across these groups and from historical classes. All 128
intervention pairs are eligible by construction, rather than selected by score.
There is no spatial-translation-disjointness claim. The small cache is the exact
first 512 rows of the larger coherent AND prefix-shuffled arrays.

Three fresh initialization seeds share ONE new data split. These are paired
initialization replicates, not independently sampled rule/data cohorts. Within
each seed, eight fits share actual saved initial weights and fresh Adam states.
Retain the known binary spatial rule, 4x4 periodic grid, eight frames, 2x2 patch
serializer, alphabet 18/context 47, six-layer/192-wide/four-head backbone,
standard factor-one versus normalized first-transition factor-six objectives,
and original optimizer/schedule/batch settings. Each model receives 1000 updates,
188000 input tokens and 168000 supervised targets. Only fresh data and seeds
change relative to the declared diversity comparison.

Uniform replacement sampling makes 4000 row draws per fit. Four arms within
each seed/cache use the exact same sampled indices. Cache sizes share RNG seed
and draw count, but changing the sampling bound changes selected examples.

| Seed | Distinct rows drawn from 512 | Distinct rows drawn from 4096 |
| --- | ---: | ---: |
{coverage}

## Frozen comparisons

The weighted primary requires ALL three coherent seeds to improve native AND
intervention first-patch NLL by at least 0.01, while the larger-cache model
retains frequency and ordering advantages of at least 0.01 on full trajectories.
The identical standard comparison is retained separately. The broader criterion
requires BOTH objective strata. These fixed criteria are not a significance test.
Positive improvements favor the larger cache. No seed, threshold, objective,
budget or final checkpoint was changed after assessment.

| Objective | Seed | Native NLL improvement | Intervention NLL improvement | Larger-cache readiness | All criteria |
| --- | ---: | ---: | ---: | --- | --- |
{primary}

Coherent-trained means across the three initialization seeds:

| Objective | Cache | First NLL | First accuracy | Intervention NLL | Intervention accuracy | Both correct | Full NLL | Full accuracy | Readiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{coherent_table}

![Fresh-phase fixed-budget comparison](confirmation_comparison.png)

Bars show means; NLL dots show individual seeds. These are fixed-budget final
endpoints, not time-to-target learning curves. First-patch predictions see only
the initial frame and next start marker. The intervention flips initial cell
(0,3), outside the target patch, changing the correct target patch's bit zero.
Paired correctness requires both native and changed predictions correct.

## Shuffled controls and frequency baseline

Shuffle-trained checkpoints scored on coherent assessment trajectories:

| Objective | Cache | First NLL | First accuracy | Intervention NLL | Intervention accuracy | Both correct | Full NLL | Full accuracy | Readiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{shuffled_table}

| Objective | Seed | Shuffled first NLL diversity gain | Coherent gain minus shuffled gain |
| --- | ---: | ---: | ---: |
{controls}

These contrasts are descriptive. A coherent-model gain alone does not establish
a dynamics-specific cause. Full-trajectory readiness can conceal poor first
predictions; retained shuffled-control readiness should be assessed alongside
the actual first-transition errors. This known rule is involutive: a state
returns after two updates, so full-trajectory targets contain repeated frames.
The short-prefix and neighbor-intervention probes avoid conditioning on those
later repetitions. This observation does not identify a learned circuit.
A separate causal frequency baseline is
calibrated on each training cache using the unchanged train-only policy.

| Training cache | Assessment causal frequency NLL | Concentration |
| ---: | ---: | ---: |
{baseline}

Exact priors, concentration trials and coherent/shuffled scores remain in
[results](results.json) and the [full-cell CSV](full_cells.csv).
In the paired CSV, `passed` is the complete comparison and `readiness_passed`
is the trajectory gate. The earlier stage 10 CSV has a documented
[column-collision caveat](../../STAGE10_CSV_CAVEAT.md); its JSON/review decision
remains correct and its original evidence is preserved.

## Training versus held-out probes

| Objective | Cache | Probe | First NLL | First accuracy | Four-patch NLL | Four-patch accuracy | Mean actually drawn probe rows |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
{probe_table}

Anchor and extension each contain 512 rows, development 64 and assessment 128.
Anchor is the small cache; extension is the first 512 additional large-cache
rows. Scores include all probe rows, not only those drawn. Retain actual exposure
masks and drawn/undrawn subset statistics, with null for empty subsets. This is
not an exhaustive 4096-row training assessment. Later first-transition patches
are teacher-forced; they see earlier true patches. Marker loss is separate.
Short/long first-patch scores agree within 1e-5 with identical correctness.

## Verification, costs and next decision

All 54 full-trajectory, 108 fit-probe and 27 intervention cases replay exactly
after reopening three initial and 24 final checkpoints in a separate invocation.
The original scorer functions are unchanged. This is scorer replay, not an
independent implementation of the loss or a rerun of fitting. Final optimizer
step counts, actual initial states, input/source/receipt identities and sampled
indices are checked. [Verification](verification.json), [summary](summary.json),
four CSVs and nine per-seed raw files retain every vector and decision.
The [provenance manifest](PROVENANCE.json) binds 71 exact evidence copies.
Private arrays, weights and optimizer moments remain preserved; compact public
files support accounting audits but neural replay needs those private artifacts.

New fitting totals {summary['new_models']} models, {summary['new_updates']:,}
updates, {summary['new_input_tokens']:,} input tokens and 4,032,000 supervised
targets. Recorded per-model CPU timers sum to
{summary['new_stage_seconds']/60:.3f} minutes. Generation, source checks,
assessment, replay and software verification add cost. Concurrent local checks
and invocation overhead prevent interpreting these timers as a controlled
throughput comparison. No GPU, language-token speedup or cost recovery is measured.

The initial command failed on a Windows status-file replacement after 21 final
models and 800 updates of the next fit. The separately journaled
[execution recovery](execution_recovery.json) preserves all 21 endpoints and
the interrupted [metrics](execution_interruptions/4096_standard_prefix_211/metrics.jsonl).
Its completed-folder guard and JSON retries leave the frozen numerical body
unchanged. Restarting the unfinished fit from its original initialization
reproduces all recorded non-time metrics through update 800; the remaining two
fits also finish. The [recovery verification](execution_recovery_verification_6950.json)
checks the exact executed helper, unchanged retained weights and all endpoints.
An overlapping alternative preparation stops before fitting and is retained
privately; the [recovery note](../../CONFIRMATION_V11_RECOVERY.md) explains it.

Charge the discarded 800 updates as well: actual training consumes
{summary['actual_updates_including_aborted']:,} updates,
{summary['actual_input_tokens_including_aborted']:,} input tokens and
{summary['actual_prediction_targets_including_aborted']:,} targets. The last
interrupted stage timer records {summary['extra_aborted_recorded_stage_seconds_lower_bound']:.6f}
seconds, a lower bound; the failed whole command lasts
{summary['failed_training_command_seconds']:.6f} seconds. Final-endpoint timers
above exclude this discarded attempt. No extra endpoint is selected or assessed.

{next_step}

This confirms or rejects one predeclared endpoint hypothesis on a known
interacting binary rule and one fresh split. It does not establish GraphNCA
acceleration, unseen-rule generalization or superiority over standard loss.
Original random-rule checkpoints remain unready. Any future language comparison
needs a separately frozen eligible synthetic configuration, fresh paired cohorts
and an identical full language stage. No official-language-test access, topology/
serializer/budget retuning, remote/paid training, push or publication occurred.
'''
    (a.out/'REVIEW.md').write_text(report, encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
