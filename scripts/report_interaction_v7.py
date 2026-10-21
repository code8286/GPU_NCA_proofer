"""Export the verified matched-rule bridge without distributing weights/data."""
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
from interaction_v7.plan import verify_plan, read
from interaction_v7.run import verify_training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'runs/interaction_v7_run1')
    p.add_argument('--out', type=Path, default=ROOT/'docs/results/interaction_v7_run1')
    a = p.parse_args()
    plan = verify_plan(a.run)
    models = verify_training(a.run, plan)
    c = plan['config']
    result, verification = read(a.run/'results.json'), read(a.run/'verification.json')
    checksum = sha_file(a.run/'results.json')
    if checksum != read(a.run/'results_receipt.json')['results_sha256'] or checksum != verification['results_sha256'] or not verification['passed']:
        raise ValueError('Verified result identity differs')
    if c['purpose'] != 'development_interaction' or len(result['cells']) != 36 or result['language_allowed']:
        raise ValueError('Expected the complete declared development run')
    first = read(a.run/'first_transition/results.json')
    first_verification = read(a.run/'first_transition/verification.json')
    if not first_verification['passed'] or first_verification['results_sha256'] != sha_file(a.run/'first_transition/results.json'):
        raise ValueError('Supplementary checkpoint replay required')
    a.out.mkdir(parents=True, exist_ok=False)
    copies = ['plan.json', 'plan_receipt.json', 'FROZEN_PROTOCOL.md', 'training_complete.json', 'training_receipt.json',
              'results.json', 'results_receipt.json', 'verification.json', 'status.json']
    copies += [f"{r['arm']}_{r['seed']}/{name}" for r in models for name in ('metrics.jsonl', 'summary.json')]
    copies += ['first_transition/'+name for name in ('plan.json', 'plan_receipt.json', 'FROZEN_PROTOCOL.md', 'eligibility.json',
                                                      'results.json', 'results_receipt.json', 'verification.json')]
    provenance = {'scope': 'New CPU paired known-rule bridge; no language-transfer result', 'files': {}}
    for name in copies:
        dest = a.out/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.run/name, dest)
        provenance['files'][name] = {'sha256': sha_file(dest), 'origin': 'Frozen CPU spatial bridge: '+name}
    dump(a.out/'PROVENANCE.json', provenance)
    flat = []
    for row in result['cells']:
        values = row['per_initial_condition']
        count = sum(r['patch_targets'] for r in values)
        flat.append({k:row[k] for k in ('rule', 'seed', 'checkpoint_arm', 'data_arm', 'patch_nll')} |
                    {'patch_targets': count, 'patch_accuracy': sum(r['patch_accuracy']*r['patch_targets'] for r in values)/count})
    with (a.out/'assessment_cells.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    cells = {(r['rule'], r['seed'], r['checkpoint_arm'], r['data_arm']):r for r in flat}
    families = []
    for family in result['families']:
        rule = family['rule']
        rows = [r for r in result['decisions'] if r['rule'] == rule]
        coherent = [cells[rule, s, 'coherent', 'coherent'] for s in c['seeds']]
        prefix_gates = [decision(cells[rule, s, 'prefix', 'coherent']['patch_nll'], cells[rule, s, 'prefix', 'prefix']['patch_nll'], family['frequency_nll'], c) for s in c['seeds']]
        families.append({'rule': rule, 'cohorts': len(rows), 'frequency_nll': family['frequency_nll'],
                         'local_table_nll': family['local_table_nll'], 'mean_coherent_nll': statistics.mean(r['patch_nll'] for r in coherent),
                         'mean_coherent_accuracy': statistics.mean(r['patch_accuracy'] for r in coherent),
                         'frequency_passes': sum(r['gate']['frequency_passed'] for r in rows),
                         'ordering_passes': sum(r['gate']['ordering_passed'] for r in rows),
                         'both_passes': sum(r['gate']['passed'] for r in rows),
                         'local_table_advantage_passes': sum(r['local_table_advantage_passed'] for r in rows),
                         'mean_frequency_advantage_nats': statistics.mean(r['gate']['frequency_advantage_nats'] for r in rows),
                         'mean_ordering_advantage_nats': statistics.mean(r['gate']['ordering_advantage_nats'] for r in rows),
                         'mean_local_table_advantage_nats': statistics.mean(r['local_table_advantage_nats'] for r in rows),
                         'prefix_model_gate_passes': sum(r['passed'] for r in prefix_gates),
                         'mean_difference_in_ordering_gaps': statistics.mean(r['difference_in_ordering_gaps'] for r in rows),
                         'oracle_passed': family['oracle_gate']['passed']})
    summary = {k:result[k] for k in ('readiness_passed', 'spatial_coupling_qualified', 'bridge_passed', 'language_allowed')}
    summary.update(families=families, models_trained=len(models), replayed_cells=verification['replayed_cells'],
                   total_updates=sum(r['steps'] for r in models), total_input_tokens=sum(r['input_tokens'] for r in models),
                   stage_seconds=sum(r['stage_seconds'] for r in models), plan_sha256=sha_file(a.run/'plan.json'))
    first_spatial = [r for r in first['cells'] if r['rule'] == 'spatial' and r['arm'] == 'coherent']
    summary.update(stricter_first_transition_passed=first['stricter_first_transition_passed'], eligible_intervention_pairs=first['eligible_pairs'],
                   mean_spatial_first_patch_nll=statistics.mean(r['native_nll'] for r in first_spatial),
                   mean_spatial_first_patch_accuracy=statistics.mean(r['native_accuracy'] for r in first_spatial),
                   mean_spatial_intervention_accuracy=statistics.mean(r['changed_accuracy'] for r in first_spatial),
                   mean_spatial_paired_both_correct=statistics.mean(r['paired_both_correct'] for r in first_spatial))
    dump(a.out/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, family in zip(axes, families):
        rule = family['rule']
        for i, seed in enumerate(c['seeds']):
            for arm, style in (('coherent', '-'), ('prefix', '--')):
                rows = [json.loads(line) for line in (a.run/f'{rule}_{arm}_{seed}/metrics.jsonl').read_text().splitlines()]
                ax.plot([r['step'] for r in rows], [r['development']['patch_nll'] for r in rows], style,
                        color=('#2563eb', '#059669', '#b45309')[i], label=f'{arm}, seed {seed}')
        ax.set(title=rule, xlabel='Fixed optimizer updates', ylabel='Native development patch NLL')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Matched rule bridge: local CPU training')
    for extension in ('png', 'svg'):
        fig.savefig(a.out/f'learning_curves.{extension}', dpi=180)
    plt.close(fig)
    table = '\n'.join(f"| {r['rule']} | {r['seed']} | {r['coherent_nll']:.6f} | {r['gate']['frequency_advantage_nats']:.6f} | {r['gate']['ordering_advantage_nats']:.6f} | {r['local_table_advantage_nats']:.6f} | {'Pass' if r['gate']['passed'] else 'Fail'} |" for r in result['decisions'])
    family_table = '\n'.join(f"| {r['rule']} | {r['mean_coherent_nll']:.6f} | {r['mean_coherent_accuracy']:.2%} | {r['frequency_nll']:.6f} | {r['local_table_nll']:.6f} | {r['prefix_model_gate_passes']}/3 |" for r in families)
    oracle_table = '\n'.join(f"| {r['rule']} | {r['oracle_nll']['coherent']:.6f} | {r['oracle_nll']['prefix']:.6f} | {'Pass' if r['oracle_gate']['passed'] else 'Fail'} |" for r in result['families'])
    first_table = '\n'.join(f"| {r['seed']} | {r['native_nll']:.6f} | {r['native_accuracy']:.2%} | {r['changed_accuracy']:.2%} | {r['paired_both_correct']:.2%} | {'Pass' if r['stricter_first_transition_passed'] else 'Fail'} |" for r in first_spatial)
    first_control_rows = []
    for rule in ('complement', 'spatial'):
        for arm in ('coherent', 'prefix'):
            rows = [r for r in first['cells'] if r['rule'] == rule and r['arm'] == arm]
            first_control_rows.append(f"| {rule} | {arm} | {statistics.mean(r['native_nll'] for r in rows):.6f} | {statistics.mean(r['native_accuracy'] for r in rows):.2%} | {statistics.mean(r['changed_accuracy'] for r in rows):.2%} | {sum(r['stricter_first_transition_passed'] for r in rows)}/3 |")
    first_control_table = '\n'.join(first_control_rows)
    next_step = ('The bridge clears both readiness and the separately declared local-table qualification. Review the first-transition/intervention errors before declaring the next single change; a residual first-transition bottleneck should be addressed before increasing rule diversity. Measure held-out-rule learning before returning to a language suite.'
                 if result['bridge_passed'] else
                 'The bridge is incomplete under the frozen criteria. Preserve every arm and inspect the failing readiness/local-table comparison before declaring one new development change. Do not extend the budget, relax margins or discard seeds in this run.')
    if not first['stricter_first_transition_passed']:
        next_step += ' The separately frozen first-transition diagnostic fails; investigate that first-transition bottleneck before attributing the primary local-table advantage to spatial computation.'
    if summary['mean_spatial_first_patch_accuracy'] < 1:
        next_step += ' The concrete next development proposal is a matched first-transition loss-weighting ablation: keep this spatial rule, model, data and total update/token budget fixed, and change only the relative weight assigned to the first transition. Freeze its protocol before fitting and retain standard-loss and shuffled controls. These inspected initial conditions are development data; confirmation needs fresh joint-phase-disjoint data. This is a proposed new study, not a retuning or extra run performed here.'
    report = f'''# Stage 7: matched spatial-interaction bridge

Completed 20 October 2026 on CPU. Readiness: **{result['readiness_passed']}**;
spatial local-table qualification: **{result['spatial_coupling_qualified']}**;
combined bridge: **{result['bridge_passed']}**. Language training remains disabled.
Separately frozen stricter first-transition diagnostic:
**{first['stricter_first_transition_passed']}**.
All twelve models finished the fixed 1000-update budget. The [protocol](FROZEN_PROTOCOL.md),
[plan](plan.json) and [receipt](plan_receipt.json) were frozen before training:
`{summary['plan_sha256']}`.

## One spatial treatment and a new matched reference

The reference flips every binary cell: x' = 1 XOR x. Treatment adds a neighbor
parity term: x' = 1 XOR x XOR left XOR right, with periodic horizontal boundaries.
It creates a genuine dependence across 2x2 patch boundaries. Both maps are
bijective involutions on this 4x4 lattice; neither collapses after a few steps.
Exhaustive tests cover all 65,536 initial states. The added interaction changes
phase-equivalence classes, so BOTH rules use a new shared split excluding their
complete joint orbits. Do not compare historical stage 6 and stage 7 as a
controlled treatment contrast. Its earlier [review](../trainability_v6_run1/REVIEW.md)
remains unchanged evidence.

Inside this stage the initial grids, 512/64/128 split sizes, actual starting
weights, model/data seeds and minibatch indices match across all four arms in
each seed. Joint-orbit classes never cross splits under either rule. Alphabet,
4x4 grid, eight frames, context 47, 2x2 serializer, full six-layer/192-wide/four-head
backbone, tied head, Adam/LR schedule, accumulation and 1000-update endpoint
retain stage 6 settings. Numerical updates and scores import unchanged v4/v5
code. The rule alone differs between the new reference and spatial treatments.
Three seeds share one split and two fixed rules; they are initialization
replicates, not independent random-rule/data/topology cohorts.

## Frozen assessment and results

No assessment likelihoods were computed until all twelve full-budget models
finished. Final checkpoints are used without selection or early stopping.
Each cell scores 128 held-out initial conditions x 28 patches = 3584 targets.
Each rule/seed has initial, coherent-trained and shuffle-trained checkpoints
scored on coherent and matched shuffled data: 36 cells total. Initial frames,
markers and supervised target histograms match within each shuffle pair.

Both inherited readiness advantages must be >=0.01 nats per patch. A separate,
prospectively declared spatial qualification requires all three spatial models
to beat a train-only add-one p(next patch | previous SAME patch) table by >=0.01.
This table lacks neighboring-patch information. That qualification is reported
separately; the reference table is not required to lose to its transformer.

| Rule | Seed | Coherent NLL | Frequency advantage | Ordering advantage | Own-patch table advantage | Both readiness checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{table}

Higher advantages are better. The local-table comparison is a finite learned
baseline, not a Bayes-optimal bound over all predictors using local information.
It does not prove an attention mechanism or the precise circuit learned.

| Rule | Mean coherent NLL | Mean patch accuracy | Frequency NLL | Own-patch table NLL | Shuffle-trained readiness passes |
| --- | ---: | ---: | ---: | ---: | ---: |
{family_table}

Shuffle-trained checkpoint gates are descriptive controls, not additional
admission conditions. Their ordering gaps and differences from coherent-trained
gaps remain in [full results](results.json). Readiness is predictive competence,
not a causal proof that coherent pre-pre-training produces language gains.
Loss levels belong to different target distributions and are not a language
performance comparison or evidence that the spatial task trains faster.

The causal frequency predictor is fitted only on each rule's training data;
its joint score is invariant under the matched shuffle. The positive control
uses each known rule and the actually observed previous FULL frame, assigning
.99 to its predicted next patch and .01/15 to other patch tokens. It is an
oracle measurement control, not a learned model. It replaces the own-patch
table control for both new treatments because that table cannot represent the
spatial rule. This evaluation change does not alter training or readiness margins.

| Oracle rule | Coherent NLL | Matched shuffled NLL | Both readiness checks |
| --- | ---: | ---: | --- |
{oracle_table}

Both frequency-only negative controls correctly fail zero self/ordering
advantages. All per-initial-condition losses/accuracies/counts and the
[36-cell CSV](assessment_cells.csv) are retained. Source-run copies have
[SHA provenance](PROVENANCE.json); [summary](summary.json) and
[verification](verification.json) bind the result.

![Fixed-budget development curves](learning_curves.png)

## Verification, scope and next step

All 36 cells and decisions replay exactly after reopening the three initial
and twelve final checkpoint files, including final optimizer counts. This is
a separate replay using the same scorer, not independent loss code or a training
rerun. Frozen identities cover sources, arrays, split classes, initial states,
summaries and final models. Private weights, optimizer moments and raw arrays
remain preserved outside publication. Ten new structural/integrity regressions
and a retained tiny four-arm execution fixture are separate from this scientific
run. The full local software suite passes 104 tests; configured CI is not a
completed remote CI run.

Recorded per-model CPU stage timers sum to {summary['stage_seconds']/60:.3f}
minutes, including setup, development evaluation and checkpoint writing. Total
updates/input tokens: {summary['total_updates']:,}/{summary['total_input_tokens']:,}.
Assessment/replay/export are additional work. These are local CPU timings,
not GPU speedups or cloud billing measurements.

## Separately frozen first transition and neighbor intervention

Both rules have period two, so later-frame own-patch recall could beat a
one-frame local table. The primary qualification alone does not rule out this
shortcut. During training, before any primary assessment or final-checkpoint
assessment was inspected, a separate [supplementary plan](first_transition/plan.json)
and [protocol](first_transition/FROZEN_PROTOCOL.md) were frozen. This diagnostic
was not part of the original pre-training plan and does not change its status.

It scores only the FIRST patch of the FIRST transition, conditioned on the full
initial frame and next start marker. No repeated prior frame or new-frame patch
is available. All 128 original queries remain. Flip initial cell (0,3), outside
the target patch but adjacent through the periodic boundary: its spatial target
patch must flip bit zero; its complement target stays unchanged. A shared mask
excludes changed initial states whose joint orbit occurs in train/development,
leaving {first['eligible_pairs']} eligible pairs, fixed before scoring. No primary
row or seed is dropped. Compare original first-patch NLL to BOTH a train-fitted
first-patch conditional table and uniform log(16), with a separate 0.01-nat
criterion. The stricter status above requires all three spatial coherent models
to pass. Counterfactual accuracy is reported without a post-hoc acceptance cutoff.

| Spatial seed | First-patch NLL | Original accuracy | Changed-query accuracy | Both correct in eligible pair | Stricter check |
| --- | ---: | ---: | ---: | ---: | --- |
{first_table}

Mean original first-patch accuracy is {summary['mean_spatial_first_patch_accuracy']:.2%};
mean eligible intervention accuracy is {summary['mean_spatial_intervention_accuracy']:.2%}.
All 18 initial/coherent/shuffle first-transition cells and their per-query values
are retained in [supplementary results](first_transition/results.json) and replay
exactly from the same saved model files. These predictions address this queried
neighbor perturbation, not unseen-rule induction or a unique internal circuit.
The new diagnostic preserves all primary and historical outcomes.

The full-trajectory spatial accuracy ({families[1]['mean_coherent_accuracy']:.2%})
therefore overstates performance at the first spatial prediction
({summary['mean_spatial_first_patch_accuracy']:.2%}). Only
{summary['mean_spatial_paired_both_correct']:.2%} of eligible intervention pairs
have BOTH target patches predicted correctly. This is partial known-rule
learning, not reliable execution across all queried initial conditions.

For context, means across the same three seeds for both trained controls:

| Rule | Training data | First-patch NLL | Original accuracy | Changed accuracy | Stricter first-transition passes |
| --- | --- | ---: | ---: | ---: | ---: |
{first_control_table}

These control comparisons are descriptive. They do not redefine the primary
decision or make the supplementary diagnostic a test of language transfer.
In particular, inspect the spatial shuffle-trained control's passing count:
the small NLL margin alone is not a causal test of coherent-training benefit.

{next_step}

This tests one fixed interacting rule on a deliberately small binary task.
It does not establish unseen-rule generalization, GraphNCA acceleration or cost
recovery. Original random-rule checkpoints retain failed stage 5 readiness.
No language-training command, graph rewiring, serializer change, paid job,
remote GPU run or official-language-test access occurred. Any later transfer
study still needs its own synthetic readiness, fresh paired cohorts and an
identical full language stage.
'''
    (a.out/'REVIEW.md').write_text(report, encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
