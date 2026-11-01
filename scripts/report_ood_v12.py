"""Export the validated OOD measurement and retain the historical format floor."""
import csv
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ood_v12.plan import verify_plan, read
from transfer_v4.common import dump, sha_file


def main():
    run = ROOT/'runs/ood_v12_run1'
    out = ROOT/'docs/results/ood_v12_run1'
    plan = verify_plan(run)
    verification, controls, legacy = [read(run/name) for name in ('verification.json', 'controls.json', 'legacy_results.json')]
    if not verification['passed'] or verification['controls_sha256'] != sha_file(run/'controls.json') or verification['results_sha256'] != sha_file(run/'legacy_results.json'):
        raise ValueError('Complete generation replay required')
    copies = ['plan.json', 'plan_receipt.json', 'FROZEN_PROTOCOL.md', 'controls.json', 'controls_receipt.json',
              'legacy_results.json', 'legacy_results_receipt.json', 'verification.json']+list(legacy['prediction_files_sha256'])
    out.mkdir(parents=True, exist_ok=False)
    provenance = {'scope': 'New CPU OOD control checks and retained historical checkpoint inference, not new training/GPU measurement', 'files': {}}
    for name in copies:
        p = out/name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run/name, p)
        provenance['files'][name] = sha_file(p)
    dump(out/'PROVENANCE.json', provenance)
    summary = {'plan_sha256': sha_file(run/'plan.json'), 'independent_oracle_examples': controls['independent_oracle_examples'],
               'data_inventory': plan['data_inventory'], 'legacy_predictions_replayed': verification['legacy_predictions_replayed'],
               'models': [{k:r[k] for k in ('arm', 'seed', 'stop_reasons', 'prediction_seconds')} | {k:r['score'][k] for k in ('overall_exact_match', 'overall_valid_format', 'coverage', 'native_and_cut_correct')} for r in legacy['models']],
               'scientific_optimizer_updates': 0, 'new_gpu_measurements': 0, 'graph_acceleration_claim': None,
               'reserved_confirmation_generated': False, 'language_training_allowed': False}
    dump(out/'summary.json', summary)
    flat = [dict(arm=m['arm'], seed=m['seed'], **r) for m in legacy['models'] for r in m['score']['cells']]
    with (out/'legacy_cells.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    models = '\n'.join(f"| {r['arm']} | 0/32 | {r['overall_valid_format']:.0%} | {r['coverage']:.0%} | {r['native_and_cut_correct']:.0%} |" for r in summary['models'])
    # Do not replace a future nonzero outcome with a hard-coded narrative.
    assert all(r['overall_exact_match'] == r['overall_valid_format'] == 0 and r['coverage'] == 1 for r in summary['models'])
    ctable = '\n'.join(f"| {k} | {v['overall_exact_match']:.2%} | {v['native_and_cut_correct']:.2%} | {v['native_and_renamed_correct']:.2%} |" for k,v in controls['controls'].items())
    inventory = '\n'.join(f"| {k} | {v['examples']} | {v['base_graph_instances']} | {v['base_topology_clusters']} | {v['reserved_topology_clusters_including_cuts']} |" for k,v in plan['data_inventory'].items())
    report = f'''# Stage 12: out-of-domain graph evaluation

Completed local CPU implementation and diagnostic, 30 October 2026.

**The OOD measurement layer works, but the historical zero-shot models hit a
formatting floor.** All five retained language models have 0/32 correct answers
and 0/32 valid distance vectors. Every prompt fits its original context. This
is not evidence that graph pre-pre-training has no OOD benefit: these models
were not taught the task interface, and the tiny single-cohort calibration
cannot distinguish reasoning competence from task elicitation.

## Methodological change and source

Private methodological feedback motivated explicit evaluation beyond training
support, with CLRS-Text as inspiration. Private correspondence, contacts and
attachments are excluded; no supervision, collaboration or endorsement is
implied, and no reply was sent. The public basis is the
[CLRS-Text paper v1](https://arxiv.org/html/2406.04229v1) and
[pinned official source](../../../sources/clrs-text-reference/reference.json).
The OpenReview landing page required browser verification, so the accessible
official arXiv version and source were used. Seven reference files and their
Apache-2.0 license are retained unchanged.

Our BFS/Dijkstra **distance-vector** tasks are local CLRS-inspired adaptations.
They are not official CLRS-Text predecessor-output tasks or published benchmark
scores. The separate official-export scorer retains original reference strings.
See the [protocol](FROZEN_PROTOCOL.md) and [usage guide](../../OOD_V12_USAGE.md).

Plan frozen before model predictions:
`{summary['plan_sha256']}`.

## Distributions, identities and functional controls

ID sizes are 8 and 10 nodes, from connected cyclic ER graphs. Size shift uses
12,16,20 nodes. Family shift uses branching recursive trees, paths and a forest
of two recursive trees at ID sizes. Joint shift changes both size and family.
The four views are native, consistent node relabeling, edge-order permutation
and a source-edge cut that provably changes at least one correct distance.
Report correctness on BOTH original and changed prompts, not merely a response
change. Relabel/order consistency of wrong answers cannot count as competence.

| Split | Task/view examples | Base graph instances | Base topology clusters | Reserved clusters incl. cuts |
| --- | ---: | ---: | ---: | ---: |
{inventory}

Unweighted WL invariants conservatively exclude isomorphic base/intervention
graphs across splits, ignoring weights, source and display labels. They are not
complete graph-isomorphism canonicalizers. Within-split repeated rooted/labeled/
weighted versions remain, particularly paths; instance counts are not numbers
of independent topology cohorts. Cell-level scores are retained; pooled numbers
are descriptive. These are development data, not reserved fresh confirmation.

Both graph algorithms' labels agree with independent Floyd-Warshall on all
5,760 written examples. The 5,120 assessment prompts cover 20 graph cells,
two tasks and four views. Measurement controls:

| Predictor/control | Overall vector exact match | Native AND cut correct | Native AND rename correct |
| --- | ---: | ---: | ---: |
{ctable}

The cached-native control deliberately knows the original answer and ignores
the changed input. Its 75% pooled score is an artifact of three unchanged-answer
views; paired native/cut correctness correctly rejects it. Source-only ignores
all edges; occasionally a cut makes that answer correct, but paired correctness
still rejects it. These validate the measurement, not learned rule following.

## Historical zero-shot calibration

Use final language checkpoints of the first declared v4 cohort, seed 59. All
five share the recorded 2,000-update /8,192,000-token language stage, its initial
head, actual data order and LR schedule. Original PPT readiness failures and
negative graph-specific primary remain unchanged. No inspected checkpoint is
continued, retuned or selected by new OOD outcomes.

Thirty-two prompts/model come from one base graph each in four cells: ID ER8,
size ER16, family path8, joint path16, each with both tasks and all four views.
This is tiny zero-shot interface calibration, not a representative OOD study.

| Initialization before the historical language stage | Correct answers | Valid vectors | Context coverage | Native AND cut correct |
| --- | ---: | ---: | ---: | ---: |
{models}

All 160 completions stop at an irrecoverably invalid vector prefix, with raw
token IDs/text/stop reasons retained under `predictions/`. Greedy logits are
unconstrained; early rejection does not force any token. There were no context
overflows: the full assessment's largest prompt is
{plan['tokenizer']['max_assessment_prompt_tokens']} GPT-2 tokens, longest reference
{plan['tokenizer']['max_reference_tokens']}, with a 128-token generation budget
inside the original 1024 context. No input truncation, head replacement or
context extrapolation occurred. Teacher-forced likelihood was not substituted
for generated-answer correctness.

Because even ID answers are invalid, an ID-to-OOD generalization gap is not
interpretable here. Five identical zero scores do not establish equivalence,
no-effect or a negative causal conclusion. The next learned test must first
establish ID task competence with a matched adaptation procedure and check
format validity separately from reasoning. Stage 12 does not run that training.

## Verification and practical use

All 160 generated outputs and score/stop records reproduce exactly after
reopening the five checkpoints in a separate invocation. Independent graph
labels and all measurement controls reproduce too. This is same-code neural
replay, not independently implemented neural inference or repeated fitting.
The [verification](verification.json), [summary](summary.json), per-cell CSV and
[provenance](PROVENANCE.json) bind 13 exact source/public evidence copies.
Private graph datasets, model weights and tokenizer caches stay excluded.

All 14 new regression checks and the 160-test local suite pass. Isolated tiny
training fixtures in the software suite are execution checks, not new scientific
fits. Initial sandbox DNS/cache setup failures remain in the append-only log;
an explicit private workspace tokenizer cache makes evaluation offline thereafter.
The evaluated models' CPU prediction timers sum to
{sum(r['prediction_seconds'] for r in summary['models']):.6f} seconds, including
per-model source/weight checks and loads but excluding initial preparation,
measurement controls and replay. No training throughput,
learning-speed ratio or GPU result is inferred from this time.

The reusable layer supports independently generated local development datasets,
strict JSONL prediction scoring, official CLRS final-reference exports, safe
local checkpoint inference, censored ID-development target crossings and
future matched-stage metadata checks. Historical replay additionally needs the
retained private v4 checkpoint folders. No Gmail access is needed to run code.

## Next declared study

First freeze an ID task-interface/trainability calibration and demonstrate
meaningful ID exact-match performance. Preserve scratch/grid/graph and both
prefix controls, and give every arm the SAME language and algorithmic-adaptation
procedure/budget. A formatting-floor comparison cannot answer the graph question.

The later graph-specific confirmation needs eligible synthetic initialization,
the separately required categorical/random-rule readiness bridge, fresh paired
rule/data/topology cohorts, ID-only selection, predeclared OOD cells/targets,
censoring and multiplicity, and total-cost accounting. Compare graph against
BOTH grid and its matched shuffle; retain scratch and prefix-grid. Improved
fixed-budget OOD accuracy and learning acceleration are different claims.

Stage 11's weighted confirmation and failed broad criterion remain intact. The
original random-rule checkpoints remain unready. No new scientific fitting,
official language test, remote/paid GPU, push, publication or email reply occurred.
'''
    (out/'REVIEW.md').write_text(report, encoding='utf-8', newline='\n')
    print('Exported 13 exact evidence copies, review, summary and per-cell CSV; zero-shot format floor preserved.')


if __name__ == '__main__':
    main()
