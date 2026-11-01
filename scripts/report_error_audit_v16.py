"""Render descriptive stage 16 findings from verified frozen output counters."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from error_audit_v16.run import PUBLIC,verify_plan
from ood_v12.plan import read
from transfer_v4.common import sha_file


def percent(x):return 'not observed' if x is None else f'{x:.2%}'


def render():
    p=verify_plan();r=read(PUBLIC/'results.json');v=read(PUBLIC/'verification.json')
    if not v['passed'] or v['results_sha256']!=sha_file(PUBLIC/'results.json'):
        raise ValueError('Separate semantic replay required')
    index={(x['arm'],x['variant'],x['probe'],x['group']):x for x in r['rows']}
    get=lambda arm,variant,probe:index[arm,variant,probe,'pooled']
    cache=[];dev=[];hops=[];signed=[];pairs=[]
    for arm in p['arms']:
        s=get(arm,'small16','cache');b=get(arm,'reference512','cache')
        cache.append(f"| {arm} | {s['exact_examples']}/16 / {b['exact_examples']}/16 | {percent(s['teacher']['digit_accuracy'])} / {percent(b['teacher']['digit_accuracy'])} | {percent(s['node_micro_accuracy'])} / {percent(b['node_micro_accuracy'])} |")
        for variant in p['variants']:
            x=get(arm,variant,'development');t=x['teacher']
            dev.append(f"| {arm} | {variant} | {x['valid_examples']}/128 | {percent(x['node_micro_accuracy'])} | {percent(x['valid_node_accuracy'])} ({x['correct_nodes']}/{x['valid_nodes']}) | {percent(t['digit_accuracy'])} | {percent(t['other_accuracy'])} |")
            hops.append(f"| {arm} | {variant} | {x['source_correct']}/{x['source_nodes']} | {x['one_hop_correct']}/{x['one_hop_nodes']} | {x['multi_hop_correct']}/{x['multi_hop_nodes']} | {x['unreachable_correct']}/{x['unreachable_nodes']} |")
            hist=', '.join(f'{k}:{value}' for k,value in x['first_wrong_counts'].items())
            signed.append(f"| {arm} | {variant} | {x['under']} / {x['equal']} / {x['over']} | {x['valid_nodes']} | {x['negative_reachable']} | {hist} ({x['first_wrong_vectors']}) |")
        q=next(x for x in r['paired'] if x['arm']==arm and x['probe']=='development' and x['group']=='pooled')
        pairs.append(f"| {arm} | {q['format_gained']} / {q['format_lost']} | {q['node_gained']} / {q['node_lost']} |")
    n='\n'
    return f'''# Stage 16: cache-versus-held-out error audit

Completed local CPU retained-output analysis, 31 October 2026. No new fitting
or model inference. All five arms and both fixed update-600 variants are retained.

**The small-cache fit does not generalize, and formatting alone cannot account
for its held-out failure.** Valid held-out vectors still have low semantic node
accuracy. Gold-prefix teacher forcing also leaves digit prediction weak. Source
and one-hop errors show that failure is not confined to longer paths. These
observations do not identify a unique causal mechanism or establish graph
equivalence, learning acceleration or cost recovery.

## Frozen scope and provenance

Plan `{sha_file(PUBLIC/'plan.json')}` and the
[frozen protocol](FROZEN_PROTOCOL.md) precede this new analysis. All 52 stage 15
evidence copies, its manifest and inherited scientific sources are hash-bound.
The SAME selected cache16 and full ID-development128 examples are compared
between small16 and the IMPORTED historical512 reference. No checkpoint or
example is selected by its new error profile. These are inspected development
data from one historical cohort, with two correlated tasks per graph.

Implementation disclosure: the inherited deterministic generator also recreates
assessment/OOD rows in memory before only ID rows are selected. The protocol's
blanket "No OOD data" wording is too broad; see the preserved-source
[regeneration note](../../ERROR_AUDIT_V16_REGENERATION_NOTE.md). No assessment/OOD
prediction, scoring, model input or selection occurs; no new forward pass runs.

See [full results](results.json), [100 group/cell rows](cells.csv) and
[50 paired comparison rows](paired.csv). Each arm/variant/probe includes pooled
and all four task/size cells; pooled summaries below do not replace them.

## Cache fitting versus held-out semantic and token errors

Cache entries are small16 / historical512 on the same 16 examples:

| Arm | Exact answers | Teacher-forced digit accuracy | Generated node-micro accuracy |
| --- | --- | --- | --- |
{n.join(cache)}

Every cache node, source and observed hop bin is correct in small16; digit and
other teacher token accuracy are both 100%. Every held-out exact score remains
0/128 for BOTH variants. Held-out details:

| Arm | Variant | Valid vectors | Node-micro, all outputs | Node-micro, valid vectors only (correct/nodes) | Teacher digit | Teacher other |
| --- | --- | --- | --- | --- | --- | --- |
{n.join(dev)}

Invalid vectors score zero at every node in unconditional rates. Conditional
rates describe different model-selected valid subsets, so their differences do
not isolate a formatting effect. Teacher scores are TOKEN-MICRO aggregates of
imported counters, supplying gold earlier answer tokens; they are not free-
generation node accuracies. Digit-bearing GPT-2 tokens can include punctuation.
Stage 15's reported token-cell means and node-example means use different
denominators; both those original metrics and this audit's explicit micro/macro
counters are preserved. No neural logit or teacher forward pass is rerun here.

## Source and hop errors

Unconditional correct nodes / all nodes in each role, including invalid outputs
as wrong. Hop bins use UNWEIGHTED paths even for weighted Dijkstra answers:

| Arm | Variant | Source | One hop | Multi hop | Unreachable |
| --- | --- | --- | --- | --- | --- |
{n.join(hops)}

These ID graphs have no unreachable nodes. Their 0/0 cells are unobserved, with
null accuracy in the results, not zero-percent performance. Source correctness
is a simple prerequisite (its answer should be zero), but the diagnosis does not
infer a unique source-index, tokenizer, optimization or serialization defect.
Valid-vector-conditional role denominators and each task/size cell are in the CSV.

## Signed and first-error distributions

Under/equal/over compare generated values to truth only among VALID vectors.
Negative/reachable counts cannot be interpreted as an unconditional rate.
First wrong is an OUTPUT NODE INDEX, not graph distance; its histogram excludes
exact and invalid vectors. All counts, including zero categories, remain in JSON.

| Arm | Variant | Under / equal / over | Valid nodes | Negative on reachable | First-wrong index:count (valid erroneous vectors) |
| --- | --- | --- | --- | --- | --- |
{n.join(signed)}

Many valid erroneous vectors already fail at output index zero. This is a
descriptive ordering fact; it does not demonstrate that later failures are
caused by the first error, exposure bias or an inability to propagate messages.
Teacher-forced digit failures also occur with correct earlier answers supplied.

## Paired held-out changes

Small16 minus historical512, on the SAME 128 examples/1152 nodes per arm.
Gained means only small16 is correct/valid; lost means only reference512 is.
Both/neither categories and identical cache comparisons remain in the full CSV.

| Arm | Format gained / lost | Node correct gained / lost |
| --- | --- | --- |
{n.join(pairs)}

This is a descriptive paired endpoint comparison, not an independent-cohort
test or causal separation of cache content, diversity, repetition and lengths.
Do not turn overlapping arm ranges into graph no-effect/equivalence evidence.

## Verification, cost and next decision

A separate invocation rescores all 1440 retained outputs with a different
strict decimal parser and all-pairs hop algorithm. All 100 diagnostic rows and
50 paired rows, serialized CSVs and input/source identities reproduce exactly.
Aggregation remains shared code; teacher counters and the stage 15 checkpoint
replay are imported evidence, not independent neural verification. New model
forwards, generations, optimizer updates, PPT/language runs and OOD predictions:
ZERO. Analysis/replay wall times are retained in the append-only log; original
fit/replay and initialization costs remain in stages 13–15.

Run `python -m error_audit_v16.run verify` through `scripts/run_logged.py` to
reproduce from distributable evidence without private weights, tokenizer cache,
raw caches or network. [Usage](../../ERROR_AUDIT_V16_USAGE.md) distinguishes
this semantic replay from the original neural replay. No remote/paid job,
official language test, email, push or external publication is performed.

Preserve all endpoints and gates. The next economical DEVELOPMENT proposal is
an intermediate 64-example cache (16 graphs per size, both tasks), at the same
600 updates and from the same five original pre-adaptation starts, using the
same outcome-blind nested ranking. Retain both 16 and 512 references on identical
evaluation examples. Change cache size alone; do not also extend training,
alter formatting/objective or choose a checkpoint. This probes a fitting/diversity
tradeoff; it does not promise a remedy or isolate repetition causally. Freeze its
data, budget and criteria separately BEFORE any fit. No such fit runs here.

The original categorical/random-rule readiness failure, separate readiness
bridge and OOD/full-language gates remain unchanged. A real graph-specific
confirmation still requires eligible synthetic starts, fresh paired rule/data/
topology cohorts, identical full language/adaptation procedures, scratch/grid/
graph/both prefix controls, ID-only selection, frozen OOD criteria/censoring/
multiplicity and total costs. This audit does not satisfy those prerequisites.
'''


if __name__=='__main__':
    path=PUBLIC/'REVIEW.md';path.write_text(render(),encoding='utf-8',newline='\n')
    print('Rendered stage 16 verified review; all five arms and both variants retained')
