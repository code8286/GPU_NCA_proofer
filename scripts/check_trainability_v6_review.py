"""Check stage 6 copies, paired budgets, nested scores and preserved old sources."""
from pathlib import Path
import hashlib
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import sha_file, effective_tokens
from readiness_v5.gate import decision
from trainability_v6.plan import read, verify_plan
from trainability_v6.run import verify_training
from scripts.check_release import selected_files


def main():
    private = ROOT/'runs/trainability_v6_run1'
    public = ROOT/'docs/results/trainability_v6_run1'
    plan = verify_plan(private)
    training = verify_training(private, plan)
    c = plan['config']
    for name, entry in read(public/'PROVENANCE.json')['files'].items():
        assert sha_file(public/name) == entry['sha256'] == sha_file(private/name), name
    assert plan == read(public/'plan.json')
    old5 = read(ROOT/'docs/results/readiness_v5_run1/plan.json')
    assert plan['v4_source_sha256'] == old5['v4_source_sha256']
    for name, expected in old5['diagnostic_source_sha256'].items():
        assert plan['source_sha256']['readiness_v5/'+name] == expected
    ids = read(private/'data/split_identities.json')
    assert len(set(sum(ids.values(), []))) == sum(c['splits'].values())
    for seed in c['seeds']:
        pair = [r for r in training if r['seed'] == seed]
        order = hashlib.sha256()
        rng = np.random.default_rng(seed+101)
        for _ in range(c['ppt']['steps']*c['accumulate']):
            order.update(rng.integers(0, c['splits']['train'], size=c['batch']).astype('<i8').tobytes())
        for row in pair:
            assert row['train_order_sha256'] == order.hexdigest()
            assert row['input_tokens'] == c['ppt']['steps']*effective_tokens(c)
            assert row['prediction_targets'] == c['ppt']['steps']*c['batch']*c['accumulate']*42
            assert row['initial_core_sha256'] == pair[0]['initial_core_sha256']
            assert row['initial_head_sha256'] == pair[0]['initial_head_sha256']
    result = read(public/'results.json')
    assert sha_file(public/'results.json') == read(public/'results_receipt.json')['results_sha256']
    assert sha_file(public/'results.json') == read(public/'verification.json')['results_sha256']
    assert not result['language_allowed']
    cells = {(r['seed'], r['checkpoint_arm'], r['data_arm']):r for r in result['cells']}
    assert len(cells) == len(result['cells']) == 18
    for row in cells.values():
        values = row['per_initial_condition']
        assert len(values) == 128 and all(r['patch_targets'] == 28 for r in values)
        assert math.isclose(sum(r['patch_nll'] for r in values)/128, row['patch_nll'], abs_tol=1e-12)
    for row in result['decisions']:
        seed = row['seed']
        assert row['coherent_nll'] == cells[seed, 'coherent', 'coherent']['patch_nll']
        assert row['shuffled_nll'] == cells[seed, 'coherent', 'prefix']['patch_nll']
        assert row['gate'] == decision(row['coherent_nll'], row['shuffled_nll'], result['frequency_nll'], c)
    assert result['trainability_passed'] == (result['table_control']['gate']['passed'] and not result['frequency_negative_gate']['passed'] and all(r['gate']['passed'] for r in result['decisions']))
    prefix_passes = sum(decision(cells[s, 'prefix', 'coherent']['patch_nll'], cells[s, 'prefix', 'prefix']['patch_nll'], result['frequency_nll'], c)['passed'] for s in c['seeds'])
    assert prefix_passes == 3  # Descriptive control finding, not a revised primary gate.
    print('PASS: 21 retained copies, v4/v5 source preservation, six reconstructed batch orders/budgets, phase-disjoint splits and all 18 nested assessment cells/decisions.')
    print('Descriptive control check: all three prefix-trained models also pass the absolute margins; this is not a causal training-mechanism test.')
    # Publication copies include existing mathematical sources and records.
    # Comparing bytes does not assert a new Lean check or TeX compilation.
    inventory = read(ROOT/'Prod/PUBLICATION_MANIFEST.json')['files']
    for group, suffix in (('proofs/lean/', '.lean'), ('docs/theory/', '.tex')):
        names = [n for n in selected_files(ROOT) if n.startswith(group)]
        required = {p.relative_to(ROOT).as_posix() for p in (ROOT/group).rglob('*'+suffix)
                    if '.lake' not in p.parts}
        assert required and required <= set(names)
        for name in names:
            assert sha_file(ROOT/name) == sha_file(ROOT/'Prod'/name) == inventory[name]
        print(f'PASS: {len(names)} {group} source/evidence copies included with exact root/Prod parity; no new compiler execution.')


if __name__ == '__main__':
    main()
