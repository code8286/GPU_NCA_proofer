"""Separately frozen first-transition/neighbor intervention; no new training."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import dump, sha_file
from transfer_v4.nca import encode_frames
from interaction_v7.plan import read, verify_plan
from interaction_v7.run import verify_training, load_model
from interaction_v7.data import RULES, unpack, pack, step, canonical

PROTOCOL = ROOT/'docs/INTERACTION_V7_FIRST_TRANSITION.md'


def freeze(parent, out):
    plan = verify_plan(parent)
    if (parent/'results.json').exists():
        raise ValueError('Freeze this diagnostic before primary assessment')
    out.mkdir(parents=True, exist_ok=False)
    ids = read(parent/'data/split_identities.json')
    original = np.array(ids['initial_ids']['assessment'])
    changed = unpack(original)
    changed[:, 0, 3] ^= 1
    changed_ids = pack(changed)
    forbidden = set(ids['joint_orbit_ids']['train']+ids['joint_orbit_ids']['development'])
    eligible = np.array([int(k) not in forbidden for k in canonical(changed_ids)])
    hashes = {}
    for rule in RULES:
        native = np.load(parent/f'data/{rule}_assessment.npy', allow_pickle=False)[:, :8]
        frames = np.stack([changed, step(changed, rule)], axis=1).reshape(len(changed), 2, 16)
        intervention = encode_frames(torch.tensor(frames), 4, 2).numpy()[:, :8].astype(np.uint16)
        np.testing.assert_array_equal(native[:, 1], intervention[:, 1])
        np.testing.assert_array_equal(native[:, 7] ^ intervention[:, 7], np.full(len(native), 1 if rule == 'spatial' else 0))
        for name, values in ((f'{rule}_native', native), (f'{rule}_changed', intervention)):
            path = out/f'{name}.npy'
            np.save(path, values, allow_pickle=False)
            hashes[path.name] = sha_file(path)
    dump(out/'eligibility.json', {'eligible': eligible.tolist(), 'pairs': int(eligible.sum()), 'original_queries': len(original)})
    hashes['eligibility.json'] = sha_file(out/'eligibility.json')
    (out/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    frozen = {'parent_plan_sha256': sha_file(parent/'plan.json'), 'source_sha256': sha_file(Path(__file__)),
              'protocol_sha256': sha_file(PROTOCOL), 'query_files_sha256': hashes,
              'scope': 'Supplementary development plan frozen during training before assessment; primary criteria unchanged'}
    dump(out/'plan.json', frozen)
    dump(out/'plan_receipt.json', {'plan_sha256': sha_file(out/'plan.json')})
    print(f"Frozen first-transition diagnostic: {len(original)} originals, {eligible.sum()} eligible pairs; plan={sha_file(out/'plan.json')}")


@torch.no_grad()
def score(model, queries):
    losses, correct, predictions = [], [], []
    model.eval()
    for start in range(0, len(queries), 16):
        z = torch.tensor(queries[start:start+16].astype(np.int64))
        logits = model(z[:, :7])[:, -1]
        loss = F.cross_entropy(logits, z[:, 7], reduction='none')
        pred = logits.argmax(-1)
        losses.extend(loss.double().tolist())
        correct.extend((pred == z[:, 7]).tolist())
        predictions.extend(pred.tolist())
    return np.array(losses), np.array(correct), np.array(predictions)


def evaluate(parent, out):
    plan = verify_plan(parent)
    models = verify_training(parent, plan)
    if not read(parent/'verification.json')['passed']:
        raise ValueError('Verified primary assessment required')
    frozen = read(out/'plan.json')
    if sha_file(out/'plan.json') != read(out/'plan_receipt.json')['plan_sha256'] or frozen['parent_plan_sha256'] != sha_file(parent/'plan.json') or frozen['source_sha256'] != sha_file(Path(__file__)) or frozen['protocol_sha256'] != sha_file(PROTOCOL) or frozen['protocol_sha256'] != sha_file(out/'FROZEN_PROTOCOL.md'):
        raise ValueError('Supplementary plan/source/protocol changed')
    for name, checksum in frozen['query_files_sha256'].items():
        if sha_file(out/name) != checksum:
            raise ValueError('Query changed: '+name)
    if (out/'results.json').exists():
        raise FileExistsError('Preserve first supplementary result')
    mask = np.array(read(out/'eligibility.json')['eligible'], dtype=bool)
    cells, baselines = [], []
    for rule in RULES:
        native = np.load(out/f'{rule}_native.npy', allow_pickle=False)
        changed = np.load(out/f'{rule}_changed.npy', allow_pickle=False)
        training = np.load(parent/f'data/{rule}_train.npy', allow_pickle=False)
        table = np.ones((16, 16), dtype=float)
        np.add.at(table, (training[:, 1], training[:, 7]), 1)
        table /= table.sum(1, keepdims=True)
        local = float(-np.log(table[native[:, 1], native[:, 7]]).mean())
        baselines.append({'rule': rule, 'first_patch_table_nll': local, 'uniform_nll': float(np.log(16))})
        for seed in plan['config']['seeds']:
            for arm in ('initial', 'coherent', 'prefix'):
                model, saved = load_model(parent, plan['config'], seed, None if arm == 'initial' else f'{rule}_{arm}')
                loss, correct, pred = score(model, native)
                other_loss, other_correct, other_pred = score(model, changed[mask])
                expected = pred[mask] ^ (1 if rule == 'spatial' else 0)
                bound_hash = plan['input_files_sha256'][f'initialization_{seed}.pt'] if arm == 'initial' else next(r['checkpoint_sha256'] for r in models if r['seed'] == seed and r['arm'] == f'{rule}_{arm}')
                cells.append({'rule': rule, 'seed': seed, 'arm': arm, 'checkpoint_sha256': bound_hash,
                              'native_nll': float(loss.mean()), 'native_accuracy': float(correct.mean()),
                              'changed_nll': float(other_loss.mean()), 'changed_accuracy': float(other_correct.mean()),
                              'paired_both_correct': float((correct[mask] & other_correct).mean()),
                              'argmax_changes_as_required': float((expected == other_pred).mean()),
                              'native_per_query_nll': loss.tolist(), 'native_per_query_correct': correct.tolist(),
                              'changed_per_query_nll': other_loss.tolist(), 'changed_per_query_correct': other_correct.tolist(),
                              'stricter_first_transition_passed': bool(min(local, np.log(16))-loss.mean() >= .01)})
                del model, saved
    passed = all(r['stricter_first_transition_passed'] for r in cells if r['rule'] == 'spatial' and r['arm'] == 'coherent')
    result = {'scope': frozen['scope'], 'plan_sha256': sha_file(out/'plan.json'), 'parent_results_sha256': sha_file(parent/'results.json'),
              'original_queries': len(mask), 'eligible_pairs': int(mask.sum()), 'baselines': baselines, 'cells': cells,
              'stricter_first_transition_passed': passed, 'language_allowed': False}
    dump(out/'results.json', result)
    dump(out/'results_receipt.json', {'results_sha256': sha_file(out/'results.json')})
    print({'stricter_first_transition_passed': passed, 'spatial_coherent': [r for r in cells if r['rule'] == 'spatial' and r['arm'] == 'coherent'] }.__repr__()[:2200])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'score'))
    p.add_argument('--parent', type=Path, default=ROOT/'runs/interaction_v7_run1')
    p.add_argument('--out', type=Path, default=ROOT/'runs/interaction_v7_run1/first_transition')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    (freeze if a.command == 'freeze' else evaluate)(a.parent, a.out)


if __name__ == '__main__':
    main()
