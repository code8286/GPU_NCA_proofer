"""Private checkpoint receipts, actual trace and public score/cost accounting."""
from datetime import datetime,timedelta
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from accel_v18.run import verify_plan
from adaptation_v13.training import encode,learning_rate
from ood_v12.plan import read,tokenizer
from ood_v12.data import read_jsonl
from scripts.report_accel_v18 import RUN,PUBLIC
from scripts.reproduce_accel_v18 import reproduce
from scripts.check_release import selected_files
from transfer_v4.common import sha_file


def main():
    p=verify_plan(RUN);r=read(PUBLIC/'results.json');v=read(PUBLIC/'verification.json');c=p['config']
    assert v['passed'] and v['checkpoints']==35 and v['predictions']==4480 and v['results_sha256']==sha_file(PUBLIC/'results.json')
    copies=read(PUBLIC/'PROVENANCE.json')['files']
    assert len(copies)==60 and all(sha_file(RUN/name)==sha_file(PUBLIC/name)==h for name,h in copies.items())
    assert r['new_scientific_updates']==3000 and r['new_language_or_ppt_updates']==r['new_ood_predictions']==0
    assert not r['confirmation_allowed'] and not p['ood_allowed'] and p['original_random_rule_readiness']=='still_failed'
    rows=read_jsonl(RUN/'data/train.jsonl');tok=tokenizer();encoded={v['id']:encode(v,tok,1024) for v in rows}
    for m in r['models']:
        trace=read_jsonl(RUN/f'models/{m["arm"]}/updates.jsonl');counts=dict.fromkeys(p['expected_counts_per_arm'],0)
        assert len(trace)==600
        for step,update in enumerate(trace,1):
            batch=[encoded[key] for key in update['sampled_ids']]
            actual={'input_tokens':sum(len(v['input']) for v in batch),
                    'supervised_targets':sum(sum(t!=-100 for t in v['target']) for v in batch),
                    'padded_positions':len(batch)*max(len(v['input']) for v in batch)}
            assert update['step']==step and update['lr']==learning_rate(step,c)
            assert math.isfinite(update['loss']) and math.isfinite(update['gradient_norm'])
            for key in counts:
                counts[key]+=actual[key];assert update[key]==actual[key] and update['cumulative_counts'][key]==counts[key]
        assert counts==m['counts']==p['expected_counts_per_arm']
        curve=m['curve'];assert [e['step'] for e in curve]==c['evaluation_updates']
        for a,b in zip(curve[:-1],curve[1:]):
            assert b['training_seconds']>a['training_seconds'] and b['evaluation_seconds']>a['evaluation_seconds'] and b['elapsed_seconds']>a['elapsed_seconds']
        for e in curve:
            assert e['elapsed_seconds']>=e['training_seconds']+e['evaluation_seconds']
            assert e['counts']==(trace[e['step']-1]['cumulative_counts'] if e['step'] else dict.fromkeys(counts,0))
    reproduce()
    events=[json.loads(line) for line in (ROOT/'logs/checks.jsonl').read_text(encoding='utf-8').splitlines()]
    labels=['accel-v18-freeze-before-first-update','accel-v18-five-matched-id-fits','accel-v18-separate-checkpoint-replay','accel-v18-export-verified-evidence']
    chain=[next(e for e in events if e['label']==label and e['exit_code']==0) for label in labels]
    for a,b in zip(chain[:-1],chain[1:]):assert datetime.fromisoformat(a['started_utc'])+timedelta(seconds=a['duration_seconds'])<datetime.fromisoformat(b['started_utc'])
    expected={k:h for k,h in p['source_sha256'].items() if k.startswith('accel_v18/')}
    assert all(e['accel_v18_sources_sha256']==expected for e in chain)
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(read(PUBLIC/'fit_started.json')['started_utc'])
    names=selected_files(ROOT);assert not any(n.startswith(('runs/','data/','Kaggle Outputs/','.uv-cache/')) for n in names)
    print('PASS:60 exact copies,35/4480 replay receipts, full numerical token/order/LR/optimizer identities, baseline/cost/target/censoring/140-row accounting and source/chronology/privacy. No eligibility gate changed.')


if __name__=='__main__':main()
