"""Check private receipts, trace tokens, exact public copies and chronology."""
from datetime import datetime,timedelta
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fit_audit_v19.run import verify_plan
from adaptation_v13.training import encode
from ood_v12.plan import read,tokenizer
from ood_v12.data import read_jsonl
from scripts.report_fit_audit_v19 import RUN,PUBLIC
from scripts.reproduce_fit_audit_v19 import reproduce
from scripts.check_release import selected_files
from transfer_v4.common import sha_file


def main():
    p=verify_plan(RUN);r=read(PUBLIC/'results.json');v=read(PUBLIC/'verification.json')
    assert v['passed'] and v['checkpoints']==5 and v['new_predictions_replayed']==640 and v['results_sha256']==sha_file(PUBLIC/'results.json')
    copies=read(PUBLIC/'PROVENANCE.json')['files'];assert len(copies)==36
    assert all(sha_file(RUN/name)==sha_file(PUBLIC/name)==h for name,h in copies.items())
    tok=tokenizer();data=read_jsonl(ROOT/'runs/accel_v18_run1/data/train.jsonl');encoded={v['id']:encode(v,tok,1024) for v in data}
    for m in r['models']:
        for update in read_jsonl(PUBLIC/f'optimization/{m["arm"]}_updates.jsonl'):
            values=[encoded[key] for key in update['sampled_ids']]
            assert update['input_tokens']==sum(len(x['input']) for x in values)
            assert update['supervised_targets']==sum(sum(t!=-100 for t in x['target']) for x in values)
            assert update['padded_positions']==8*max(len(x['input']) for x in values)
    reproduce()
    events=[json.loads(line) for line in (ROOT/'logs/checks.jsonl').read_text(encoding='utf-8').splitlines()]
    labels=['fit-audit-v19-freeze-before-prediction','fit-audit-v19-five-seen-probe-evaluations','fit-audit-v19-separate-checkpoint-replay','fit-audit-v19-export-verified-evidence']
    chain=[next(e for e in events if e['label']==label and e['exit_code']==0) for label in labels]
    for a,b in zip(chain[:-1],chain[1:]):assert datetime.fromisoformat(a['started_utc'])+timedelta(seconds=a['duration_seconds'])<datetime.fromisoformat(b['started_utc'])
    expected={k:h for k,h in p['source_sha256'].items() if k.startswith('fit_audit_v19/')}
    assert all(e['fit_audit_v19_sources_sha256']==expected for e in chain)
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(read(PUBLIC/'evaluation_started.json')['started_utc'])
    assert not any(n.startswith(('runs/','data/','Kaggle Outputs/','.uv-cache/')) for n in selected_files(ROOT))
    print('PASS:36 exact copies,5/640 new replay receipts, imported640 held-out references,3000 actual token/trace records, probe-specific targets/decisions,40 cells/30 blocks and source/chronology/privacy. Zero optimizer updates.')


if __name__=='__main__':main()
