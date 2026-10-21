"""Execute only the trusted canonical v4 notebook payload in an isolated folder."""
from pathlib import Path
import json
import os
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_mechanism_notebook import NAME, SOURCES


def main():
    work = ROOT/'tmp'/('v4_notebook_check_'+uuid.uuid4().hex)
    work.mkdir(parents=True, exist_ok=False)
    seen = set()
    notebook = json.loads((ROOT/NAME).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        content = ''.join(cell['source'])
        if cell['cell_type'] != 'code' or not content.startswith('%%writefile '):
            continue
        header, body = content.split('\n', 1)
        name = header[len('%%writefile '):]
        if name not in SOURCES or name in seen or body != (ROOT/name).read_text(encoding='utf-8'):
            raise ValueError('Unexpected, duplicate or stale embedded source')
        seen.add(name)
        path = work/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding='utf-8', newline='\n')
    if seen != set(SOURCES):
        raise ValueError('Missing embedded notebook sources')
    with zipfile.ZipFile(ROOT/'nca_transfer_v4_starter.zip') as archive:
        names = SOURCES+[NAME, 'build_mechanism_notebook.py']
        if archive.testzip() or set(archive.namelist()) != set(names) or len(archive.namelist()) != len(names):
            raise ValueError('Starter ZIP CRC or inventory failure')
        for name in names:
            if archive.read(name) != (ROOT/name).read_bytes():
                raise ValueError(f'Starter ZIP content differs: {name}')
    fixture = work/'fixture'
    fixture.mkdir()
    (fixture/'train.txt').write_text('Standalone notebook execution fixture.\n'*512)
    (fixture/'validation.txt').write_text('Separate validation fixture.\n'*128)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-m', 'transfer_v4.run', 'all', '--config', 'configs/transfer_v4_smoke.json',
                             '--data', 'data/smoke', '--out', 'runs/smoke', '--fixture-dir', 'fixture', '--device', 'cpu'],
                            cwd=work, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(result.stdout.replace(str(work), '<isolated-notebook>'), end='')
    print(result.stderr.replace(str(work), '<isolated-notebook>'), end='', file=sys.stderr)
    if result.returncode:
        raise RuntimeError('Isolated embedded notebook workflow failed')
    verification = json.loads((work/'runs/smoke/verification.json').read_text())
    if verification['selected_stage_checkpoints_verified'] != 13 or verification['saved_initializations_verified'] != 1:
        raise ValueError('Wrong standalone checkpoint count')
    if (work/'data/smoke/text_test.npy').exists():
        raise ValueError('Unexpected test fixture access')
    print(f'PASS: {len(seen)} canonical notebook payload files, starter ZIP bytes/CRC, isolated 13-checkpoint CPU workflow.')
    print(f'Artifacts retained at {work.relative_to(ROOT)}; execution only, not GPU evidence.')


if __name__ == '__main__':
    main()
