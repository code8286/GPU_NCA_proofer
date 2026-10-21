"""Build the self-contained v4 Kaggle pre-pre-training transfer notebook."""
import argparse
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = "kaggle_nca_transfer_v4.ipynb"
SOURCES = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/"transfer_v4").glob("*.py"))]+[
    "configs/historical_sources_pre_v4.json", "configs/transfer_v4.json", "configs/transfer_v4_smoke.json", "tests/test_transfer_v4.py",
    "requirements.txt", "requirements-v2.txt", "docs/TRANSFER_V4_PROTOCOL.md",
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
    "sources/nca-pre-pretraining-reference/bdd1c71e04eb2e5a2c91481faf278e95846b4431/LICENSE"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(True), "outputs": [], "execution_count": None}


def notebook():
    folders = sorted({str(Path(p).parent).replace("\\", "/") for p in SOURCES if str(Path(p).parent) != "."})
    cells = [md("""# Does NCA pre-pre-training accelerate subsequent language learning?

**Version 4: exact-prefix controls and fresh paired replication.**
The objective is faster language learning through pre-pre-training. Seven
initializations enter the SAME language stage: scratch, grid NCA, graph NCA,
the two original shuffled controls, and two stronger prefix-matched controls.
Those new controls preserve initial conditioning, supervised-prefix patch
frequencies and marker formatting while scrambling later patch order.

Select one GPU accelerator and enable Internet before Run All. Six fresh
cohorts use 1000 PPT updates (zero for scratch) plus 2000 equal language updates.
The transformer, rule filter, topology and serializer retain v3 settings.
The official test stays untouched. This is a prospective validation study,
not confirmation on a new language task. No positive outcome is guaranteed.

The co-primary comparisons are graph versus grid and graph versus the new
prefix-matched graph control, at the SAME paired scratch-endpoint loss.
All six cohorts stay in the analysis, including ties and missed targets.
Two one-sided sign tests use alpha .025 each; six strict wins are required.
The protocol/config/source identities are archived before benchmark training.

Allow roughly **7-8 hours on a comparable T4, plus verification/export overhead**:
this extrapolates the returned v3 run, not a new timing measurement. Disposable
calibration on your GPU prints an estimate. The final ZIP may be several GB
because it retains all stage checkpoints and six real starting-weight snapshots.
If a structured synthetic gate fails, the language suite is skipped and the
diagnostics still export. Passing a gate or finishing does not prove acceleration.
"""), code("""import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
ROOT = Path('/kaggle/working') / f'nca_transfer_v4_{RUN_ID}'
ROOT.mkdir(parents=True, exist_ok=False)
os.chdir(ROOT)
os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'tokenizer_cache')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'plot_cache')
print('Run:', RUN_ID, 'Workspace:', ROOT)
"""+f"\nfor directory in {folders!r}:\n    Path(directory).mkdir(parents=True, exist_ok=True)\n"),
        md("## Embedded, reviewable sources\nThese files match the locally checked v4 source. No Git checkout is required.")]
    for name in SOURCES:
        cells.append(code(f"%%writefile {name}\n"+(ROOT/name).read_text(encoding="utf-8")))
    cells += [md("""## Dependencies and controls
Kaggle supplies its CUDA PyTorch build; this cell retains it and installs only
the small tokenizer/Parquet reader if missing. A GPU is required for the actual
development run. CPU tests are engineering checks, not acceleration evidence.
"""), code("""import importlib.metadata
from packaging.specifiers import SpecifierSet
missing = []
for package, bounds in [('tiktoken', '>=0.7,<1'), ('pyarrow', '>=17,<27')]:
    try:
        suitable = importlib.metadata.version(package) in SpecifierSet(bounds)
    except importlib.metadata.PackageNotFoundError:
        suitable = False
    if not suitable:
        missing.append(package + bounds)
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
import torch
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator and restart the notebook runtime.'
print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))

def run(*arguments):
    subprocess.run([sys.executable, *arguments], check=True)

run('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_transfer_v4.py', '-v')
"""), md("""## Tiny GPU execution fixture
This checks CUDA execution, stage checkpoints, common language controls and
verification. It deliberately uses a tiny byte fixture and does not supply a
scientific result or count toward any development model's training.
"""), code("""fixture = Path('fixture')
fixture.mkdir()
(fixture / 'train.txt').write_text('Offline execution fixture; no research inference.\\n' * 512)
(fixture / 'validation.txt').write_text('Distinct validation fixture for checks.\\n' * 128)
run('-m', 'transfer_v4.run', 'all', '--config', 'configs/transfer_v4_smoke.json',
    '--data', 'data/smoke', '--out', 'runs/smoke', '--fixture-dir', str(fixture), '--device', 'cuda')
"""), md("""## Real development run
Data, budgets, selection and gate settings are fixed in `configs/transfer_v4.json`.
The CLI prepares filtered rule-disjoint NCA data, exact-prefix controls and pinned WikiText/GPT-2 tokens,
prints an estimate measured on disposable models on your actual GPU, runs PPT,
then conditionally runs the matched language stage and verifies checkpoints.

The synthetic gate requires at least 5% held-out patch-loss improvement and a
0.01-nat advantage over both fitted unigram/copy controls for every GRID AND GRAPH seed.
It is never relaxed automatically. Test data are neither downloaded nor evaluated.
"""), code("""run('-m', 'transfer_v4.run', 'all', '--config', 'configs/transfer_v4.json',
    '--data', 'data/development', '--out', 'runs/development', '--device', 'cuda')
"""), md("""## Inspect the outcome
If status is `complete`, read `primary_analysis` first: graph versus grid and
graph versus prefix_graph at each cohort's scratch endpoint. Ratios use the
same target; any missing/zero-token crossing withholds the aggregate. All six
cohorts, ties and censored rows remain visible. Native shuffled controls and
reference-arm endpoints are secondary. Language tokens/time and cumulative
cost are separate measurements; token gains need not repay PPT overhead.

If status is `stopped_at_synthetic_gate`, outputs contain diagnostics, not a
downstream acceleration result. The frozen protocol and plan are in the ZIP.
"""), code("""import json
from IPython.display import SVG, display

OUT = Path('runs/development')
status = json.loads((OUT / 'status.json').read_text())
print('RUN STATUS:', status['state'])
display(json.loads((OUT / 'ppt_gate.json').read_text()))
result = json.loads((OUT / 'report.json').read_text())
display(result['primary_analysis'])
display(result['aggregate'])
display(result['treatment_comparisons'])
display(json.loads((OUT / 'verification.json').read_text()))
display(SVG(filename=str(OUT / 'ppt_diagnostics.svg')))
if (OUT / 'learning_curves.svg').exists():
    display(SVG(filename=str(OUT / 'learning_curves.svg')))
"""), md("""## Download the verifiable result
Save a Kaggle notebook version with outputs, and download this ZIP. A stopped
synthetic gate is a completed diagnostic workflow, not an error or a successful
transfer claim. It is still useful to inspect and share that output archive.
"""), code("""import zipfile
from IPython.display import FileLink

archive = Path('/kaggle/working') / f'nca_transfer_v4_results_{RUN_ID}.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
    for path in ROOT.rglob('*'):
        if path.is_file() and not {'__pycache__', 'plot_cache'}.intersection(path.relative_to(ROOT).parts):
            z.write(path, path.relative_to(ROOT))
print('WORKFLOW FINISHED:', status['state'])
print('Download:', archive, f'({archive.stat().st_size / 1e6:.1f} MB)')
display(FileLink(str(archive)))
""")]
    for i, cell in enumerate(cells):
        cell["id"] = f"nca-transfer-v4-{i:03d}"
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"},
                         "kaggle": {"isInternetEnabled": True, "language": "python", "sourceType": "notebook"}},
            "cells": cells}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    rendered = json.dumps(notebook(), indent=1)
    target = ROOT/NAME
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Stale v4 notebook: run python build_mechanism_notebook.py")
        print(f"PASS: v4 notebook parity; {len(SOURCES)} embedded sources.")
        return
    target.write_text(rendered, encoding="utf-8", newline="\n")
    with zipfile.ZipFile(ROOT/"nca_transfer_v4_starter.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES+[NAME, "build_mechanism_notebook.py"]:
            z.write(ROOT/name, name)
    print(f"Wrote {NAME}; {len(SOURCES)} embedded files and matching starter ZIP.")


if __name__ == "__main__":
    main()
