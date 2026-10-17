"""Build the self-contained v2 Kaggle pre-pre-training transfer notebook."""
import argparse
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = "kaggle_nca_transfer_v2.ipynb"
SOURCES = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/"transfer_v2").glob("*.py"))]+[
    "configs/transfer_v2.json", "configs/transfer_v2_smoke.json", "tests/test_transfer_v2.py",
    "requirements.txt", "requirements-v2.txt", "docs/TRANSFER_V2_PROTOCOL.md",
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
    "sources/nca-pre-pretraining-reference/bdd1c71e04eb2e5a2c91481faf278e95846b4431/LICENSE"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(True), "outputs": [], "execution_count": None}


def notebook():
    folders = sorted({str(Path(p).parent).replace("\\", "/") for p in SOURCES if str(Path(p).parent) != "."})
    cells = [md("""# Does NCA pre-pre-training accelerate subsequent language learning?

**Version 2: independent stage budgets.** First validate synthetic prediction on
unseen NCA rules, then compare the same language-training procedure from random,
grid-NCA and shuffled-NCA initializations. A positive result is not guaranteed.
The graph extension is deferred until the grid reference works.

Before Run All: select one available GPU accelerator and enable Internet.
The notebook uses one GPU, downloads pinned WikiText train/validation data and
GPT-2 tokenizer assets, and leaves the official confirmation test split untouched.
Default budgets are 1000 PPT updates and **2000 equal language updates** per arm.
Three paired seeds. A small RoPE/RMSNorm/SwiGLU transformer, categorical 10-state
NCA, complexity-filtered rules and ten burn-in steps replace pilot 1's toy setup.

If held-out synthetic learning fails the declared gate, language training is
skipped and diagnostic outputs are still exported. Passing the gate is only
permission to test transfer, not evidence that it works. The final ZIP contains
checkpoints, source, configuration, rule/data identities, logs and verification.
"""), code("""import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
ROOT = Path('/kaggle/working') / f'nca_transfer_v2_{RUN_ID}'
ROOT.mkdir(parents=True, exist_ok=False)
os.chdir(ROOT)
os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'tokenizer_cache')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'plot_cache')
print('Run:', RUN_ID, 'Workspace:', ROOT)
"""+f"\nfor directory in {folders!r}:\n    Path(directory).mkdir(parents=True, exist_ok=True)\n"),
        md("## Embedded, reviewable sources\nThese files match the locally checked v2 source. No Git checkout is required.")]
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

run('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_transfer_v2.py', '-v')
"""), md("""## Tiny GPU execution fixture
This checks CUDA execution, stage checkpoints, common language controls and
verification. It deliberately uses a tiny byte fixture and does not supply a
scientific result or count toward any development model's training.
"""), code("""fixture = Path('fixture')
fixture.mkdir()
(fixture / 'train.txt').write_text('Offline execution fixture; no research inference.\\n' * 512)
(fixture / 'validation.txt').write_text('Distinct validation fixture for checks.\\n' * 128)
run('-m', 'transfer_v2.run', 'all', '--config', 'configs/transfer_v2_smoke.json',
    '--data', 'data/smoke', '--out', 'runs/smoke', '--fixture-dir', str(fixture), '--device', 'cuda')
"""), md("""## Real development run
Data, budgets, selection and gate settings are fixed in `configs/transfer_v2.json`.
The CLI prepares filtered rule-disjoint NCA data and pinned WikiText/GPT-2 tokens,
prints an estimate measured on disposable models on your actual GPU, runs PPT,
then conditionally runs the matched language stage and verifies checkpoints.

The synthetic gate requires at least 5% held-out patch-loss improvement and a
0.01-nat advantage over both fitted unigram/copy controls for every grid model seed.
It is never relaxed automatically. Test data are neither downloaded nor evaluated.
"""), code("""run('-m', 'transfer_v2.run', 'all', '--config', 'configs/transfer_v2.json',
    '--data', 'data/development', '--out', 'runs/development', '--device', 'cuda')
"""), md("""## Inspect the outcome
If status is `complete`, report language-stage learning acceleration first and
cumulative PPT+language costs separately. The target is each paired scratch
model's final validation loss. Unreached targets remain censored; all-seed
speedups are withheld when a seed misses the target. Three seeds are exploratory.

If status is `stopped_at_synthetic_gate`, this run contains diagnostics and PPT
checkpoints; it has not measured downstream learning acceleration.
"""), code("""import json
from IPython.display import SVG, display

OUT = Path('runs/development')
status = json.loads((OUT / 'status.json').read_text())
print('RUN STATUS:', status['state'])
display(json.loads((OUT / 'ppt_gate.json').read_text()))
result = json.loads((OUT / 'report.json').read_text())
display(result['aggregate'])
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

archive = Path('/kaggle/working') / f'nca_transfer_v2_results_{RUN_ID}.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
    for path in ROOT.rglob('*'):
        if path.is_file() and not {'__pycache__', 'plot_cache'}.intersection(path.relative_to(ROOT).parts):
            z.write(path, path.relative_to(ROOT))
print('WORKFLOW FINISHED:', status['state'])
print('Download:', archive, f'({archive.stat().st_size / 1e6:.1f} MB)')
display(FileLink(str(archive)))
""")]
    for i, cell in enumerate(cells):
        cell["id"] = f"nca-transfer-v2-{i:03d}"
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
            raise SystemExit("Stale v2 notebook: run python build_transfer_notebook.py")
        print(f"PASS: v2 notebook parity; {len(SOURCES)} embedded sources.")
        return
    target.write_text(rendered, encoding="utf-8", newline="\n")
    with zipfile.ZipFile(ROOT/"nca_transfer_v2_starter.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES+[NAME, "build_transfer_notebook.py"]:
            z.write(ROOT/name, name)
    print(f"Wrote {NAME}; {len(SOURCES)} embedded files and matching starter ZIP.")


if __name__ == "__main__":
    main()
