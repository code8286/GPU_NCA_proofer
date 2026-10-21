"""Build the portable Kaggle notebook directly from the checked project sources."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(True),
            "outputs": [], "execution_count": None}


cells = [markdown("""# Does graph-NCA pre-training accelerate a small language model?

This notebook executes **historical matched-total-budget pilot 1**.
Its first returned GPU run did not demonstrate a graph benefit. Scratch receives
1000 text updates while synthetic arms receive 100 synthetic + 900 text updates.
It is not the corrected equal-language-stage transfer protocol. Read the embedded
methodology audit before interpreting results. A positive result is not guaranteed.
It contains all source code; no repository checkout is required.

**Before Run All:** select one available GPU accelerator and enable Internet for
the public Tiny Shakespeare download. Two T4s, if allocated, are not both used.
Keep the notebook private unless you choose to publish it. Save a version with
outputs to preserve the checkpoints. No paid API or rental is used.

The pilot trains the same 1,857,024-parameter causal byte transformer across five
arms and three paired seeds: target text only, separate text, grid NCA, graph NCA,
and shuffled graph trajectories. All arms receive 4,096,000 training tokens per
seed. The declared target is 2.5 validation nats/byte; unreached targets remain
unresolved. Pipeline timing includes synthetic generation and pre-training.
This is a binary graph-topology pilot, not a reproduction of the larger-scale
[2026 NCA paper](https://arxiv.org/abs/2603.10055).

Runtime must be measured on the GPU you receive. Deterministic fp32 favors
auditability over peak throughput. The final archive includes model checkpoints,
metrics, dataset checksums, source, environment, and verification results.
"""), code("""import os
from pathlib import Path
import sys
import subprocess
from datetime import datetime, timezone

# Set before any CUDA operation, including imports in subprocesses.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path('/kaggle/working/gnn_nca_pilot')
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
Path('configs').mkdir(exist_ok=True)
Path('tests').mkdir(exist_ok=True)
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
print('Project:', ROOT, 'Run:', RUN_ID)

import torch
assert torch.cuda.is_available(), 'Enable a GPU accelerator in Kaggle notebook settings, then rerun.'
print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))
print('One GPU is used. No distributed training is required.')
""")]

sources = ["experiment.py", "verify.py", "plot_results.py", "tests/test_experiment.py", "configs/pilot.json",
           "configs/smoke.json", "configs/cpu_check.json", "requirements.txt", "README.md",
           "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md",
           "RESEARCH_CONTEXT.md", "METHODOLOGY_AUDIT.md", "VALIDATION.md"]
cells.append(markdown("## Embedded, reviewable experiment sources\nThese cells write exactly the project files tested locally."))
for name in sources:
    cells.append(code(f"%%writefile {name}\n"+(ROOT/name).read_text(encoding="utf-8")))

cells += [markdown("""## Verify the scientific controls
These checks test causality, initialization, graph degree, convolution equivalence,
patch tokenization, shuffling, trajectory boundaries, and censored target handling.
"""), code("""def run(*args):
    subprocess.run([sys.executable, *args], check=True)

run('-m', 'unittest', 'discover', '-s', 'tests', '-v')
"""), markdown("""## GPU smoke run
Six updates per arm exercise CUDA, caching, stage transfer, reporting, and saved
checkpoint verification. Smoke losses are execution checks, not research results.
"""), code("""SMOKE_DATA = f'data/smoke_{RUN_ID}'
SMOKE_OUT = f'runs/smoke_{RUN_ID}'
run('experiment.py', 'all', '--config', 'configs/smoke.json',
    '--data', SMOKE_DATA, '--out', SMOKE_OUT, '--device', 'cuda')
run('verify.py', '--data', SMOKE_DATA, '--out', SMOKE_OUT, '--device', 'cuda')
"""), markdown("""## Three-seed pilot
Configuration and target are fixed before looking at pilot outcomes. All arm
comparisons use the same fixed validation and held-out test subsets. There is no
generator selection against test loss. Complete preparation is charged per run
even though data are cached across seeds. See README.md for exact timing scope.
The run uses a fresh output folder; failed runs never overwrite old checkpoints.
"""), code("""PILOT_DATA = f'data/pilot_{RUN_ID}'
PILOT_OUT = f'runs/pilot_{RUN_ID}'
run('experiment.py', 'all', '--config', 'configs/pilot.json',
    '--data', PILOT_DATA, '--out', PILOT_OUT,
    '--text', str(Path(SMOKE_DATA) / 'input.txt'), '--device', 'cuda')
run('verify.py', '--data', PILOT_DATA, '--out', PILOT_OUT, '--device', 'cuda')
run('plot_results.py', '--out', PILOT_OUT)
"""), markdown("""## Inspect the result
A pipeline speedup above one favors graph NCA. Report censored seeds and compare
against scratch, grid NCA, and shuffled trajectories. Three seeds are exploratory;
they cannot prove a general benefit. Byte perplexity is not comparable to BPE
perplexity. This pilot does not test language reasoning or train a GNN rule.
"""), code("""import json
import csv
from IPython.display import SVG, display

with (Path(PILOT_OUT) / 'summary.csv').open(encoding='utf-8') as handle:
    display(list(csv.DictReader(handle)))
display(SVG(filename=str(Path(PILOT_OUT) / 'curves.svg')))
result = json.loads((Path(PILOT_OUT) / 'report.json').read_text())
display(result['comparisons'])
display(result['aggregate'])
print('Never substitute a made-up speedup for an unresolved target.')
"""), markdown("""## Download the verifiable result bundle
The archive includes the exact input tokens and checkpoints, so held-out losses
can be independently recomputed. Save the notebook version with outputs too.
"""), code("""import zipfile
from IPython.display import FileLink

archive = Path('/kaggle/working') / f'gnn_nca_results_{RUN_ID}.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
    for file in ROOT.rglob('*'):
        if file.is_file() and '__pycache__' not in file.parts:
            z.write(file, file.relative_to(ROOT))
print('Saved:', archive, f'({archive.stat().st_size / 1e6:.1f} MB)')
display(FileLink(str(archive)))
""")]

notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"},
                         "kaggle": {"isInternetEnabled": True, "language": "python", "sourceType": "notebook"}},
            "cells": cells}
for i, cell in enumerate(notebook["cells"]):
    cell["id"] = f"gnca-{i:03d}"
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check generated notebook without writing")
    args = parser.parse_args()
    target = ROOT / "kaggle_gnn_nca_pilot.ipynb"
    rendered = json.dumps(notebook, indent=1)
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
            parser.exit(1, "FAIL: stale notebook; run python build_notebook.py\n")
        print(f"PASS: notebook syntax/source generation parity ({len(sources)} embedded files).")
        return
    target.write_text(rendered, encoding="utf-8", newline="\n")
    # The local starter archive is a complete source snapshot; the notebook
    # embeds only the files needed by its direct scientific execution cells.
    from scripts.check_release import MANIFEST, inventory
    snapshot = inventory(ROOT)
    with zipfile.ZipFile(ROOT / "gnn_nca_starter.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, expected_hash in snapshot['files'].items():
            content = (ROOT/name).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError(f'Publication input changed while packaging: {name}; rebuild the snapshot')
            z.writestr(name, content)
        z.writestr(MANIFEST, json.dumps(snapshot, indent=2) + "\n")
    print(f"Wrote {target.name}: {len(cells)} cells; embedded sources match local files.")


if __name__ == "__main__":
    main()
