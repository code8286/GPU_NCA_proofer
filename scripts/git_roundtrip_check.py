"""Verify publication hashes survive Git staging and a fresh checkout."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from check_release import MANIFEST, ROOT, check, safe_file, selected_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.root.resolve()
    check(source)
    parent = ROOT / "tmp/checks"
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Temporary Git check must stay inside the repository")
    with tempfile.TemporaryDirectory(prefix="git_roundtrip_", dir=parent) as directory:
        staging = Path(directory) / "staging"
        checkout = Path(directory) / "checkout"
        staging.mkdir()
        checkout.mkdir()
        for name in selected_files(source) + [MANIFEST]:
            target = safe_file(staging, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(source, name), target)
        ignored = [".env", "data/private.txt", "runs/model.pt", ".venv/marker.txt",
                   "__pycache__/entry.pyc", "Kaggle Outputs/private.zip", "tmp/private.txt",
                   "proofs/lean/.lake/probe.txt", "proofs/lean/GraphNCA.olean",
                   "proofs/lean/GraphNCA.ilean"]
        for name in ignored:
            target = safe_file(staging, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("Disposable ignore-rule probe\n", encoding="utf-8")
        commands = [
            ["git", "init", "--quiet"],
            ["git", "-c", "core.autocrlf=true", "add", "--all"],
            ["git", "-c", "core.autocrlf=true", "checkout-index", "--all",
             "--prefix=" + checkout.as_posix() + "/"],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=staging, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", check=False)
            if result.returncode:
                raise RuntimeError(result.stderr.replace(str(directory), "<temporary-git>"))
        tracked = subprocess.run(["git", "ls-files"], cwd=staging, capture_output=True,
                                 text=True, encoding="utf-8", check=True).stdout.splitlines()
        if set(tracked) != set(selected_files(source)) | {MANIFEST}:
            raise RuntimeError("Git tracked unexpected files or omitted publication inputs")
        check(checkout)
        print("PASS: Git staging/checkout preserves inventory, historical source and evidence hashes.")
        print("No commit, remote or push created; temporary repository removed.")
        print(f"PASS: {len(ignored)} disposable secret/cache/artifact probes excluded by .gitignore.")


if __name__ == "__main__":
    main()
