"""Check a publication inventory, notebook parity and pinned reference hashes."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "PUBLICATION_MANIFEST.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_file(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe publication path: {name}")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Publication path escapes root: {name}")
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != root.parent):
        raise ValueError(f"Symlinks are not publication inputs: {name}")
    return target


def selected_files(root):
    policy = json.loads((root / "publication.json").read_text(encoding="utf-8"))
    names = set(policy["files"])
    for directory in policy["directories"]:
        folder = safe_file(root, directory)
        if not folder.is_dir():
            raise ValueError(f"Missing publication directory: {directory}")
        for path in folder.rglob("*"):
            relative = path.relative_to(root)
            if {"__pycache__", ".lake"}.intersection(relative.parts) or path.suffix in (
                ".pyc", ".pyo", ".olean", ".ilean"
            ):
                continue
            if path.is_file():
                names.add(relative.as_posix())
    for name in names:
        path = safe_file(root, name)
        if not path.is_file():
            raise ValueError(f"Missing publication file: {name}")
        forbidden = {"Prod", "data", "runs", "Kaggle Outputs", ".venv", ".git", ".env"}
        if forbidden.intersection(PurePosixPath(name).parts) or path.suffix.lower() in {
            ".zip", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".pdf", ".pem", ".key"
        } or path.name.startswith(".env"):
            raise ValueError(f"Non-distributable artifact: {name}")
        if path.stat().st_size > 5_000_000:
            raise ValueError(f"Publication file exceeds 5 MB: {name}")
    return sorted(names)


def inventory(root):
    return {"schema_version": 1, "hash_algorithm": "sha256",
            "note": "Inventory excludes this manifest and local ignored artifacts.",
            "files": {name: sha(safe_file(root, name)) for name in selected_files(root)}}


def write_inventory(root):
    (root / MANIFEST).write_text(json.dumps(inventory(root), indent=2) + "\n",
                                 encoding="utf-8", newline="\n")


def check(root):
    recorded = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    actual = inventory(root)
    if recorded != actual:
        changed = sorted(name for name in set(recorded["files"]) | set(actual["files"])
                         if recorded["files"].get(name) != actual["files"].get(name))
        raise ValueError(f"Stale publication inventory: {changed}")
    for name in actual["files"]:
        path = root / name
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=name)
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    embedded = 0
    for notebook_name in (n for n in actual["files"] if n.endswith(".ipynb")):
        notebook = json.loads((root / notebook_name).read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                raise ValueError("Published notebook contains execution outputs")
            source = "".join(cell["source"])
            if source.startswith("%%writefile "):
                header, content = source.split("\n", 1)
                name = header[len("%%writefile "):]
                if content != safe_file(root, name).read_text(encoding="utf-8"):
                    raise ValueError(f"Stale embedded notebook source: {name}")
                if name.endswith(".py"):
                    ast.parse(content, filename=name)
                embedded += 1
            else:
                ast.parse(source)
    reference_root = root / "sources/nca-pre-pretraining-reference"
    reference = json.loads((reference_root / "reference.json").read_text(encoding="utf-8"))
    for name, entry in reference["files"].items():
        if sha(safe_file(reference_root, reference["commit"] + "/" + name)) != entry["sha256"]:
            raise ValueError(f"Modified pinned reference source: {name}")
    provenance = json.loads((root / "docs/results/PROVENANCE.json").read_text(encoding="utf-8"))
    for name, entry in provenance["files"].items():
        if sha(safe_file(root, name)) != entry["published_sha256"]:
            raise ValueError(f"Modified historical evidence copy: {name}")
    print(f"PASS: {len(actual['files'])} inventoried files; {embedded} embedded sources; "
          f"{len(reference['files'])} pinned reference files; historical evidence integrity.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--refresh", action="store_true", help="Refresh inventory after changes and tests")
    args = parser.parse_args()
    if args.refresh:
        write_inventory(args.root)
    check(args.root)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
