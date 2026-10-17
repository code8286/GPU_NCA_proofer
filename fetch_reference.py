"""Fetch or verify the pinned upstream reference without overwriting evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

PINNED_COMMIT = "bdd1c71e04eb2e5a2c91481faf278e95846b4431"
REPOSITORY = "https://github.com/danihyunlee/nca-pre-pretraining"
FILES = ["README.md", "LICENSE", "utils/nca.py", "utils/tokenizers.py", "utils/models.py",
         "utils/training_args.py", "src/nca_ppt.py", "src/language_train.py", "src/openwebtext_pt.py",
         "scripts/prepretraining/nca_prepretraining.sh", "scripts/pretraining/owt_ft.sh",
         "scripts/pretraining/ft_codeparrot.sh"]


def read(url):
    request = urllib.request.Request(url, headers={"User-Agent": "NCA-methodology-audit",
                                                 "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT, help="Full immutable 40-character commit SHA")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "sources/nca-pre-pretraining-reference")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("Use a full lowercase commit SHA; moving branch names are not reproducible")
    root = args.out
    manifest_path = root / "reference.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["commit"] != args.commit:
            parser.error("Existing reference has a different revision; choose a new --out directory")
        for name, entry in manifest["files"].items():
            if hashlib.sha256((root / args.commit / name).read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Reference checksum mismatch: {name}")
        print(f"PASS: existing pinned reference verified ({len(manifest['files'])} files); no network needed.")
        return
    if root.exists() and any(root.iterdir()):
        parser.error("Output directory must be empty or a verified existing snapshot")
    commit_data = json.loads(read(f"https://api.github.com/repos/danihyunlee/nca-pre-pretraining/commits/{args.commit}"))
    if commit_data["sha"] != args.commit:
        raise ValueError("Upstream revision mismatch")
    # Fetch everything before writing, so network failures leave no partial snapshot.
    contents = {name: read(f"https://raw.githubusercontent.com/danihyunlee/nca-pre-pretraining/{args.commit}/{name}")
                for name in FILES}
    manifest = {"repository": f"{REPOSITORY}/tree/{args.commit}",
                "paper": "https://arxiv.org/html/2603.10055v1", "commit": args.commit,
                "commit_date": commit_data["commit"]["committer"]["date"], "files": {}}
    for name, content in contents.items():
        target = root / args.commit / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest["files"][name] = {
            "url": f"https://raw.githubusercontent.com/danihyunlee/nca-pre-pretraining/{args.commit}/{name}",
            "sha256": hashlib.sha256(content).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved {len(contents)} pinned reference files at revision {args.commit}.")


if __name__ == "__main__":
    main()
