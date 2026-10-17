"""Check first-party Markdown links and an optional starter archive inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from check_release import MANIFEST, ROOT, safe_file, selected_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter", type=Path, help="Also validate a generated starter ZIP")
    args = parser.parse_args()
    links = 0
    for name in selected_files(ROOT):
        if not name.endswith(".md") or name.startswith("sources/"):
            continue  # Upstream snapshot is deliberately partial and unchanged.
        source = ROOT / name
        text = source.read_text(encoding="utf-8")
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        for link in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", text):
            if "://" in link or link.startswith(("#", "mailto:")):
                continue
            target = link.split("#", 1)[0]
            resolved = (source.parent / target).resolve()
            if not resolved.is_relative_to(ROOT.resolve()) or not resolved.exists():
                raise ValueError(f"Broken or escaping documentation link: {name}: {link}")
            links += 1
    print(f"PASS: {links} first-party local Markdown links.")
    if args.starter:
        with zipfile.ZipFile(args.starter) as archive:
            if archive.testzip() is not None:
                raise ValueError("Starter archive CRC failure")
            manifest = json.loads(archive.read(MANIFEST))
            expected = set(manifest["files"]) | {MANIFEST}
            if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
                raise ValueError("Starter inventory/archive member mismatch")
            for name, expected_hash in manifest["files"].items():
                safe_file(ROOT, name)
                if hashlib.sha256(archive.read(name)).hexdigest() != expected_hash:
                    raise ValueError(f"Starter checksum mismatch: {name}")
        print(f"PASS: starter ZIP CRC and all {len(manifest['files'])} inventoried content hashes.")


if __name__ == "__main__":
    main()
