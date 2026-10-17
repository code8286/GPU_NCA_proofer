"""Copy selected research-root files into Prod, or check drift without writing."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from check_release import MANIFEST, check, inventory, safe_file, selected_files, write_inventory

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if (ROOT / MANIFEST).exists():
        parser.error("Already a standalone publication. Use check_release.py --refresh; do not nest Prod.")
    destination = ROOT / "Prod"
    if destination.is_symlink() or destination.resolve().parent != ROOT.resolve():
        raise ValueError("Prod must be a direct, non-symlink child of the research root")
    names = selected_files(ROOT)
    if args.check:
        check(destination)
        if inventory(ROOT) != inventory(destination):
            raise ValueError("Prod differs from canonical root files; run scripts/sync_prod.py")
        print("PASS: Prod matches canonical research-root publication files.")
        return
    # Never delete unknown or stale files; ask the maintainer to inspect them.
    extras = []
    if destination.exists():
        for path in destination.rglob("*"):
            relative = path.relative_to(destination)
            if path.is_file() and relative.as_posix() not in set(names) | {MANIFEST}:
                if "__pycache__" not in relative.parts and ".git" not in relative.parts:
                    extras.append(relative.as_posix())
    if extras:
        raise ValueError(f"Unmanaged files in Prod require inspection: {sorted(extras)}")
    destination.mkdir(exist_ok=True)
    for name in names:
        target = safe_file(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(ROOT, name), target)
    write_inventory(destination)
    check(destination)
    print(f"Synced {len(names)} selected files to Prod. No GitHub action performed.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
