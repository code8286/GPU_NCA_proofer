"""Run a check and append its complete captured output to logs/checks.jsonl."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cwd", type=Path, default=ROOT,
                        help="Check working directory, e.g. Prod for standalone validation")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --")
    recorded = command.copy()
    if command[0] == "python":
        command[0] = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    started = datetime.now(timezone.utc).isoformat()
    v2_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((ROOT / 'transfer_v2').glob('*.py'))}
    v3_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((ROOT / 'transfer_v3').glob('*.py'))}
    v4_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((ROOT / 'transfer_v4').glob('*.py'))}
    readiness_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((ROOT / 'readiness_v5').glob('*.py'))}
    trainability_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT / 'trainability_v6').glob('*.py'))}
    interaction_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT / 'interaction_v7').glob('*.py'))}
    weighting_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((ROOT / 'loss_weight_v8').glob('*.py'))}
    fit_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT / 'fit_v9').glob('*.py'))}
    diversity_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((ROOT / 'diversity_v10').glob('*.py'))}
    confirmation_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / 'confirmation_v11').glob('*.py'))}
    ood_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT / 'ood_v12').glob('*.py'))}
    adaptation_sources = {str(p.relative_to(ROOT)).replace(chr(92), "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "adaptation_v13").glob("*.py"))}
    id_audit_sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((ROOT / "id_audit_v14").glob("*.py"))}
    overfit_sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT / "overfit_v15").glob("*.py"))}
    error_audit_sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "error_audit_v16").glob("*.py"))}
    cache_sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT / "cache_v17").glob("*.py"))}
    scientific_sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "scientific_transfer").glob("*"))
                          if p.is_file() and p.suffix in {".py", ".json", ".md", ".txt"}}
    tick = time.perf_counter()
    accel_sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT / "accel_v18").glob("*.py"))}
    fit_audit_sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((ROOT / "fit_audit_v19").glob("*.py"))}
    try:
        result = subprocess.run(command, cwd=args.cwd, env=env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", check=False)
        status, stdout, stderr = result.returncode, result.stdout, result.stderr
    except OSError as exc:
        status, stdout, stderr = 127, "", f"{type(exc).__name__}: {exc}\n"
    # Keep public transcripts portable. Do not run commands exposing secrets.
    def portable(value):
        value = value.replace(str(ROOT), "<repo>").replace(ROOT.as_posix(), "<repo>")
        value = value.replace(str(ROOT).replace("\\", "\\\\"), "<repo>")
        value = value.replace(sys.executable, "python")
        value = value.replace(str(Path.home()).replace("\\", "\\\\"), "<user-home>")
        value = value.replace(str(Path.home()), "<user-home>")
        return re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s'\"]+", "<user-home>", value)
    event = {
        "started_utc": started, "label": args.label,
        "cwd": portable(str(args.cwd.resolve())),
        "command": [portable(part) for part in recorded],
        "exit_code": status, "duration_seconds": round(time.perf_counter() - tick, 6),
        "python": sys.version.split()[0],
        "experiment_sha256": hashlib.sha256((ROOT / "experiment.py").read_bytes()).hexdigest(),
        "transfer_v2_sources_sha256": v2_sources,
        "transfer_v3_sources_sha256": v3_sources,
        "transfer_v4_sources_sha256": v4_sources,
        "readiness_v5_sources_sha256": readiness_sources,
        "trainability_v6_sources_sha256": trainability_sources,
        "interaction_v7_sources_sha256": interaction_sources,
        "loss_weight_v8_sources_sha256": weighting_sources,
        "fit_v9_sources_sha256": fit_sources,
        "diversity_v10_sources_sha256": diversity_sources,
        "confirmation_v11_sources_sha256": confirmation_sources,
        "ood_v12_sources_sha256": ood_sources,
        "adaptation_v13_sources_sha256": adaptation_sources,
        "id_audit_v14_sources_sha256": id_audit_sources,
        "overfit_v15_sources_sha256": overfit_sources,
        "error_audit_v16_sources_sha256": error_audit_sources,
        "cache_v17_sources_sha256": cache_sources,
        "accel_v18_sources_sha256": accel_sources,
        "fit_audit_v19_sources_sha256": fit_audit_sources,
        "scientific_transfer_sources_sha256": scientific_sources,
        "stdout": portable(stdout), "stderr": portable(stderr),
    }
    log = ROOT / "logs/checks.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, allow_nan=False) + "\n")
    print(event["stdout"], end="", flush=True)
    print(event["stderr"], end="", file=sys.stderr, flush=True)
    print(f"Recorded {args.label}: exit {status}, {event['duration_seconds']:.3f}s")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
