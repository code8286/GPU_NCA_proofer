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
    tick = time.perf_counter()
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
