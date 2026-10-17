"""Offline five-arm CPU integration check using temporary, non-research text."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    temporary_root = ROOT / "tmp/checks"
    temporary_root.mkdir(parents=True, exist_ok=True)
    if not temporary_root.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Temporary check directory must stay inside the repository")
    with tempfile.TemporaryDirectory(prefix="gnn_nca_smoke_", dir=temporary_root) as directory:
        work = Path(directory)
        text = work / "fixture.txt"
        text.write_text("Offline smoke fixture; not a research corpus.\n" * 1024,
                        encoding="utf-8")
        data, out = work / "data", work / "run"
        commands = [
            ["experiment.py", "all", "--config", "configs/smoke.json", "--text", str(text),
             "--data", str(data), "--out", str(out), "--device", "cpu"],
            ["verify.py", "--data", str(data), "--out", str(out), "--device", "cpu"],
            ["plot_results.py", "--out", str(out)],
        ]
        for command in commands:
            completed = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                                       capture_output=True, text=True, encoding="utf-8",
                                       errors="replace", check=False)
            print(completed.stdout.replace(str(work), "<temporary-smoke>"), end="")
            print(completed.stderr.replace(str(work), "<temporary-smoke>"),
                  end="", file=sys.stderr)
            if completed.returncode:
                return completed.returncode
        verification = json.loads((out / "verification.json").read_text(encoding="utf-8"))
        if verification["checkpoints_verified"] != 5:
            raise RuntimeError("Expected all five arms to be verified")
        for filename in ("summary.csv", "report.json", "curves.svg", "curves.png"):
            if not (out / filename).is_file() or (out / filename).stat().st_size == 0:
                raise RuntimeError(f"Missing smoke output: {filename}")
        print("PASS: five arms, checkpoint verification, reports and plots; offline CPU fixture.")
        print("Temporary artifacts removed after this check. No scientific inference from fixture loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
