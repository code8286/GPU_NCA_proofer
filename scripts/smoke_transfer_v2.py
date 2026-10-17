"""Offline end-to-end checks of both complete and gate-stopped v2 workflows."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def run(command, work):
    result = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=False)
    print(result.stdout.replace(str(work), "<temporary-v2-smoke>"), end="")
    print(result.stderr.replace(str(work), "<temporary-v2-smoke>"), end="", file=sys.stderr)
    if result.returncode:
        raise RuntimeError(f"V2 integration command failed: exit {result.returncode}")


def main():
    # Keep artifacts (including failures) within the writable research workspace.
    # Some managed Windows sessions cannot create nested OS-temp directories.
    work = ROOT/"tmp"/("transfer_v2_check_"+uuid.uuid4().hex)
    work.mkdir(parents=True, exist_ok=False)
    if not work.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Check workspace escaped repository")
    fixture = work/"fixture"
    fixture.mkdir()
    (fixture/"train.txt").write_text("Offline training fixture; no research inference.\n"*512)
    (fixture/"validation.txt").write_text("Distinct validation fixture for execution checks.\n"*128)
    (fixture/"test.txt").write_text("Explicit confirmation-path execution fixture.\n"*128)
    for name in ("complete", "blocked", "confirmation_fixture"):
        blocked = name == "blocked"
        c = json.loads((ROOT/"configs/transfer_v2_smoke.json").read_text())
        if blocked:
            c["gate"].update(enforce=True, minimum_relative_improvement=2., baseline_margin_nats=100.)
        if name == "confirmation_fixture":
            c["text"]["evaluate_test"] = True
        config = work/f"{name}.json"
        config.write_text(json.dumps(c))
        data, out = work/f"data_{name}", work/f"run_{name}"
        run(["-m", "transfer_v2.run", "all", "--config", str(config), "--data", str(data),
             "--out", str(out), "--fixture-dir", str(fixture), "--device", "cpu"], work)
        status = json.loads((out/"status.json").read_text())
        expected = "stopped_at_synthetic_gate" if blocked else "complete"
        if status["state"] != expected:
            raise RuntimeError(f"Unexpected status: {status}")
        verification = json.loads((out/"verification.json").read_text())
        if verification["selected_stage_checkpoints_verified"] != (2 if blocked else 5):
            raise RuntimeError("Wrong checkpoint verification count")
        if not (out/"ppt_diagnostics.png").exists():
            raise RuntimeError("Missing synthetic diagnostics figure")
        if blocked:
            if list(out.glob("*/language_final.pt")):
                raise RuntimeError("Language training ran despite failed synthetic gate")
        else:
            summaries = [json.loads(p.read_text()) for p in out.glob("*/language_summary.json")]
            if len(summaries) != 3 or {s["language_updates"] for s in summaries} != {6}:
                raise RuntimeError("Unequal language-stage updates")
            scratch = next(s for s in summaries if s["arm"] == "scratch")
            if scratch["ppt_updates"] != 0 or scratch["ppt_tokens"] != 0:
                raise RuntimeError("Scratch received an unintended PPT/text head start")
            if not (out/"learning_curves.png").exists():
                raise RuntimeError("Missing transfer curves")
            if any((s["test"] is not None) != c["text"]["evaluate_test"] for s in summaries):
                raise RuntimeError("Final test policy was not respected")
        if (data/"text_test.npy").exists() != c["text"]["evaluate_test"]:
            raise RuntimeError("Reserved test fixture was unexpectedly read")
        print(f"PASS: {name} workflow, checkpoint reloads, matched language controls, declared test policy.")
    print("V2 offline fixtures validate execution only. No GPU or transfer benefit was measured.")
    print(f"Execution artifacts retained under {work.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
