"""Export a diagnostic validation report without rewriting historical records."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed, independently verified run directory")
    parser.add_argument("--report", type=Path, required=True, help="New Markdown destination; existing files are refused")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; choose a new path to preserve earlier evidence")
    with (args.run / "summary.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    verification = json.loads((args.run / "verification.json").read_text(encoding="utf-8"))
    environment = json.loads((args.run / "environment.json").read_text(encoding="utf-8"))
    expected = len(records)
    if not expected or verification["checkpoints_verified"] != expected:
        raise ValueError("Summary count and checkpoint verification count differ")
    checked = {(row["arm"], row["seed"]): row for row in verification["records"]}
    if len(checked) != expected:
        raise ValueError("Duplicate checkpoint verification identities")
    for row in records:
        entry = checked.get((row["arm"], int(row["seed"])))
        if entry is None or not entry["checkpoint_sha256_valid"] or not 0 <= entry["absolute_error"] <= 1e-4:
            raise ValueError("Checkpoint consistency requirement failed")
    lines = ["# Diagnostic validation export", "",
             f"Exported UTC: {datetime.now(timezone.utc).isoformat()}",
             f"Recorded training device: {environment['device']}; verification device: {verification['device']}.",
             f"Completed summary/checkpoint records: {expected}.",
             f"Experiment SHA-256: `{environment['source_sha256']}`.", "",
             "This exports existing measurements; it does not rerun training or verification.",
             "Historical fixed-total-budget results do not establish equal-language-stage transfer.", "",
             "| Arm | Seed | Final validation NLL | Test NLL |", "| --- | ---: | ---: | ---: |"]
    for row in records:
        lines.append(f"| {row['arm']} | {row['seed']} | {float(row['final_val_nll']):.6f} | {float(row['test_nll']):.6f} |")
    lines += ["", "Test loss is nats per byte. Consult report.json for censored target outcomes.",
              "This report does not claim that unit tests or new GPU measurements were executed."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Exported {expected} verified summary records to {args.report}")


if __name__ == "__main__":
    main()
