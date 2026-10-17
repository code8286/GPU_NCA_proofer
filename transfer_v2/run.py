"""CLI: python -m transfer_v2.run all --config configs/transfer_v2.json ..."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from .common import ARMS, REFERENCE_COMMIT, dump, load_config, sha_file, source_hashes, sync
from .calibrate import calibrate
from .data import load_data, prepare
from .report import report
from .training import train_language, train_ppt
from .verify import verify


def setup(device):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA GPU is available")


def initialize_run(c, root, data_root, device):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    dump(root/"environment.json", {
        "schema_version": 2, "config": c, "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(),
        "numpy": np.__version__,
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cuda": torch.version.cuda, "tf32": False, "reference_commit": REFERENCE_COMMIT,
        "source_sha256": source_hashes(), "data_manifest_sha256": sha_file(Path(data_root)/"manifest.json"),
        "graph_treatment": "Deferred until grid transfer is established",
        "timing_scope": "Language seconds start after model/head/optimizer construction and include validation. Cumulative adds text preparation, relevant synthetic preparation/cache writes, head/optimizer construction, shared manifest work and measured PPT stage. Excludes package startup, cloud queue, optional disposable calibration and final language checkpoint/verification/plot/archive work."})
    dump(root/"status.json", {"state": "ppt_pending", "purpose": c["purpose"]})


def check_identity(root, c, data_root):
    env = json.loads((Path(root)/"environment.json").read_text())
    if env["config"] != c or env["data_manifest_sha256"] != sha_file(Path(data_root)/"manifest.json"):
        raise ValueError("Run/config/data mismatch")
    if env["source_sha256"] != source_hashes():
        raise ValueError("Source changed during this staged run; create a fresh run")


def ppt_stage(c, root, arrays, manifest_hash, device):
    summaries = []
    dump(Path(root)/"status.json", {"state": "ppt_running", "purpose": c["purpose"]})
    for i, seed in enumerate(c["seeds"]):
        order = ("grid_nca", "shuffled_nca") if i % 2 == 0 else ("shuffled_nca", "grid_nca")
        for arm in order:
            summaries.append(train_ppt(c, arm, seed, arrays, manifest_hash, root, device))
    failed = [s["seed"] for s in summaries if s["arm"] == "grid_nca" and not s["gate"]["passed"]]
    allowed = not failed or (c["purpose"] == "execution_check" and not c["gate"]["enforce"])
    status = {"state": "ppt_ready" if allowed else "stopped_at_synthetic_gate", "purpose": c["purpose"],
              "grid_seeds_failing_gate": failed, "language_allowed": allowed,
              "execution_only_bypass": bool(failed and allowed),
              "note": "Passing this gate only permits testing transfer; it does not establish acceleration."}
    dump(Path(root)/"status.json", status)
    dump(Path(root)/"ppt_gate.json", {"status": status, "summaries": summaries})
    return allowed


def language_stage(c, root, arrays, manifest, manifest_hash, device):
    status = json.loads((Path(root)/"status.json").read_text())
    if status["state"] != "ppt_ready" or not status["language_allowed"]:
        raise ValueError("Held-out synthetic diagnostics must permit language training first")
    dump(Path(root)/"status.json", {"state": "language_running", "purpose": c["purpose"]})
    for i, seed in enumerate(c["seeds"]):
        order = ARMS[i % len(ARMS):]+ARMS[:i % len(ARMS)]
        summaries = []
        for arm in order:
            summaries.append(train_language(c, arm, seed, arrays, manifest, manifest_hash, root, device))
        for key in ("core_initial_sha256", "language_head_initial_sha256", "language_train_order_sha256", "language_schedule_sha256"):
            if len({s[key] for s in summaries}) != 1:
                raise RuntimeError(f"Language controls failed: {key}")
    dump(Path(root)/"status.json", {"state": "complete", "purpose": c["purpose"],
                                  "scientific_claim": "Inspect paired language-stage curves and censored reference crossings; completion is not success."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "calibrate", "ppt", "language", "report", "verify", "all"))
    p.add_argument("--config", default="configs/transfer_v2.json")
    p.add_argument("--data", default="data/transfer_v2")
    p.add_argument("--out", default="runs/transfer_v2")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--fixture-dir", help="Execution checks only: train.txt and validation.txt")
    a = p.parse_args()
    c = load_config(a.config)
    device = "cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu" if a.device == "auto" else a.device
    setup(device)
    if a.command in ("prepare", "all"):
        prepare(c, a.data, device, a.fixture_dir)
    if a.command == "all":
        arrays, manifest = load_data(a.data, c)
        calibrate(c, arrays, manifest, device, a.data)
    if a.command == "calibrate":
        arrays, manifest = load_data(a.data, c)
        calibrate(c, arrays, manifest, device, a.data)
        return
    if a.command in ("ppt", "all"):
        arrays, manifest = load_data(a.data, c)
        initialize_run(c, a.out, a.data, device)
        allowed = ppt_stage(c, a.out, arrays, sha_file(Path(a.data)/"manifest.json"), device)
        if not allowed:
            report(a.out, c)
            verify(a.out, a.data, device)
            print("STOPPED AT SYNTHETIC GATE: diagnostics saved; language benchmark not launched.", flush=True)
            return
    if a.command in ("language", "all"):
        check_identity(a.out, c, a.data)
        arrays, manifest = load_data(a.data, c)
        language_stage(c, a.out, arrays, manifest, sha_file(Path(a.data)/"manifest.json"), device)
    if a.command in ("language", "report", "all"):
        report(a.out, c)
    if a.command in ("language", "verify", "all"):
        verify(a.out, a.data, device)


if __name__ == "__main__":
    main()
