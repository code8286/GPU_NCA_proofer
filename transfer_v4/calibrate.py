"""Disposable on-device timing calibration; never changes a research model."""
import hashlib
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .common import ARMS, dump, sync
from .model import LM, language_model
from .training import evaluate, optimizer_for, update


def calibrate(c, arrays, manifest, device, output):
    values = {}
    for name, synthetic in (("ppt", True), ("language", False)):
        stage = c[name]
        model = (LM(c["model"], c["nca"]["states"]**4+2, 0, 1) if synthetic else
                 language_model(c, manifest["text"]["vocab"], 0)).to(device)
        optimizer = optimizer_for(model, stage, synthetic)
        array = arrays["nca_train"] if synthetic else arrays["text_train"]
        validation = arrays["nca_validation"] if synthetic else arrays["text_validation"]
        rng = np.random.default_rng(0)
        order_hash = hashlib.sha256()
        measurements = []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(4):
            sync(device)
            started = time.perf_counter()
            update(model, optimizer, array, c, stage, min(step, stage["steps"]-1), rng, device, synthetic, order_hash)
            sync(device)
            if step:
                measurements.append(time.perf_counter()-started)
        started = time.perf_counter()
        evaluate(model, validation, c, device, synthetic)
        sync(device)
        values[name] = {"seconds_per_update_median": statistics.median(measurements),
                        "validation_seconds": time.perf_counter()-started,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "peak_allocated_gpu_mb": torch.cuda.max_memory_allocated()/1e6 if device == "cuda" else None}
        del model, optimizer
    count = len(c["seeds"])
    approx = 0.
    for name, jobs in (("ppt", (len(ARMS)-1)*count), ("language", len(ARMS)*count)):
        steps = c[name]["steps"]
        evals = (steps+c["eval_every"]-1)//c["eval_every"]+1
        approx += jobs*(steps*values[name]["seconds_per_update_median"]+evals*values[name]["validation_seconds"])
    result = {"purpose": "Disposable warmup/timing calibration only; no checkpoint used for transfer",
              "device": device, "stage_measurements": values,
              "approx_full_training_and_validation_minutes": approx/60,
              "excludes": "Generation, checkpoint saves, verification, plotting, downloads, queueing and changing GPU load; not a runtime guarantee"}
    dump(Path(output)/"calibration.json", result)
    print(f"Measured estimate for full training + validation: {approx/60:.1f} minutes, excluding preparation/checkpoint/verification/archive work.", flush=True)
    return result
