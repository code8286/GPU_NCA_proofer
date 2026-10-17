from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np

from .common import dump, sha_file, source_hashes
from .nca import generate_split, shuffled


def download(url, path):
    temporary = Path(str(path)+".part")
    req = urllib.request.Request(url, headers={"User-Agent": "NCA-prepretraining-transfer-v2"})
    try:
        with urllib.request.urlopen(req, timeout=90) as src, temporary.open("wb") as dst:
            while chunk := src.read(1024*1024):
                dst.write(chunk)
        temporary.replace(path)
    except Exception:
        # Preserve failed download for diagnosis; never consume a partial file.
        raise


def prepare_text(c, root, fixture_dir=None):
    started = time.perf_counter()
    provenance = {"source": c["text"], "splits": {}, "tokenizer": c["text"]["tokenizer"]}
    if fixture_dir is not None:
        if c["purpose"] != "execution_check":
            raise ValueError("Synthetic local text fixtures are not research data")
        splits = ["train", "validation"]+(["test"] if c["text"]["evaluate_test"] else [])
        for split in splits:
            raw = (Path(fixture_dir)/f"{split}.txt").read_bytes()
            np.save(root / f"text_{split}.npy", np.frombuffer(raw, dtype=np.uint8).astype(np.uint16))
            provenance["splits"][split] = {"bytes_sha256": hashlib.sha256(raw).hexdigest(), "tokens": len(raw)}
        provenance["vocab"] = 256
        provenance["test_status"] = "execution_fixture_only" if "test" in splits else "not_loaded"
    else:
        if c["text"]["tokenizer"] != "gpt2":
            raise ValueError("Execution checks require an explicit --fixture-dir")
        import pyarrow.parquet as pq
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str((root/"tokenizer_cache").resolve()))
        import tiktoken
        encoder = tiktoken.get_encoding("gpt2")
        provenance["vocab"] = encoder.n_vocab
        provenance["tokenizer_package_version"] = importlib.metadata.version("tiktoken")
        provenance["parquet_reader_version"] = importlib.metadata.version("pyarrow")
        # Fingerprint the actual merge table + special tokens, not just its name.
        merge = sorted((token.hex(), rank) for token, rank in encoder._mergeable_ranks.items())
        provenance["tokenizer_sha256"] = hashlib.sha256(json.dumps([merge, encoder._special_tokens], separators=(",", ":")).encode()).hexdigest()
        splits = ["train", "validation"]+(["test"] if c["text"]["evaluate_test"] else [])
        for split in splits:
            url = (f"https://huggingface.co/datasets/{c['text']['dataset']}/resolve/{c['text']['revision']}/"
                   f"{c['text']['subset']}/{split}-00000-of-00001.parquet")
            path = root / f"wikitext_{split}.parquet"
            download(url, path)
            rows = pq.read_table(path, columns=["text"])["text"].to_pylist()
            raw = "\n".join(rows)
            ids = np.array(encoder.encode_ordinary(raw), dtype=np.uint16)
            np.save(root / f"text_{split}.npy", ids)
            provenance["splits"][split] = {"url": url, "parquet_sha256": sha_file(path),
                                           "text_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                                           "tokens": len(ids), "rows": len(rows)}
        provenance["test_status"] = "loaded_for_predeclared_final_evaluation" if "test" in splits else "reserved_not_downloaded"
        provenance["license"] = "WikiText: CC-BY-SA-3.0 / GFDL; see Salesforce/wikitext dataset card"
    provenance["seconds"] = time.perf_counter()-started
    return provenance


def prepare(c, root, device, fixture_dir=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    import torch
    manifest = {"config": c, "text": prepare_text(c, root, fixture_dir), "nca": {}, "files": {},
                "generation": {"device": str(device), "torch": str(torch.__version__), "numpy": np.__version__,
                               "source_sha256": source_hashes()}}
    for split, count in (("train", c["nca"]["train_rules"]), ("validation", c["nca"]["val_rules"])):
        tick_prepare = time.perf_counter()
        try:
            tokens, rules, stats = generate_split(c, count, split, device)
        except Exception as exc:
            dump(root / "PREPARATION_FAILED.json", {"stage": split, "error": str(exc), "config": c})
            raise
        np.save(root / f"nca_{split}.npy", tokens)
        np.savez(root / f"rules_{split}.npz", **rules)
        tick = time.perf_counter()
        np.save(root / f"shuffled_{split}.npy", shuffled(tokens, c, c["nca"]["data_seed"]+(10 if split == "train" else 20)))
        stats["shuffle_seconds"] = time.perf_counter()-tick
        stats["prepare_seconds"] = time.perf_counter()-tick_prepare
        manifest["nca"][split] = stats
    train_ids = set(manifest["nca"]["train"]["rule_ids"])
    val_ids = set(manifest["nca"]["validation"]["rule_ids"])
    if train_ids & val_ids:
        raise RuntimeError("Train/validation rule identities overlap")
    for path in sorted(root.iterdir()):
        if path.suffix in (".npy", ".npz", ".parquet"):
            manifest["files"][path.name] = sha_file(path)
    manifest["total_prepare_seconds"] = time.perf_counter()-started
    manifest["shared_finalize_seconds"] = max(0., manifest["total_prepare_seconds"]-manifest["text"]["seconds"]
                                               -sum(s["prepare_seconds"] for s in manifest["nca"].values()))
    dump(root / "manifest.json", manifest)
    return manifest


def load_data(root, c):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["config"] != c:
        raise ValueError("Cached data/config mismatch")
    for name, expected in manifest["files"].items():
        if sha_file(root / name) != expected:
            raise ValueError(f"Cached file checksum mismatch: {name}")
    arrays = {path.stem: np.load(path, allow_pickle=False) for path in root.glob("*.npy")}
    return arrays, manifest
