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
from .nca import shuffled
from .graph import generate_paired, rewire, topology_hash
from .controls import prefix_shuffled, validate_prefix_control


def download(url, path):
    temporary = Path(str(path)+".part")
    req = urllib.request.Request(url, headers={"User-Agent": "NCA-prepretraining-transfer-v4"})
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
    manifest = {"config": c, "text": prepare_text(c, root, fixture_dir), "replicates": {}, "files": {},
                "generation": {"device": str(device), "torch": str(torch.__version__), "numpy": np.__version__,
                               "source_sha256": source_hashes()}}
    for seed in c["seeds"]:
        folder = root / f"replicate_{seed}"
        folder.mkdir()
        edges, graph_stats = rewire(c["nca"]["grid"], c["graph"]["rewire_fraction"], c["graph"]["topology_seed"] + seed)
        np.save(folder / "graph_edges.npy", edges)
        rep = {"text": manifest["text"], "nca": {}, "graph": graph_stats, "shared_finalize_seconds": 0.}
        for split, count in (("train", c["nca"]["train_rules"]), ("validation", c["nca"]["val_rules"])):
            tick = time.perf_counter()
            try:
                paired, rules, stats = generate_paired(c, count, split, seed, device, edges)
            except Exception as exc:
                dump(root / "PREPARATION_FAILED.json", {"replicate": seed, "stage": split, "error": str(exc), "config": c})
                raise
            for kind, tokens in paired.items():
                np.save(folder / f"{kind}_{split}.npy", tokens)
                shuffled_kind = "shuffled" if kind == "nca" else "shuffled_graph"
                np.save(folder / f"{shuffled_kind}_{split}.npy",
                        shuffled(tokens, c, c["nca"]["data_seed"] + 100003*seed + (10 if split == "train" else 20)))
                prefix_kind = "prefix_grid" if kind == "nca" else "prefix_graph"
                control = prefix_shuffled(tokens, c, c["nca"]["data_seed"] + 100003*seed + (30 if split == "train" else 40))
                validate_prefix_control(tokens, control, c)
                np.save(folder / f"{prefix_kind}_{split}.npy", control)
            np.savez(folder / f"rules_{split}.npz", **rules)
            stats["prepare_seconds"] = time.perf_counter() - tick
            rep["nca"][split] = stats
        if set(rep["nca"]["train"]["rule_ids"]) & set(rep["nca"]["validation"]["rule_ids"]):
            raise RuntimeError("Train/validation rule identities overlap")
        manifest["replicates"][str(seed)] = rep
    for path in sorted(root.rglob("*")):
        if path.suffix in (".npy", ".npz", ".parquet"):
            manifest["files"][path.relative_to(root).as_posix()] = sha_file(path)
    manifest["total_prepare_seconds"] = time.perf_counter()-started
    manifest["prefix_control"] = {"conditioning": "Unchanged first full frame",
                                  "histogram_scope": "Per trajectory, supervised prefix and unused suffix separately",
                                  "boundary_exclusive": c["model"]["context"]+1,
                                  "markers": "Unchanged values and positions",
                                  "selection": "Generated after joint rule selection; controls never affect admission"}
    manifest["shared_finalize_seconds"] = max(0., manifest["total_prepare_seconds"]-manifest["text"]["seconds"]
                                               -sum(s["prepare_seconds"] for r in manifest["replicates"].values() for s in r["nca"].values()))
    for rep in manifest["replicates"].values():
        rep["shared_finalize_seconds"] = manifest["shared_finalize_seconds"]
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
    text = {path.stem: np.load(path, allow_pickle=False) for path in root.glob("text_*.npy")}
    arrays = {}
    for seed in c["seeds"]:
        arrays[str(seed)] = dict(text, **{p.stem: np.load(p, allow_pickle=False) for p in (root/f"replicate_{seed}").glob("*.npy")})
        if topology_hash(arrays[str(seed)]["graph_edges"]) != manifest["replicates"][str(seed)]["graph"]["sha256"]:
            raise ValueError("Graph identity mismatch")
    return arrays, manifest
