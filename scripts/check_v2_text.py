"""Check the actual pinned WikiText/GPT-2 input path without training a model."""
import json
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transfer_v2.common import load_config
from transfer_v2.data import prepare_text


def main():
    c = load_config(ROOT/"configs/transfer_v2.json")
    folder = ROOT/"tmp"/("transfer_v2_text_"+uuid.uuid4().hex)
    folder.mkdir(parents=True, exist_ok=False)
    metadata = prepare_text(c, folder)
    if metadata["vocab"] != 50257 or metadata["test_status"] != "reserved_not_downloaded":
        raise RuntimeError("Wrong tokenizer or confirmation-data policy")
    for split in ("train", "validation"):
        tokens = np.load(folder/f"text_{split}.npy", allow_pickle=False)
        if tokens.ndim != 1 or len(tokens) <= 32768 or tokens.max() >= 50257:
            raise RuntimeError("Invalid tokenized split")
    if (folder/"wikitext_test.parquet").exists() or (folder/"text_test.npy").exists():
        raise RuntimeError("Reserved test split was unexpectedly downloaded")
    (folder/"text_provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"dataset_revision": c["text"]["revision"], "vocab": metadata["vocab"],
                      "splits": {k: {"tokens": v["tokens"], "rows": v["rows"]} for k, v in metadata["splits"].items()},
                      "tokenizer_sha256": metadata["tokenizer_sha256"], "test_status": metadata["test_status"]}, indent=2))
    print("PASS: real pinned WikiText/GPT-2 preparation. No training or confirmation-test evaluation performed.")


if __name__ == "__main__":
    main()
