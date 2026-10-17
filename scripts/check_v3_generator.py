"""Small CPU check of default paired rule-filter feasibility, without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from transfer_v3.common import load_config, source_hashes, sha_file
from transfer_v3.graph import generate_paired, rewire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.rules <= 64:
        parser.error("This check is restricted to 1–64 retained rules per replicate")
    torch.set_num_threads(2)
    c = load_config(ROOT/"configs/transfer_v3.json")
    original = json.loads((ROOT/"docs/results/transfer_v2_run1/gpu/environment.json").read_text())
    for name, expected in original["source_sha256"].items():
        if sha_file(ROOT/"transfer_v2"/name) != expected:
            raise ValueError("Historical v2 source changed")
    if (ROOT/"transfer_v3/model.py").read_bytes() != (ROOT/"transfer_v2/model.py").read_bytes():
        raise ValueError("Transformer implementation changed from measured v2")
    results = []
    for seed in c["seeds"]:
        edges, topology = rewire(c["nca"]["grid"], c["graph"]["rewire_fraction"], c["graph"]["topology_seed"]+seed)
        arrays, _, stats = generate_paired(c, args.rules, "train", seed, "cpu", edges)
        results.append({"paired_seed": seed, "candidates": stats["candidates"], "accepted": stats["accepted"],
                        "changed_donor_fraction": topology["actual_changed_fraction"],
                        "grid_profile_quartiles": stats["nca"]["profile_gzip_quartiles"],
                        "graph_profile_quartiles": stats["graph"]["profile_gzip_quartiles"],
                        "grid_fresh_quartiles": stats["nca"]["fresh_rollout_gzip_quartiles"],
                        "graph_fresh_quartiles": stats["graph"]["fresh_rollout_gzip_quartiles"],
                        "shape": list(arrays["graph"].shape)})
    print(json.dumps({"scope": "Small CPU generation check only; no synthetic learning, language transfer or GPU performance measured",
                      "v2_runner_unchanged": True, "transformer_byte_identical_to_v2": True,
                      "v3_sources_sha256": source_hashes(), "replicates": results}, indent=2))


if __name__ == "__main__":
    main()
