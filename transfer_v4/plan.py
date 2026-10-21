"""Archive a prospective analysis plan before any benchmark model is trained."""
from pathlib import Path
import json

from .common import STUDY, dump, sha_file, source_hashes


def record_plan(root, c):
    root = Path(root)
    protocol = Path(__file__).resolve().parents[1]/"docs/TRANSFER_V4_PROTOCOL.md"
    (root/"FROZEN_PROTOCOL.md").write_bytes(protocol.read_bytes())
    dump(root/"analysis_plan.json", {"config": c, "study": STUDY,
                                   "source_sha256": source_hashes(),
                                   "protocol_sha256": sha_file(root/"FROZEN_PROTOCOL.md"),
                                   "scope": "Prospective validation replication and mechanism study, not untouched-domain confirmation"})
    return sha_file(root/"analysis_plan.json")


def verify_plan(root, c):
    root = Path(root)
    env = json.loads((root/"environment.json").read_text())
    if sha_file(root/"analysis_plan.json") != env["analysis_plan_sha256"]:
        raise ValueError("Frozen analysis plan checksum mismatch")
    plan = json.loads((root/"analysis_plan.json").read_text())
    if plan["config"] != c or plan["study"] != STUDY or plan["source_sha256"] != source_hashes():
        raise ValueError("Frozen analysis plan identity mismatch")
    if sha_file(root/"FROZEN_PROTOCOL.md") != plan["protocol_sha256"]:
        raise ValueError("Frozen protocol checksum mismatch")
