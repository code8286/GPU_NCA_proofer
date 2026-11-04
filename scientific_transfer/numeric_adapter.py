"""Shared numeric heads around an unchanged pretrained transformer backbone."""
import math
import numpy as np
import torch
from torch import nn
from transfer_v4.model import Backbone
from .hybrid_v2 import HybridParameters, wall
from .physics import boundary

PARAMETERS = tuple(HybridParameters.__dataclass_fields__)


class NumericAdapter(nn.Module):
    def __init__(self, architecture, core_state, seed):
        super().__init__()
        # Core construction cannot consume the common adapter initialization RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.core = Backbone(architecture)
        self.core.load_state_dict(core_state, strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input = nn.Linear(9, architecture["width"])
            self.roles = nn.Parameter(torch.randn(2, architecture["width"])*0.02)
            self.output = nn.Linear(architecture["width"], 7)

    def forward(self, values):
        # Two tokens for each independent (parameter tuple, eta) query. No gold
        # prior field values and no teacher-forced profile tokens are inputs.
        param = torch.cat([values[:, :8], torch.zeros_like(values[:, :1])], dim=1)
        query = torch.cat([torch.zeros_like(values[:, :8]), values[:, 8:]], dim=1)
        z = torch.stack([param, query], dim=1)
        return self.output(self.core(self.input(z)+self.roles)[..., -1, :])


def encode(case, config):
    ref = case["reference"]
    params = case["parameters"]
    features = [(params[k]-config["parameter_ranges"][k][0]) /
                (config["parameter_ranges"][k][1]-config["parameter_ranges"][k][0]) for k in PARAMETERS]
    x = np.column_stack([np.tile(features, (len(ref["eta"]), 1)), np.array(ref["eta"])/10])
    y = np.array(ref["state"]).T/np.array(config["output_scales"])
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def learning_rate(step, c):
    if step <= c["warmup_updates"]:
        return c["learning_rate"]*step/c["warmup_updates"]
    phase = (step-c["warmup_updates"])/(c["updates"]-c["warmup_updates"])
    return c["learning_rate"]*.5*(1+math.cos(math.pi*phase))


def decision(rows, c):
    if (len(rows) != c["development_cases"] or
            len({r["case_id"] for r in rows}) != c["development_cases"] or
            not c["require_all_development_cases"]):
        return False
    return all(r["valid"] and r["profile_rmse"] <= c["id_profile_rmse_max"] and
               r["wall_absolute_max"] <= c["id_wall_absolute_max"] and
               r["boundary_max"] <= c["id_boundary_max"] for r in rows)


def crossing(curve, c, sustained=False):
    for i, row in enumerate(curve):
        if decision(row["cases"], c) and (not sustained or all(decision(r["cases"], c) for r in curve[i:])):
            return {"reached": True, "step": row["step"], "query_tokens": row["query_tokens"]}
    return {"reached": False, "step": None, "query_tokens": None, "censored_at": c["updates"]}


@torch.no_grad()
def evaluate(model, cases, c):
    model.eval()
    rows, predictions = [], {}
    for case in cases:
        x, _ = encode(case, c)
        y = (model(x).cpu().numpy()*np.array(c["output_scales"])).T
        predictions[case["case_id"]] = y.tolist()
        valid = bool(np.isfinite(y).all())
        if not valid:
            rows.append({"case_id": case["case_id"], "valid": False})
            continue
        ref = np.array(case["reference"]["state"])
        p = HybridParameters(**case["parameters"])
        predicted_wall = wall(y, p)
        rows.append({"case_id": case["case_id"], "valid": True,
                     "profile_rmse": float(np.sqrt(np.mean((y[[1, 3, 5]]-ref[[1, 3, 5]])**2))),
                     "wall_absolute_max": max(abs(predicted_wall[k]-case["reference"]["wall"][k]) for k in predicted_wall),
                     "boundary_max": float(np.max(np.abs(boundary(y[:, 0], y[:, -1]))))})
    return rows, predictions
