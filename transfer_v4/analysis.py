"""Frozen common-target comparisons; never discard ties or censored cohorts."""
import math
import statistics

from .common import STUDY


def first_hit(rows, threshold):
    return next((r for r in rows if r["validation_nll"] <= threshold), None)


def common_target_analysis(curves, c):
    comparisons = []
    keys = ("language_tokens", "language_seconds", "cumulative_seconds")
    secondary_pairs = (("prefix_grid", "grid_nca"), ("shuffled_graph", "graph_nca"),
                       ("shuffled_nca", "grid_nca"), ("scratch", "graph_nca"))
    for reference_arm, arm in list(STUDY["primary_pairs"])+list(secondary_pairs):
        rows = []
        for seed in c["seeds"]:
            scratch = curves.get(("scratch", seed), [])
            reference = curves.get((reference_arm, seed), [])
            treatment = curves.get((arm, seed), [])
            threshold = scratch[-1]["validation_nll"] if scratch and scratch[-1]["language_step"] == c["language"]["steps"] else None
            a = first_hit(reference, threshold) if reference and threshold is not None else None
            b = first_hit(treatment, threshold) if treatment and threshold is not None else None
            row = {"seed": seed, "target_nll": threshold,
                   "reference_reached": a is not None, "graph_reached": b is not None,
                   "reference_step": a["language_step"] if a else None,
                   "graph_step": b["language_step"] if b else None}
            for key in keys:
                # A step-zero target does not define a positive token ratio.
                row[key+"_speedup"] = a[key]/b[key] if a and b and a[key] > 0 and b[key] > 0 else None
            rows.append(row)
        item = {"reference_arm": reference_arm, "arm": arm, "rows": rows,
                "target_policy": STUDY["target"], "declared_cohorts": len(c["seeds"])}
        for key in keys:
            values = [r[key+"_speedup"] for r in rows]
            complete = bool(values) and all(v is not None and v > 0 for v in values)
            item["geomean_"+key+"_speedup_all_cohorts"] = math.exp(statistics.mean(math.log(v) for v in values)) if complete else None
        ratios = [r["language_tokens_speedup"] for r in rows]
        complete = bool(ratios) and all(v is not None for v in ratios)
        wins = sum(v is not None and v > 1 for v in ratios)
        n = len(rows)
        # Ties count as non-wins (conservative). Missing/censored: no p-value.
        p = sum(math.comb(n, k) for k in range(wins, n+1))/2**n if complete else None
        item.update(strict_token_wins=wins, ties_count_as_nonwins=True,
                    one_sided_sign_p=p, alpha=STUDY["alpha_per_comparison"],
                    criterion_met=bool(complete and n >= STUDY["minimum_cohorts"]
                                       and p <= STUDY["alpha_per_comparison"]
                                       and item["geomean_language_tokens_speedup_all_cohorts"] > 1))
        comparisons.append(item)
    primary, secondary = comparisons[:2], comparisons[2:]
    for item in secondary:
        for key in ("one_sided_sign_p", "alpha", "criterion_met"):
            item.pop(key)
        item["scope"] = "Descriptive secondary comparison; no additional significance claim"
    eligible = c["purpose"] == "development"
    return {"study": STUDY, "comparisons": primary,
            "secondary_common_target_comparisons": secondary,
            "eligible_for_research_interpretation": eligible,
            "supports_graph_advantage_on_declared_target": eligible and all(i["criterion_met"] for i in primary),
            "multiplicity": "Two co-primary one-sided sign tests, alpha .025 each (Bonferroni family bound .05)",
            "scope": "Language-token transfer advantage on an inspected validation task; not end-to-end GPU speed or untouched confirmation",
            "censoring": "Any missing crossing withholds the all-cohort ratio and significance claim; no imputation or success-only subset"}
