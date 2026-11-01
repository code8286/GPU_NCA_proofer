# Stage 10 paired CSV column caveat

Identified 21 October 2026 while preparing stage 11 reporting. Original evidence
is preserved. No model, score, JSON decision or criterion changes.

The [stage 10 paired CSV](results/diversity_v10_run1/paired_comparisons.csv)
has a derived-column collision: flattening the nested readiness object overwrote
the complete comparison's `passed` value with the readiness-only `passed` value.
Thus its `passed` column means trajectory readiness, not the complete
native/intervention/readiness comparison. In particular, standard seeds 131 and
149 have negative native/intervention improvements even though that CSV field
reads `True`.

The [original JSON](results/diversity_v10_run1/results.json) retains both values
correctly: `paired_comparisons[*].passed` is the complete criterion;
`paired_comparisons[*].readiness.passed` is readiness alone. The
[stage 10 review](results/diversity_v10_run1/REVIEW.md) also reports the correct
result: weighted 3/3, standard 1/3, and failed both-objective primary. There is no
change to that interpretation.

Stage 11 exports `passed` and `readiness_passed` separately and checks both
against the nested JSON. A regression exercises a failed comparison with passing
readiness. Earlier files and the original stage 10 export template remain intact
so historical evidence and its exact reproduction are preserved.
