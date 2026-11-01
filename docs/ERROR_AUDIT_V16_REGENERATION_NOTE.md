# Stage 16 regeneration disclosure

The frozen protocol says “No ... OOD data”. That wording is too broad for the
executed implementation. Its unchanged `ood_v12.data.generate` helper regenerates
the full deterministic synthetic bundle in memory, including the assessment/OOD
rows, while reproducing the exact ID train/development identities. Stage 16
selects only cache16 and ID-development128 for scoring and counter aggregation.
No assessment/OOD rows, predictions or scores are written, inspected as study
outcomes, passed to any model, or used for selection in stage 16. No official
language-test data are accessed. There are no new model forwards of any kind.

Preserve the original frozen wording and executed sources; do not silently
rewrite the protocol or claim that only ID data were regenerated. This is a
data-regeneration scope deviation, not a change to the frozen evaluation cases,
endpoints or metrics. Original OOD and synthetic-readiness gates remain closed.
