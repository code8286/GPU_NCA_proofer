# Stage 13 input-integrity wording clarification

Recorded during the frozen fit, 30 October 2026. No scientific source, input,
configuration, target, selection or fitting procedure is changed.

The frozen protocol says the prior assessment is not read by the adaptation
command. More precisely: the inherited stage 12 receipt verifier rehashes ALL
its recorded files, including the prior assessment bytes, solely to detect
changes. Only train/development JSONL is parsed into examples by stage 13;
assessment prompts/labels never enter fitting, checkpoint selection, generation
or new scores. No new OOD examples or predictions are produced by fitting.
The assessment was already inspected in stage 12 and is development evidence,
not untouched confirmation. This clarification does not claim that a filesystem
read never occurred. Preserve the original protocol/receipt verbatim.

The independent replay repeats the same integrity hashing. The reserved official
language test and ungenerated fresh confirmation are not accessed.
