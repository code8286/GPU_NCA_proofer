# Scientific-transfer foundation: local CPU pipeline check

1 November 2026. This is implementation verification of a separately declared
steady bridge, not a neural transfer experiment or manuscript replication.
See [scope and usage](../../../scientific_transfer/README.md),
[equation audit](../../../scientific_transfer/EQUATION_AUDIT.md) and
[machine-readable check record](PIPELINE_CHECK.json).

Eight of eight train and four of four development reference cases pass the
declared solver/residual/BC and tighter-mesh/longer-domain checks. Maximum
common-domain profile discrepancy is 9.623207e-8; maximum raw wall-derivative
discrepancy is 2.194326e-9. Both are below the predeclared 1e-4 numerical
agreement bound. The verification solve uses the same SciPy implementation.
This does not prove the physical model or uniqueness of the solution branch.

Four analytic starting guesses pass the scoring/correction pipeline; each
corrected full state matches its retained reference exactly. The supplied
analytic guess is identical to the ordinary baseline guess, so this is a
positive pipeline check, not a learned-method comparison or speedup result.
Raw reference profiles, solve receipts, guesses, scores and correction results
are retained locally under runs/scientific_transfer/dev1. The compact public
record includes their SHA256 identities, source hashes and dependency versions.

All ten new numerical/integrity unit tests and 213 existing root regression
tests pass in actually executed local checks. Tests include finite-parameter
validation, an exact interior solution, magnetic/rotation sign terms, immutable
evidence, failed solver rejection, split/source tamper checks, withheld-generation
and training gates, numerical convergence and missing/invalid prediction coverage.
Existing regression fixtures perform their own tiny local training checks;
they are not new scientific fits or historical measurement replacements.

Retain the failed dependency setup checks and first convergence unit failure.
SciPy installation succeeded after network escalation. Linear interpolation
initially added 2.34346e-4 profile discrepancy despite ~3.6e-9 wall agreement;
cubic Hermite interpolation using RHS derivatives resolved it without changing
the acceptance bound. All executed commands/status/output/source identities
remain in the append-only repository check log.

No neural adaptation, PPT, language, GPU, OOD or official-language-test
measurement occurs in this experiment. Assessment/extrapolation SOLUTIONS remain
ungenerated. Reserved parameter tuples exist in the manifest; later training
and outcome inspection need a separate frozen protocol. Original random-rule
readiness failures, stage18 censoring and graph-confirmation requirements stand.
Full manuscript replication is disabled until the equation audit is resolved.
