# Verification record — 2026-10-17

The corrected [browser bundle](../BrowserBundle.lean) was accepted by the
[official Lean Web editor](https://live.lean-lang.org/), using its stable
**Lean 4.35.0 with Mathlib** environment. The editor contents were copied back
and compared with the saved bundle, normalizing Windows clipboard line endings;
they matched exactly. Source hashes are in [status.json](status.json).

The completed editor showed **46 informational messages**: the version,
44 theorem-axiom reports and the completion marker. No proof errors or warnings
remained. Every theorem's axiom report contains only a subset of `propext`,
`Classical.choice` and `Quot.sound`, or no axioms. There is no `sorryAx` or
unchecked native computation axiom in the accepted result.

- [Final diagnostics](final-diagnostics.txt): rendered infoview transcript.
- [All 44 axiom reports](axiom-reports.txt): extracted rendered report lines.
- [Editor screenshot](lean-check.png): stable version and completed message list.
- [First full-check diagnostics](attempt-1-diagnostics.txt): retained failure.

## Failures and corrections

An exploratory library-name probe referenced an unavailable power inequality
lemma. The final monotonicity proof instead uses induction on natural numbers.
An early multiline editor insertion changed indentation and the editor later
timed out. That session supplied no accepted whole-project result. A fresh
editor session using plain-text paste preserved the proof source.

The first correctly pasted full bundle produced four source errors: three
real-valued definitions needed `noncomputable` annotations, and the two-mode
formula left a polynomial reassociation goal. The definitions were annotated;
the algebraic proof was finished with `ring`. The failed axiom report exposed
`sorryAx` on that unfinished theorem; it is preserved in the failed transcript.
The corrected full bundle was rechecked and all 44 final axiom reports are clean.
The original and corrected bundle hashes are recorded in `status.json` and
the append-only research check log.

## Limits of this evidence

This is browser-based Lean checking of the mechanically assembled source.
A local modular `lake build` was **not run**: Lean/Lake are absent from this
workstation. The modular project and GitHub workflow are prepared for that
separate check. No remote CI result is claimed. Mathlib is release-tag-pinned
for future local builds; no resolved local Lake manifest is fabricated.

The records establish acceptance of the specified statements. The
[proof map](../PROOF_MAP.md) lists their hypotheses and the mathematics that
remains informal. The numerical examples are exact analytic toy calculations,
not CPU simulations or GPU measurements.
