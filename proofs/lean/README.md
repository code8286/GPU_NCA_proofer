# Lean proof documentation

This folder formally checks a selected part of the [mathematical note](../../docs/theory/main.tex)
in **Lean 4.35.0 with Mathlib**. It contains proofs, a mapping to the LaTeX
arguments, reproducible project files, and retained verification records.
The 44 named theorems include supporting lemmas and exact arithmetic facts;
they are not 44 independent research claims.

## Contents

- [PROOF_MAP.md](PROOF_MAP.md): exact coverage, hypotheses and remaining obligations.
- [GraphNCA/](GraphNCA): the six proof modules.
- [GraphNCA.lean](GraphNCA.lean): modular library entry point.
- [Audit.lean](Audit.lean): representative kernel-axiom queries.
- [BrowserBundle.lean](BrowserBundle.lean): mechanically assembled single-file
  version, with an axiom query for every named theorem.
- [verification/](verification): diagnostics, source hashes and checking evidence.
- [lean-toolchain](lean-toolchain) and [lakefile.toml](lakefile.toml): pinned
  Lean and Mathlib release configuration.

The formalization proves graph relabeling equivariance and preservation of a
typed source permutation; total/mean serialization-lag invariance; binary patch
range, injectivity and digit recovery; parent-window bounds; head-reset identities;
the algebraic consequence of an assumed Taylor bound; finite-coordinate
linear-surrogate identities; scalar/diagonal recurrence formulas and a ranking
reversal; exact rational first-hit results at steps 39 and 25; and selected cost
and parameter-count calculations.

It does **not** prove real language-model acceleration. Entropy, the complete
Fourier/spectral argument, smoothness-to-Taylor estimates and full matrix
diagonalization remain informal. The [proof map](PROOF_MAP.md) makes those
boundaries explicit. A supplied inequality is a theorem hypothesis, not a proof
that the actual model satisfies it.

## Recheck with a local Lean installation

Install Lean using the [official installation instructions](https://lean-lang.org/install/).
From this folder:

```bash
lake update
lake exe cache get
lake build
lake env lean Audit.lean
```

The first setup downloads Mathlib and its compiled cache. Do not commit `.lake/`.
Commit the generated Lake manifest after checking its resolved dependency
revisions when establishing a local release build. No manifest is fabricated
here: the workstation used for initial preparation had no Lean/Lake executable.

From the research root, retain subsequent build output using:

```bash
python scripts/run_logged.py --label lean-build --cwd proofs/lean -- lake build
python scripts/run_logged.py --label lean-axioms --cwd proofs/lean -- lake env lean Audit.lean
```

The configured GitHub workflow builds the modular library and audits its axioms.
A configured workflow does not mean that a remote CI job has already run.

## Recheck in the official web editor

Open [Lean Web](https://live.lean-lang.org/), select the stable release with
Lean 4.35.0, and paste the entire `BrowserBundle.lean` as plain text.
Wait for checking to finish and inspect **All Messages**. The expected output
is the Lean version, 44 axiom reports and the completion marker, with no errors
or placeholder axioms. The marker alone is insufficient: Lean can continue
processing later commands after an earlier proof error.

The initial check uses this browser route; its exact scope is recorded under
`verification/`. A modular `lake build` is a distinct check and is not described
as locally completed.

## Trust and maintenance

Proof sources use no `sorry`, `admit`, project-defined axioms, `native_decide`,
or unchecked runtime escape. The allowed foundational axioms are `propext`,
`Classical.choice`, and `Quot.sound`; individual theorems can use fewer.
Exact rational inequalities use proof-producing arithmetic tactics.

Edit the six canonical modules, then run from the research root:

```bash
python scripts/run_logged.py --label lean-sources -- python scripts/check_lean_sources.py --write-bundle
```

This script checks source hygiene and bundle consistency; it is **not** a Lean
proof checker. Recheck in Lean after every proof change, retain failures and
new source hashes, and update the proof map if a theorem's scope changes.
After saving a new browser verification record, use
`python scripts/check_lean_sources.py --check-record` from the research root
to detect stale hashes, missing theorem reports or unexpected axiom dependencies
in the retained evidence. This inspects saved records; it does not rerun Lean.
Root sources are canonical and the publication copy is maintained through the
normal `Prod` synchronization process. Project licensing follows the root
[Apache 2.0 license](../../LICENSE).
