# Scientific transfer: declared hybrid model numerical milestone

1 November 2026. New local CPU numerical development evidence. No neural fits,
checkpoint adaptation, language/PPT/GPU/OOD or official-language-test measurement.

See [derivation and scope](../../../scientific_transfer/DERIVATION_V2.md),
[visual manuscript audit](../../../scientific_transfer/EQUATION_AUDIT.md),
[frozen source/config plan](PLAN.json), [complete solve evidence](CHECKS.json)
and [exact-copy SHA256 receipt](PROVENANCE.json).

## Milestone decisions

1. Manuscript audit is PARTIAL. Visual inspection withdraws the suspected Pr
   inconsistency: Eq11 agrees with Eq20. Other printed density/momentum problems
   remain. Exact author code/reference values have not been located. Faithful
   reproduction remains disabled; no claim of resolving author intent is made.
2. steady_hybrid_v2 is DECLARED AND NUMERICALLY CHECKED. The separate steady
   local-pole model has final-volume mixture closures, conservative nonlinear
   Rosseland radiation and explicitly named dimensionless wall coefficients.
   This is not the manuscript's unsteady system or experimental validation.
3. Neural freeze/import/training remains pending; no five-arm neural result.
4. Held-out ID competence is unmeasured; reserved ID/OOD outcomes remain untouched
   by these fixed four-case numerical checks. No parameter-sweep dataset is fitted.
5. New reference-check costs are recorded; full inherited neural costs await an
   actual neural protocol. No zero historical costs or cost-recovery ratio.

## Fixed numerical outcomes

All four declared cases pass: pure, Zn-only, TiO2-only and hybrid radiative.
Each uses one base solve and separately checks tolerance, initial mesh and domain
extension. Across all12 comparisons, maximum profile discrepancy is
2.4200894622e-9 and wall-coefficient discrepancy is4.0103629306e-9, both below
the frozen absolute1e-4 criterion. All per-solve status/residual/BC/iteration/
node/time values and full exported states are retained in CHECKS.json.

The independent Chebyshev/root velocity and spectral linear thermal methods
do not call solve_bvp or start from its solution. Both49/65-node variants pass
the fixed1e-5 wall agreement check. Maximum velocity-wall discrepancy against
collocation is2.2594370819e-9; thermal-wall discrepancy8.0424511495e-10.
Both agree with the rounded literature Homann scalar1.311938 within3.1e-7.
Agreement at this limit is not validation of every mixture/radiation closure,
global geometry, branch uniqueness, or the unsteady manuscript physics.

All19 scientific-transfer unit tests (ten retained and nine new) and all213
existing root regression tests pass in executed logged commands. Root regression
fixtures contain their own tiny training checks; these are not new scientific
fits or replacements for historical measurements. Zero-loading material ratios
are exactly1; the radiation-free zero-loading RHS agrees with v1; conservative
nonlinear radiation and finite parameter/closure limits are tested.

## Costs and preservation

The complete new numerical milestone command's internal check timer is
0.2718832 CPU seconds, including reference variants and independent limiting
checks, but excluding interpreter startup, imports, file output and packaging.
Each collocation solve also retains its own seconds; the logged command wall
duration is separate. Hardware/timers are not controlled throughput claims.
No inherited neural cost is charged as zero or combined across hardware.

PLAN.json and CHECKS.json are byte-identical copies of the retained local
runs/scientific_transfer/milestone2 artifacts; PROVENANCE.json binds their hashes.
Old v1 numerical profiles/plans/outputs and every earlier neural result remain
unchanged. The inspected PDF and rendered audit pages remain local and are not
redistributed. No contact message, push or external publication occurs.
