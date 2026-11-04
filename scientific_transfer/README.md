# Scientific transfer: rotating-sphere development benchmark

Separate root experiment, created 1 November 2026. This directory does not
modify or supersede stages 1-18, their checkpoints, gates or negative results.
Stage 18 is now completed: all five arms were censored at 600 updates, with
undefined speedup ratios. No positive graph-transfer claim motivates this build.

## Implemented scope

An executable CPU reference-solver and evaluation foundation is implemented.
The initial model is **steady_bridge_v1**, an explicitly declared simplified
axisymmetric stagnation-flow boundary-value problem with rotation, magnetic
damping, buoyancy and a passive temperature equation. It is NOT the manuscript's
unsteady hybrid-nanofluid/radiation system. See [equation audit](EQUATION_AUDIT.md)
and [protocol](PROTOCOL.md). V1 has no material-property fits, radiation, heat source,
unsteady acceleration, dimensional reconstruction or manuscript Cf/Nu values
implementation. Its raw wall derivatives are deliberately labeled as such.

Implemented: seven-state collocation solver; analytic ordinary initial guess;
solver status, BC and spline-defect checks; independent tighter/longer-domain
numerical rerun; whole-parameter-tuple split manifest; configuration/source
hash binding; retained failed cases; complete-coverage prediction scoring;
predicted-guess correction versus an ordinary-guess solve; immutable JSON output.
The verification rerun uses the SAME SciPy implementation, not an independent
physics derivation or numerical algorithm. Multiple branches are not excluded.

The [milestone2 completion contract](MILESTONE2_COMPLETION.md) adds independent
full-system discretization and separate replay to the original numerical checks.
The [milestone3 protocol](MILESTONE3_PROTOCOL.md) now specifies a numeric adapter,
all five retained language starts, ID-only data, common optimization and fixed
competence/censoring criteria. Checkpoint preparation and a readiness-gated
training runner are implemented; the retained starts remain unready. No learned
starting guesses or training speedup are established by implementation/tests.

## Usage

Run from the repository root with an environment containing
`scientific_transfer/requirements.txt`. Use the existing append-only logger:

```powershell
python scripts/run_logged.py --label scientific-transfer-tests -- python -m unittest scientific_transfer.test_scientific_transfer -v
python scripts/run_logged.py --label scientific-transfer-plan -- python -m scientific_transfer.run plan --output runs/scientific_transfer/dev1/plan.json
python scripts/run_logged.py --label scientific-transfer-dev -- python -m scientific_transfer.run generate --plan runs/scientific_transfer/dev1/plan.json --split development --output runs/scientific_transfer/dev1/development.json
```

Use a fresh output path for a new run; existing evidence is never overwritten.
The solver plan is immutable and binds local source/config/doc hashes. A source
edit requires a new plan; old plans and outputs remain evidence. The plan
contains reserved parameter tuples but the CLI cannot generate assessment or
extrapolation solutions. No reserved outcome has been inspected by this build.
The small counts (8 train/4 development) are for pipeline development only.

Predictions are a JSON object keyed by `case_id`, each value a finite 7-by-N
array ordered `[f, f_prime, f_double_prime, g, g_prime, theta, theta_prime]`
on the EXACT reference eta mesh. Profiles are not accepted on an implicit
normalized/remeshed coordinate system.

```powershell
python scripts/run_logged.py --label scientific-transfer-score -- python -m scientific_transfer.run evaluate --dataset runs/scientific_transfer/dev1/development.json --predictions runs/scientific_transfer/dev1/predictions.json --output runs/scientific_transfer/dev1/scores.json
python scripts/run_logged.py --label scientific-transfer-correct -- python -m scientific_transfer.run correct --dataset runs/scientific_transfer/dev1/development.json --predictions runs/scientific_transfer/dev1/predictions.json --output runs/scientific_transfer/dev1/corrections.json
```

Missing/invalid guesses and reference failures remain explicit denominator rows;
no average silently drops them. Correction records both solves and agreement
with the retained reference. Solver-only CPU timers exclude inference, generation,
fitting, preparation and verification overhead; they cannot support cost recovery.
Raw runtime artifacts belong under ignored `runs/`, never this selected source
directory. The supplied manuscript remains external and is not redistributed.

## Next implementation milestones

Update: the [v2 derivation](DERIVATION_V2.md) implements a separate steady hybrid
model with final-volume material closures and conservative nonlinear radiation.
The visual audit withdraws the suspected Pr mismatch, while confirming other
printed issues. An independent spectral Homann/thermal check and four-case
material/tolerance/domain checks are executable after a source-bound freeze:

```powershell
python scripts/run_logged.py --label scientific-v2-freeze -- python -m scientific_transfer.milestones freeze --output runs/scientific_transfer/milestone2
python scripts/run_logged.py --label scientific-v2-check -- python -m scientific_transfer.milestones check --output runs/scientific_transfer/milestone2
python scripts/run_logged.py --label scientific-v2-tests -- python -m unittest scientific_transfer.test_hybrid_v2 -v
```

The original five milestones remain below. Milestone1 is partly resolved and
milestone2 now has an explicit implementation; later neural/ID/OOD milestones
are not relabeled completed. V1's historical plans bind its earlier source state;
retain them rather than rebuilding over old artifacts after these additions.

## Completion and matched-training preparation

Use NEW output directories for each freeze. Finish milestone2's certificate
before preparing milestone3. Preparation loads/checks original language states
but performs no neural prediction or optimizer update; the published v5 guard
currently blocks fitting before its first fitting marker. No stage18 exception
is inherited. A changed eligibility/cohort policy requires a new declaration.

```powershell
python scripts/run_logged.py --label scientific-m2-completion-freeze -- python -m scientific_transfer.complete_m2 freeze --output runs/scientific_transfer/m2_completion_r2
python scripts/run_logged.py --label scientific-m2-completion-check -- python -m scientific_transfer.complete_m2 check --output runs/scientific_transfer/m2_completion_r2
python scripts/run_logged.py --label scientific-m2-completion-replay -- python -m scientific_transfer.complete_m2 replay --output runs/scientific_transfer/m2_completion_r2
python scripts/run_logged.py --label scientific-m3-prepare -- python -m scientific_transfer.training prepare --completion runs/scientific_transfer/m2_completion_r2 --output runs/scientific_transfer/m3_preparation
python scripts/run_logged.py --label scientific-m3-verify -- python -m scientific_transfer.training verify --output runs/scientific_transfer/m3_preparation
```

M2 completion is numerical/specification completion for steady_hybrid_v2, not
experimental validation or completion of manuscript milestone1. Its source
receipt binds selected mathematical files so later neural work remains separate.
The M3 plan binds all current scientific sources, actual inherited LM/core/new
adapter hashes, data/order and readiness evidence. Numeric query tokens are
counted separately from historical language tokens. See its protocol for the
limits of inherited cost aggregates and the additional work needed for milestone5.

Completion revision2 passed all four cases at both spectral resolutions and
the separate-invocation replay. The first completion run remains a failed
first-order spectral discretization diagnostic; revision2 corrects that
discretization without changing model equations, cases, resolutions or bounds.
Milestone2 is complete for the DECLARED steady model, with exact manuscript
replication and experimental validation still outside that completion claim.

1. Resolve the manuscript audit using the original derivation/code and reproduce
   at least one numerical reference with tolerance/domain/material checks.
2. Declare a versioned physical model before extending beyond the steady bridge.
3. Freeze training interface, numerical encoding, fixed model/head policy,
   budgets, seeds, ID competence thresholds, crossing/censoring and wall-error
   targets BEFORE fitting. Integrate all five retained initialization arms with
   complete checkpoint/start identities and identical language/adaptation stages.
4. Establish held-out ID accuracy before reserved parameter extrapolation and
   separate mesh-refinement assessment. Retain graph/grid and graph/prefix-graph
   co-primary contrasts; use fresh paired cohorts for confirmation.
5. Count full synthetic generation/PPT/language/task training and inference,
   reference/correction/verification costs separately, retaining hardware and
   failures. Earlier failed synthetic readiness remains failed.
