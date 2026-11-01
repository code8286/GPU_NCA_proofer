# Development protocol: solver foundation, version 1

Date: 1 November 2026. Status: numerical pipeline development; no neural fits.
This protocol freezes no confirmatory scientific hypothesis. Configuration
and source hashes are saved before each local reference-generation command.

## Acceptance and numerical checks

For every preassigned parameter tuple, solve on L=10, 161 initial points at
tol=1e-6, then rerun independently from the analytic ordinary guess on L=14,
321 initial points at tol=1e-7. Maximum nodes are 5000/10000. Solver status must
be successful; all sampled values and defects finite; boundary max <=10*tol;
absolute spline derivative minus RHS defect <=100*tol on a 4N-3 dense probe.
Each individual solve retains status, message, adaptive nodes, iterations and
wall time. A finite-domain boundary being satisfied is not a tail-error check.

Compare velocity/rotation/temperature profiles at COMMON physical eta locations
using cubic Hermite interpolation with RHS derivatives from the verification
export mesh. Compare all three
raw wall derivatives directly. Both maximum profile and wall absolute
differences must be <=1e-4 for reference acceptance. This combined tolerance,
initial-mesh and domain-extension check does not separately identify error
sources, prove uniqueness or establish independent-code agreement. The
interpolation contributes to the recorded difference. Failed cases are retained
in the original split without replacement, removal or transfer to another split.

Seed 20261101 assigns continuous tuples to train8/dev4/assessment4/extrapolation4
before solving, with SHA256 tuple identities and no exact tuple overlap. Whole
profiles and all their meshes remain in one split. Magnetic extrapolation is
outside the ID interval with other ranges held fixed. The CLI permits only
train/dev generation for now. No reserved predictions/solutions are evaluated.

## Model-independent scoring contract

Score profiles f', g and theta with absolute RMSE; retain full boundary error,
all seven finite-difference ODE defects and individual absolute wall-derivative
errors. Finite-difference defects are mesh-dependent diagnostics, distinct from
the solver spline acceptance calculation. Missing/invalid predictions are
explicit rows; denominator includes every planned case and reference failure.
No acceptance threshold for neural competence has yet been frozen.

The correction CLI compares a supplied full-state guess to the analytic guess
using the same solver settings and retains both new solve receipts. A successful
correction must also match the retained full reference state to absolute 1e-4.
These CPU solver-only timers omit neural inference and all training/preparation
cost; do not interpret them as a speedup. A later timing protocol must control
order, repetitions, warmup, hardware and include failure/verification costs.

## Matched transfer extension: draft and disabled

Retain scratch, grid, graph, prefix-grid and prefix-graph; use identical common
language stage and then identical scientific adaptation. Existing failed
random-rule readiness and current checkpoint limitations are not waived.
Any diagnostic use of historical starts must be explicitly labeled exploratory.
No checkpoint is silently selected, continued or assigned from a different cohort.
Training codec/head policy, actual starts, data order, optimization schedule,
budget, metric thresholds and censoring need a separate prospective freeze.
Architecture changes create a separate experiment, not a transfer comparison.

Do not run remote/paid training or publish without user authorization. No new
language/PPT training, neural adaptation, OOD result or physical validation is
authorized merely by running the reference-solver commands in this directory.
