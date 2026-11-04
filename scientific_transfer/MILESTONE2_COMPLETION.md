# Milestone2 completion contract

2 November 2026. Milestone2 means a versioned, reproducible physical/numerical
specification of **steady_hybrid_v2**. It does not mean exact manuscript
replication, engineering validation, all-domain uniqueness or neural competence.
The earlier derivation, model code, material inputs and v1/v2 results are retained.

The fixed four previously declared development cases must pass all original
separate tolerance/initial-mesh/domain-extension checks. Additionally, an
independent full-system Chebyshev/root discretization must pass at BOTH49/65
points, initialized from the analytic ordinary guess rather than the collocation
solution. All full-system cross-method profile and wall errors must be <=1e-5.
Spectral collocation defect and BC must be <=1e-7 and dense off-node defect<=1e-4.
No failing case, node count or criterion may be removed after inspection.
The spectral code shares the declared RHS: independent discretization is not
independent physical derivation. Material inputs remain imported assumptions.

Revision2 solves f, g and theta directly as polynomial fields, imposing the
third/second/second-order equations and recovering all seven states by polynomial
differentiation. An analytic nonlinear Jacobian is supplied. The original
first-order formulation's failed off-node checks remain in its immutable run;
no case, resolution or tolerance is removed/relaxed by this numerical correction.

Freeze completion_config.json and explicitly selected mathematical/numerical
source hashes BEFORE checking. Bind only these model sources so later training
code/docs need not invalidate this certificate. Preserve all earlier broader
source-bound freezes unchanged. Complete a separate-invocation replay requiring
all numerical arrays, decisions/status and errors to match, except timers.
Retain full freeze/check/replay evidence under a new run and public result folder.
Run material, limit, conservative-radiation, boundary/failure and replay-integrity
tests. Source/package and root/Prod parity checks complete distribution.

The certificate is complete only if all fixed checks, replay and relevant tests
actually pass. No reserved train/test parameter data is used by this contract.
New numerical CPU costs remain distinct from imported GPU costs. This closes
milestone2 for the declared steady model while milestone1's author-code/intended
equation questions remain unresolved. Milestone3 starts from this versioned
model and a separate prospective neural interface/data/budget protocol.
