# LaTeX-to-Lean proof map

All names below are in namespace `GraphNCA`. Proof status is tied to exact
source hashes in `verification/`, not merely to a theorem name.

| LaTeX result | Lean coverage | Explicit boundary |
| --- | --- | --- |
| Section 2, relabeling equivariance | `graphStep_equivariant` | Shared arbitrary local rule with consistently relabeled states and typed neighbors. No transformer serialization invariance is inferred. |
| Section 2, endpoint swaps preserve typed degree | `swappedRelation_bijective`, `swappedRelation_unique_preimage` | Composition with a receiver swap preserves a permutation. Absence of self/duplicate edges, connectivity and improved mixing are not consequences of these statements. |
| Section 4, mean lag invariant | `totalLag_permutation`, `meanLag_permutation` | A finite vertex set and one source permutation. Nonempty vertices are required for division by cardinality. The result is independent of the patch assignment; every typed relation satisfies the same identity separately. |
| Equations `eq:lagrange` and full-window visibility | `patchLag_bounds`, `fullWindow_visibility`, `prefix_visibility`, `binary_parent_lag_fits` | Patch indices are bounded; visibility means sufficient preceding-token distance, not successful learned prediction. |
| Crop-prefix qualification | `crop_prefix_can_miss_parent`, `qualifying_prefix_count` | An exact 33-lag example exceeds an available prefix of 32; the count 96 concerns target offsets, not an empirically measured fraction of useful patches. |
| Equation `eq:patch`, binary specialization | `encodeBinaryPatch_lt_sixteen`, `encodeBinaryPatch_injective`, `binary_patch_digits` | Four binary digits only. General categorical bijectivity, a bundled finite-type bijection and entropy preservation are not claimed formalized. |
| Equation `eq:reset` | Three `resetHead_*` theorems | A product-parameter reset preserves the backbone, ignores the old head, and is idempotent. Its derivative/Jacobian has not been formalized. |
| Proposition `prop:alignment` | `descent_from_remainder_bound`, `sufficient_backbone_transfer`, `one_step_alignment` | The Taylor upper bound is an explicit hypothesis. The calculus theorem deriving it from Lipschitz-gradient assumptions remains open in this folder. |
| Equation `eq:graphvsgrid` | `compare_from_two_sided_bounds` | Requires an upper graph remainder and lower grid remainder. It does not subtract two upper bounds. |
| Equations `eq:graphcovtransfer`, `eq:exactgraphtarget` | `linear_surrogate_expansion`, `equal_size_target_comparison`, `squared_difference_identity` | Exact finite-coordinate squared-loss algebra, allowing flattened matrix indices. Equal squared size is assumed. Expectation/covariance reduction, degree-to-size counting and the six-vertex graph construction are not formalized. |
| Equation `eq:linearpretrain` | `synthetic_linear_recurrence` | Solves the explicitly supplied scalar affine recurrence for every natural step. Connecting it to an optimizer derivative is separate. |
| Equation `eq:modeformula` | `scalar_recurrence_solution`, `two_mode_closed_form`, `coefficient_dominance` | Exact scalar recurrence and diagonal-coordinate loss. Nonnegative weights/factors are stated. Full Hessian spectral decomposition is not formalized. |
| Initial-ranking reversal | `initial_loss_ranking`, `after_one_step_ranking_reverses` | Exact real arithmetic for the displayed two-dimensional quadratic. No claim about an observed model trajectory. |
| Section 6, first hits 39 and 25 | Four boundary inequalities, three monotonicity lemmas, `scratch_first_hit`, `transferred_first_hit` | Exact rational powers and quantified exclusion of all earlier natural steps. These are analytic toy losses, not real validation hitting times. |
| Equation `eq:breakeven` | `total_cost_break_even`, `cost_examples` | An algebraic equivalence for supplied costs plus exact rational examples. It does not measure hardware costs. |
| Equation `eq:paramcount` and token arithmetic | `model_parameter_count`, `retained_token_count`, `linear_surrogate_losses` | Checks displayed numerical substitutions. Deriving the architecture's parameter formula from executable model definitions remains outside the formal project. |

## Not yet formalized

- Grid convolution/message-passing equivalence at the array/kernel level.
- Fourier eigenvectors, the 8×8 diffusion spectral gap and directed singular norms.
- Gather-operator Frobenius perturbation and logit-margin bounds.
- Entropy/rule-count bounds, conditional rule inference, KL/mutual-information
  decompositions and shuffle covariance.
- Lag variance and the random non-self comparison law.
- Analytic Taylor remainder estimates, probability expectations and complete
  matrix/Hessian diagonalization.
- Any empirical transfer effect, actual-data integrity claim, cost measurement
  or conclusion about the research hypothesis.

These entries remain mathematical arguments in the LaTeX note. They have no
placeholder Lean theorem pretending to prove them. Future work should add
separate declarations with precise hypotheses and extend the axiom audit.
