# Stage 5: checkpoint-only readiness and order sensitivity

Prepared after the v4 review, before computing the cross-evaluation below.
This stage uses inspected v4 development data. Frequency-baseline failures have
already been inspected; order cross-evaluation has not. This is not untouched
confirmation or a new GPU experiment. Preserve all v1-v4 sources and results.

## Frozen diagnostic

Use exactly the six v4 cohorts, the first 32 held-out rules, selected PPT
checkpoints, original context/mask and both families (grid and graph).
For each family/cohort evaluate a 2x2 table:

| Selected checkpoint | Coherent held-out data | Matched prefix shuffle |
| --- | --- | --- |
| Structured-trained | Lss | Lsp |
| Prefix-shuffle-trained | Lps | Lpp |

Retain per-rule patch NLL, accuracy and target counts. Native cells Lss and Lpp
must reproduce the original selected-checkpoint metrics within 1e-4. No marker
targets enter patch scores; the conditioning frame stays masked. Use the actual
saved weights, full trained prefix, same initial frame and exact target counts.
No alternate shuffles, rule subset, seed exclusion or margin retuning is allowed
after reading this diagnostic. The fixed margins are development decisions.

Fit an add-one global frequency prior on all training supervised patch targets.
Select concentration from [0.1, 1, 10, 100, 1000, 10000] using only the first
256 training rules. The causal count predictor starts from the actual initial
frame, updates after observing each patch and scores exactly the same targets.
Its per-trajectory joint score is invariant under the matched target shuffle.

For every structured family/cohort, require BOTH:

1. **Frequency advantage:** Lfrequency - Lss >= 0.01 nats/patch target.
2. **Ordering advantage:** Lsp - Lss >= 0.01 nats/patch target.

Report the shuffle-trained model's ordering gap Lpp-Lps and the difference
(Lsp-Lss)-(Lpp-Lps) as descriptive diagnostics, not additional admission tests.
Report each of the 12 structured cohort decisions and all 48 model/data cells.
Rule rows are nested observations; do not treat them as independent model seeds
or manufacture a language-transfer p-value from them. No language training is
permitted unless all 12 structured decisions and measurement controls pass.
Passing permits designing the next experiment; it does not prove transfer.

## Known-rule measurement control

Before main checkpoint scoring, validate the measurement on a tiny categorical
cellular rule: increment each binary cell modulo two at each frame. The 2x2
patch token therefore maps deterministically to its bitwise complement.
Use a 4x4 grid, eight frames, 256 training initial conditions and 32 disjoint
held-out initial conditions. This is one known rule, not held-out rule transfer.
Fit a smoothed previous-frame patch transition table only on training data.
It must beat the causal frequency predictor and have an ordering advantage
greater than the same fixed margins on the matched held-out shuffle.

The causal frequency predictor itself must fail the two-part gate (zero self
advantage and exchangeable joint scores). Independently check the closed form
against explicit sequential prediction in software tests. This control validates
the measurement; it does not show that the v4 transformer learned this toy rule
or that the random-rule NCA family is learnable by this architecture.

## Evidence and allowed next work

Archive this protocol, config, trusted v4 source identities, new diagnostic
source identities, input manifest and checkpoint SHA values BEFORE scoring.
Use a fresh output directory. Save positive/negative control, per-rule cross
scores, gate decisions and provenance. Reject modified sources/inputs/plans.
The public report excludes private paths, weights, corpora and raw caches.

This package has no training command. A reusable authorization guard verifies
the plan, input identities, positive control and all two-part decisions before
returning approved structured checkpoint identities for a future runner.
It must reject this stage if any decision fails. Existing v4 gates and metrics
remain unchanged. The official language test is never read by this stage.

If readiness fails, investigate prediction/representation with one change at a
time. Do not launch a language suite or alter the gate to salvage these models.
If it succeeds, declare one matched serializer/topology ablation while holding
the language stage fixed. PPT-duration changes are separate cost experiments.
Fresh cohorts and untouched confirmation data are required for later claims.
