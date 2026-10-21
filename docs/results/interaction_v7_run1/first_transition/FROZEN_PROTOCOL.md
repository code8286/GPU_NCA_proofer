# Stage 7 supplementary first-transition diagnostic

Declared during fixed-budget training, BEFORE any stage 7 assessment likelihood
or final-model assessment was inspected. Main training plan, sources, endpoints
and readiness/coupling criteria stay unchanged. This is a separate development
diagnostic, not a retroactive amendment or confirmatory endpoint.

Structural motivation: both rules have period two. A transformer can use its
own patch two frames ago on later transitions, whereas the declared local table
only sees one frame ago. Beating that table cannot by itself establish use of
neighbor information. Measure the FIRST patch of the FIRST transition, at
absolute token 7, conditioned only on tokens 0..6 (initial full frame plus next
frame's start marker). No earlier repeated frame or new-frame patch is available.

Freeze queries before inspecting model scores. Keep all 128 original assessment
initial states. For a paired intervention, flip initial cell (row 0, column 3),
which is outside the top-left target patch but adjacent through the periodic
boundary. The complement target patch stays unchanged; the spatial target
patch must flip bit zero. Exclude intervention pairs whose changed initial
state's JOINT orbit appears in training or development; determine this shared
eligibility mask from identities alone before scoring. Do not remove any
original primary assessment row or model seed. The number of eligible pairs is
reported, and pairs are not treated as independent model replicates.

After all twelve training arms and the primary assessment finish, score initial,
coherent-trained and shuffle-trained checkpoints on the original first-patch
queries and eligible changed queries. Report NLL, accuracy, paired-both-correct
rate and the fraction of pairs whose argmax changes as the rule requires.
Fit an add-one table p(next first patch | initial own first patch) only on the
512 training first transitions, and also report uniform NLL log(16).

As a separately labeled stricter diagnostic, all three spatial coherent-trained
models must beat BOTH the fitted first-patch table and uniform predictor by
0.01 nats on all 128 original queries. This does not change the primary bridge
status or authorize language training. Interventions and this finite comparator
are evidence about these queried predictions, not proof of a unique internal
algorithm, Bayes-optimal local bound or unseen-rule generalization.

Bind the parent plan, query arrays, this protocol, source script and expected
checkpoint identities (rule/arm/seed under the parent's final-update policy)
before scoring. Final hashes are taken only from the parent's verified training
receipt. Keep source-run output and public copies with provenance. Preserve any
negative result. No new training or hyperparameter selection occurs here.
