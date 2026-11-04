# Milestone 3 execution and milestone 4 ID assessment

3 November 2026. All five cohort 59 arms completed the separately frozen local
CPU exploratory protocol. Each arm completed 400 updates, 515,200 numeric query
tokens and 1,803,200 supervised scalars with the same 32 training and eight ID
development profiles. See the [protocol](../../SCIENTIFIC_M3_EXPLORATORY.md)
and [complete score curves](RESULTS.json).

| Arm | Worst profile RMSE | Worst wall error | Worst boundary error | Profiles passing all criteria |
| --- | ---: | ---: | ---: | ---: |
| Scratch | .03607 | .50162 | .26997 | 0/8 |
| Grid | .02919 | .26012 | .56847 | 0/8 |
| Graph | .02569 | .37087 | .11913 | 0/8 |
| Prefix grid | .03159 | .36066 | .31428 | 0/8 |
| Prefix graph | .02853 | .37943 | .23519 | 0/8 |

The endpoint limits were .02 for profile RMSE and .05 each for wall and boundary
error. Every arm missed all three worst-case limits. All first and sustained
competence crossings were censored at update 400. Speedup to competence is
undefined. The original stronger readiness decision remains failed, and the
ID result keeps the OOD gate closed.

The training-loop timers total about 1,189 seconds of elapsed time on CPU.
The machine-readable results list evaluation times and imported T4 measurements
separately. Raw predictions, optimizer states and weights are kept in the local run.

A separate invocation replayed all 25 checkpoints and matched all 200 ID
predictions exactly. It recomputed each score and crossing decision and verified
the optimizer step counts and learning-rate traces. All 26 scientific-transfer
regression tests passed locally.
