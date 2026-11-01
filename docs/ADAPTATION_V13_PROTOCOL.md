# Stage 13: matched ID task-interface and trainability calibration

Prospective development protocol, 30 October 2026. Freeze this document, code,
configuration, input hashes, full batch order and historical checkpoint identities
before any adaptation update. Preserve stage 12 and all earlier negative results.

## Question and scope

Can the existing five language-trained models learn the BFS/Dijkstra distance-
vector interface and generalize to structurally held-out ID graphs under one
identical bounded supervised adaptation procedure? This is a third training
stage, not synthetic PPT or another language stage. User authorization for this
calibration does not override the original random-rule readiness failure.
These historical weights are admitted ONLY for interface/trainability diagnosis,
not a confirmatory graph-transfer or learning-acceleration test. One inspected
cohort cannot support a graph-specific statistical claim.

Use stage 12 v4 cohort 59 scratch/grid_nca/graph_nca/prefix_grid/prefix_graph final
language endpoints. Verify all five historical hashes and the identical 2000
language updates, 8192000 tokens, initial language head, actual order and schedule.
Do not redo or extend that language stage. Preserve the entire inherited model,
including each learned tied embedding/head; no head reset, output-vocabulary
restriction, added tokens, context extension, trace, hint or few-shot example.
Their adaptation-start heads legitimately differ after language training;
identical head POLICY does not mean erasing these trained differences.

## Fixed data and adaptation

Reuse the already-declared stage 12 training (512 examples, 256 base graphs) and
ID development (128 examples, 64 graphs), cyclic ER sizes 8 and 10, both tasks.
They are development data, not untouched confirmation. Original WL split/cut
exclusions and independent graph oracles must verify before fitting. No new
OOD examples or confirmation split is generated. Preserve train/development
bytes exactly; the prior assessment is NOT read by the adaptation command.

Full-backbone fp32 CPU fitting: 600 optimizer updates per arm, batch 8, no
accumulation, AdamW betas (0.9,0.999), eps 1e-8, weight decay .01 on ALL trained
parameters, gradient norm clip 1.0. Fresh optimizer for every arm. LR warms
linearly for 30 updates to .0003 then cosine decays to .00003 at update 600.
Two CPU threads, deterministic PyTorch. No dropout. All arms receive the exact
same precomputed repeated-permutation example order, seed 20261031, and the same
schedule; no adaptive stopping or extra training for a struggling arm.

Encode prompt and reference separately with the unchanged GPT-2 tokenizer;
concatenate prompt + answer + EOS. Mask all prompt targets and right-padding
from loss. Teacher forcing is used ONLY during supervised training and NLL
diagnostics. Mean cross entropy is over supervised answer/EOS tokens, so longer
answers have proportionally more weight. Project vocabulary logits only at
supervised hidden positions; test loss AND gradients against ordinary full LM
projection. Right padding cannot affect earlier positions under causal attention.
No input truncation. Log real unpadded input tokens, supervised targets, padded
positions, updates, sampled IDs/order hash, learning rate, loss and timing.

## Frozen measurements and decision

At updates 0,100,300,600, greedy-generate answers for ALL 128 ID-development
examples. Use the unchanged stage 12 unconstrained decoder/scorer and 128-token
cap. Predictor sees only ID/prompt. Report valid-vector rate, exact vector match,
node accuracy, coverage and task/size cells separately. Record teacher-forced
answer/EOS NLL on the first 64 training examples and ALL development examples
at each checkpoint to distinguish fitting from generalization; these NLLs are
diagnostics, never replacements for generated-answer correctness.

Primary calibration succeeds only if EVERY arm at the fixed final update has
at least 80% exact match pooled AND in EACH of four task/size ID cells, at least
95% valid vectors pooled, and 100% context coverage. Report first observed pooled
80% ID-development crossing versus actual adaptation input/target tokens and
training time, without interpolation; unreached targets are censored. A transient
crossing cannot override a failed final/cell criterion. Targets remain fixed.

Always complete all 600 updates and retain all checkpoints, initial model
identities, raw generations and optimizer state. Compare no checkpoint by OOD
score. Do not relax a target, extend budget, drop an arm or retune after inspection.
This bounded diagnostic may fail; it does not promise learned graph algorithms.

OOD remains gated. Only after all fixed endpoints pass the ID criterion may a
separate invocation evaluate ALL predeclared stage 12 assessment cells at the
fixed final checkpoint for all five arms. Otherwise write an explicit skipped
decision and produce NO new OOD predictions. Even a pass supplies development
calibration only, not original-PPT readiness or confirmatory graph acceleration.

## Verification, costs and next boundary

In a separate process, reopen all 20 retained adaptation checkpoints, reproduce
every development generation/score and train/development NLL, verify optimizer
steps and common actual sampled order/token/schedule counts. Bind freeze-before-
fit chronology and source/input/result hashes. Same-code replay is not a second
independent fit or neural implementation. Tiny software fixtures are separate
from the scientific 5 x 600 = 3000 updates.

Separate newly measured CPU adaptation/update time, evaluation time and complete
command wall time. Historical GPU language/PPT/generation costs remain imported
evidence and are not newly timed or treated as zero. No speedup, GPU throughput,
cost recovery or significance claim is defined by this single-cohort study.

After failure, inspect fixed train versus development errors before choosing a
separately frozen treatment. After success, an actual graph claim still requires
eligible synthetic initialization and the separate categorical/random-rule bridge,
fresh paired cohorts, identical full language AND task-adaptation stages, frozen
OOD targets/censoring/multiplicity and full cost accounting. Retain scratch,
grid and both prefix controls; graph must beat grid AND its matched shuffle.
No official language test, remote/paid training, email reply or push is used.
