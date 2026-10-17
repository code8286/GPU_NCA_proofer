# Where pilot 1 diverges from NCA pre-pre-training

This audit uses the [paper v1](https://arxiv.org/html/2603.10055v1), the
[author repository](https://github.com/danihyunlee/nca-pre-pretraining/tree/main)
at revision `bdd1c71e04eb2e5a2c91481faf278e95846b4431`, our `experiment.py`, and
the independently verified first Kaggle output. The references and source
checksums are retained locally in `sources/nca-pre-pretraining-reference/`.

## Main correction

Our code really did train synthetic-then-text; it did not omit the language
stage. The error was in the experimental question and in interpreting its result.
The pilot fixed the total number of updates and replaced some text training with
synthetic training, rather than first measuring how a pre-pre-trained
initialization changes an otherwise identical language-training run.

That is a resource-allocation experiment. The intended first experiment
is a **transfer-initialization experiment**, followed by an additional cost audit.
The earlier roughly 43% timing penalty is a correct calculation for pilot 1's
declared total-cost metric. It is not a clean measurement of the paper's
language-stage convergence question, and should not be used as a decisive
rejection of pre-pre-training.

The paper does retain a no-pre-pre-training scratch control. Keeping that control
is appropriate; giving it extra target-language exposure at the nominal language
stage boundary is the problem. Graph and grid NCA should be compared as
alternative pre-pre-training treatments feeding the same downstream procedure.
The paper's Appendix D additionally counts synthetic and language tokens
together; total cost remains a useful secondary question.

## Definite design mismatches

### 1. Unequal language exposure and an incorrect scratch boundary

In `experiment.py::run_one`, every arm runs `pre_steps=100`, then
`total_steps-pre_steps=900` updates. For `scratch`, both stages use target text.
For grid/graph/shuffle, the first stage uses synthetic data. Therefore:

| Arm | Updates before nominal common language stage | Nominal common stage | Target-text updates in total |
| --- | --- | --- | ---: |
| Scratch | 100 target-text updates | 900 target-text updates | 1000 |
| Grid / graph / shuffled NCA | 100 synthetic updates | 900 target-text updates | 900 |

Scratch already has learned byte embeddings and a text-trained backbone at the
nominal language-stage origin. Its first 100 text updates also use a separate
short cosine cycle, followed by an optimizer/scheduler restart. Replotting against
target-text tokens removes the exposure mismatch on the x-axis, but cannot undo
the different scheduler or recover the missing final scratch test checkpoint.

For the next controlled run, use separate `ppt_steps=K` and `language_steps=N`.
Skip the PPT loop entirely for scratch, and start its language optimizer at
language step zero alongside the transferred initializations.

### 2. We changed the generator's function family

The reference [generator](https://github.com/danihyunlee/nca-pre-pretraining/blob/bdd1c71e04eb2e5a2c91481faf278e95846b4431/utils/nca.py)
uses categorical one-hot states, a spatial perception layer with four channels,
then a pointwise hidden layer with 16 channels and categorical output. Our pilot
uses signed binary inputs and a different 9-to-16-to-2 MLP. Thus it is not just a
smaller grid or a different implementation of the same reference rule family.

Our test proving convolution/message-passing equivalence is correct for **our
own binary rule**. It does not establish equivalence to the author's NCA.

The default reference state alphabet has 10 values, yielding 10,000 possible
2x2 patch patterns. Our binary alphabet yields 16. The reference launcher actually
allocates 64,000 model vocabulary slots; possible patch patterns and model output
vocabulary must not be conflated. Our shared 274-way byte/synthetic vocabulary is
a separate design choice.

### 3. Complexity selection and burn-in were omitted

The reference [PPT launcher](https://github.com/danihyunlee/nca-pre-pretraining/blob/bdd1c71e04eb2e5a2c91481faf278e95846b4431/scripts/prepretraining/nca_prepretraining.sh)
enables rule selection and skips 10 initial automaton steps. Our generator
accepts every sampled rule and includes its initial random frame.

From the returned arrays, **36.33% of graph trajectories** and **35.74% of grid
trajectories** have identical final two frames. For a deterministic rule this
means that rollout has reached a fixed point. This is direct evidence that the
unfiltered corpus contains trivial late dynamics. It does **not** establish that
this alone caused negative transfer.

Our logged gzip ratios are not comparable to the reference filter threshold.
We compressed uint16 token IDs with frame markers; the author filter removes
markers and compresses its token representation separately per trajectory. Raw
integer width, alphabet, sequence length and burn-in all affect the ratio. The
new diagnostic explicitly labels a marker-free int32 calculation as
non-equivalent because our cached rollouts still lack reference burn-in.

Do not simply apply the author's 0.5 threshold to our binary uint16 files and
call that a faithful filter. First standardize generation and serialization.

### 4. We never established that the PPT stage learned the intended task

The [author PPT trainer](https://github.com/danihyunlee/nca-pre-pretraining/blob/bdd1c71e04eb2e5a2c91481faf278e95846b4431/src/nca_ppt.py)
constructs separate training and held-out rule sets. It evaluates synthetic
prediction loss and has facilities for next-grid generalization evaluation.
Its dataset masks the initial context grid from prediction loss.

Our runner only logs **text validation loss**, even while the model is training
on synthetic symbols. It does not log synthetic training/validation loss, does
not hold out NCA rules, and does not save the PPT checkpoint before transferring.
We can verify that the updates executed, but cannot establish from the artifacts
whether the model learned transferable rule inference, merely frequencies, or
trivial copying. The first-stage checkpoint cannot be recovered from the final
language checkpoint.

Random cropped windows, unmasked context and repeated small cached trajectories
also differ from the reference trajectory-aligned task. Frame markers account
for 11.11% of our synthetic tokens. Predicting their periodicity is an easy task;
a falling overall NCA loss would therefore need patch-only and held-out-rule
checks before being treated as mechanistic evidence.

### 5. Both optimization stages use an untested common configuration

The reference PPT code uses Adam with learning rate 1e-4 and no weight decay in
its main setup. The language launcher uses a separate learning rate/optimizer
configuration. Our pilot uses AdamW at 6e-4, weight decay 0.1, and gradient clipping
in both stages. Those changes are not automatically wrong for a tiny model, but
we never validated the PPT learning curve before committing to them.

The paper's language-stage weight decay is 1e-4, versus our 0.1. The much larger
value and other hyperparameter differences are plausible sensitivity issues;
they are not proven causes of the result. Raw absolute PPT token counts also do
not establish undertraining by themselves because model size changed greatly.

### 6. Model, context and text task were changed simultaneously

Our 1.86M-parameter model uses learned absolute position embeddings and a GELU/
LayerNorm transformer with context 128 and byte-level Shakespeare training.
The reference [model implementation](https://github.com/danihyunlee/nca-pre-pretraining/blob/bdd1c71e04eb2e5a2c91481faf278e95846b4431/utils/models.py)
uses a Llama backbone with rotary positions; its principal launcher uses context
1024. The paper's reported model scales and language datasets are far larger.

Small-scale experiments remain useful, but reducing the state alphabet, context,
model family, corpus and pretraining budget together does not preserve the
conditions under which transfer was demonstrated. Whether our tiny model can
learn the relevant in-context dynamics is a question to measure, not assume.

### 7. The graph extension was introduced before validating grid transfer

We retained spatial 2x2 patch serialization while rewiring graph edges. Many
serialized spatial neighbors then cease to be actual graph neighbors, while
actual graph neighbors can be far apart in the sequence. This may make rule
inference harder or change complexity. It is an additional hypothesis, not a
guaranteed improvement from using a GNN.

First implement the reference grid update as relational message passing on the
same lattice and verify equivalent outputs. Changing connectivity should then
be a separately declared treatment. The first pilot's grid-vs-graph comparison
still is a valid comparison of its two synthetic treatments, but neither has
established the intended grid-transfer effect.

## What we should retain

- Causal next-token prediction on serialized trajectories is appropriate.
- Frozen randomly sampled NCA rules are appropriate; the generator does not need
  a separate learned task before being used as synthetic data.
- **Reinitializing input/output token embeddings is part of the reference
  transfer protocol.** It is not an error to fix by preserving arbitrary NCA
  symbol semantics. Transfer the backbone; initialize a common byte/text head
  consistently across arms. Reference code:
  [language transfer](https://github.com/danihyunlee/nca-pre-pretraining/blob/bdd1c71e04eb2e5a2c91481faf278e95846b4431/src/openwebtext_pt.py).
- Paired seeds, fixed language evaluation, checksum verification, a grid control
  and a frequency-preserving shuffled control remain useful.
- Total-cost accounting remains useful, but must be presented alongside the
  primary language-stage transfer measurements rather than replacing them.

Our separate-text arm preserved its learned byte embeddings, whereas the primary
synthetic arms reset theirs. That is a meaningful practical text baseline, but
does not isolate backbone transfer under a common embedding policy. If retaining
a text-PPT baseline in the next run, distinguish text-PPT with a reset language
head from a separate text-PPT embedding-preservation ablation. Do not silently
compare different transfer policies and attribute the entire difference to NCA
structure.

## What existing logs say after correcting the exposure axis

The retrospective plot is [the retained retrospective plot](docs/results/pilot1/language_aligned_curves.png).
The exact calculations and generator checks are saved beside it in
`docs/results/pilot1/prepretraining_diagnosis.json`.
Plot lines are means and shaded bands are the minimum/maximum across three
seeds, not confidence intervals.

Mean validation loss across the original three seeds:

| Target-text updates seen | Scratch | Grid PPT → text | Graph PPT → text | Shuffled PPT → text |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 2.6141 | 2.6100 | 2.6428 | 2.6184 |
| 200 | 2.4512 | 2.4344 | 2.4460 | 2.4398 |
| 500 | 2.0193 | 2.0065 | 2.0287 | 2.0133 |
| 900 | 1.7542 | 1.8002 | 1.8170 | 1.8099 |

There are small intermediate improvements on this axis, but no sustained graph
advantage. Scratch's scheduler remains different, so this is diagnosis rather
than a clean replacement experiment. These are **validation** comparisons;
scratch was not saved at the matched 900-text-update point for test evaluation.
Correcting the question does not turn the old result into proof of success.

## Correct next protocol

1. **Separate stage budgets.** Define PPT length K and language length N
   independently. Grid, graph and shuffled arms run K+N updates. The no-PPT arm
   runs N text updates from a fresh initialization, not K+N text updates under a
   different schedule. If a fixed-total-cost arm is desired, label it separately.
2. **Validate synthetic learning before transfer.** Hold out entire rules;
   retain full trajectory boundaries; mask initial context; log synthetic loss,
   patch-only loss/accuracy and held-out-rule prediction against frequency/copy
   controls. Save the PPT checkpoint and rule/data identities.
3. **Recover a faithful grid reference first.** Preserve its categorical state
   representation, update structure, patch encoding, complexity selection and
   burn-in. Scale the transformer and budgets transparently rather than changing
   all mechanistic assumptions at once. Audit reference-code caveats below.
4. **Enter an identical language stage.** Use common language embedding/head
   initialization, architecture, data sequence, optimizer state, scheduler and N
   across no-PPT, grid, graph and shuffled initializations. Full-backbone transfer
   is the primary setup; attention-only transfer can be a declared ablation.
5. **Report both questions.** Plot loss vs language tokens/time from stage zero,
   and separately loss vs total training cost including PPT/generation. The
   paper's convergence comparison uses the scratch endpoint as a reference;
   our arbitrary 2.5 threshold is an additional metric, not an exact reproduction.
6. **Add the graph treatment only after the grid check.** Match rule family,
   tokenization and data-complexity policy. Separate changing the numerical
   implementation from changing the graph topology.

Use development validation for generator/hyperparameter decisions, and reserve
untouched confirmation data for the next claims. Pilot 1's test set has already
been examined. No new GPU benchmark has been launched by this audit.

## Reference-code caveats to record rather than silently inherit

The public code is useful but should not be treated as a bit-exact executable
specification of every sentence in the paper:

- Paper v1 states temperature 1e-3; the pinned launcher passes 1e-4. Record the
  chosen value explicitly.
- `src/nca_ppt.py` contains `epoch+1 % args.generate_rules == 0`. Python parses
  that as `epoch + (1 % args.generate_rules) == 0`, not
  `(epoch + 1) % args.generate_rules == 0`. With the launcher value one it only
  triggers at epoch zero, despite the intended periodic refresh. This is a
  static code finding; the author code has not been executed locally.
- `utils/nca.py` applies scalar-width `jnp.pad` to a three-axis array; that pads
  channels as well as spatial dimensions. A port must explicitly decide whether
  to reproduce that behavior or implement the spatial padding described by the
  paper. Do not call the two bit-identical without testing.
- The tokenizer constructs masked marker targets, but `NCADataset.__getitem__`
  reconstructs targets from `seq`, then masks the initial context grid. Therefore
  do not claim the currently executed reference loss necessarily masks all
  markers just because the tokenizer produces a masked target array.
- Model constructors have weight-tying defaults that need explicit inspection;
  the paper's description alone is insufficient to infer launcher behavior.

These source caveats are separate from the faults in our independent pilot; they
do not excuse its unmeasured synthetic learning or unequal language budgets.
