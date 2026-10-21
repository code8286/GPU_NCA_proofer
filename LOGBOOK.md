# Research logbook

Dates use Asia/Kolkata (UTC+05:30); machine timestamps use UTC. Append entries;
retain negative outcomes. `logs/checks.jsonl` stores complete captured check
output, command, exit code, duration and experiment SHA-256. It starts with this
publication work; earlier command transcripts are not reconstructed.

## 2026-10-17 — Historical evidence imported

Sources: pre-existing validation, research context, methodology audit and Kaggle
review. These are historical records, not newly executed publication checks.

- CPU diagnostic: 11 scientific checks; five checkpoint reloads with zero held-out
  loss error. The short run never reached 2.5 validation NLL; time-to-target was
  unresolved. See `docs/results/cpu_check/`.
- Returned Tesla T4 pilot: 15 completed runs; maximum independent CPU/GPU loss
  difference 8.2e-08. Graph took 1.432x scratch pipeline time to target and had
  higher final test NLL by 0.053230 nats/byte. See `docs/results/pilot1/`.
- Scientific failure/limitation: no demonstrated graph benefit. Scratch saw 1000
  text updates; NCA arms saw 100 synthetic + 900 text updates, with a scheduler
  confound on the language-exposure axis. Retrospective alignment neither fixed
  the confound nor established sustained graph benefit.
- Held-out-rule synthetic learning and stage-boundary checkpoints were absent.
  Those measurements cannot be recovered from final checkpoints.
- Original archives/checkpoints/data remain in the research workspace. Public
  compact evidence has provenance and does not replace full artifact verification.

## 2026-10-17 — Publication preparation

Intent: a contributor-ready repository preserving prior evidence. Root files are
canonical; `Prod/` is the checked publication copy.

Inspection: no pre-existing root Git repository or applicable `AGENTS.md` found.
Installed: Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.5.2, Matplotlib 3.11.2;
CUDA unavailable. Historical experiment SHA-256:
`76b535acbf08050295648d5692d46690366c1f4802d5668c99edeca003cf1981`.

Baseline: `python -m unittest discover -s tests -v` passed 11 tests in 2.838s before
edits. This check predates the logger and is rerun through it below.

Changes: public docs, licensing/notices, contributor templates, CI and maintenance
rules; compact evidence and pinned MIT reference source; notebook parity,
publication integrity, offline smoke and check-output logging. The historical
experiment source is preserved byte-for-byte.

Failed preparation attempt: the first documentation patch used both delete and
add operations for `.gitignore`; the patch tool rejected the duplicate target.
No files changed. Reapplied using an update operation.

Results and follow-ups are appended after executing the new checks.

## Entry structure for subsequent work

Append a dated entry containing intent/hypothesis; files/behavior changed;
protocol/configuration/seeds; commands/environment; outputs/checksums; observed
results including failures; interpretation/limitations; next action. Use
`scripts/run_logged.py` for every check and retain its exit code.

## 2026-10-17 — First maintenance checks and environment fixes

- Expected failure: notebook parity returned exit 1 before regeneration, correctly
  detecting stale embedded source. This intentional guard check is retained.
- Unexpected failure: first expanded suite ran 17 tests; the original 11 passed,
  but six new test methods produced nine errors because this sandbox denies file
  writes/cleanup in the operating-system temporary directory. Moved disposable
  test work under the repository's ignored `tmp/checks/` directory; cleanup stays
  within that verified workspace path. No production training failure occurred.
- The initial full transcript contained user/runtime paths in Python tracebacks.
  Preserved the original transcript locally at
  `tmp/publication_original_check_transcripts.jsonl`; the public transcript keeps
  all errors with machine-identifying path prefixes normalized. Future captured
  outputs use this normalization automatically. Do not log commands with secrets.
- Direct license download failed because shell DNS/network access is restricted.
  The official Apache 2.0 text was inspected through the web tool. Used the
  locally installed canonical Apache terms and added the official appendix;
  included the complete license and separate project copyright notice.

## 2026-10-17 — Successful publication validation

- Expanded suite: **17 tests passed** in 2.268s after the workspace temporary-file
  fix. Cache/config tampering, unsafe publication paths, missing inputs, secret/raw
  artifact selection and stale inventories are covered alongside the original
  11 scientific controls.
- Offline smoke: all five arms completed six updates (384 tokens/arm); all five
  checkpoints reloaded with **zero held-out loss error**, and CSV/JSON/SVG/PNG
  outputs were produced. Full fixture metrics are in the check transcript. No
  target was reached and no scientific transfer inference is drawn from this text.
- New independent CPU re-audit of the original ZIP: **15 checkpoints** and all
  **12 paired comparisons** passed source/data/budget/hash checks. Maximum CPU/GPU
  loss difference was **8.195638656616211e-08**. Completed in 41.035s. New derived
  files are in `runs/publication_audit_20261017/`; original artifacts are unchanged.
- Reproduced retrospective diagnosis in the new audit directory; its JSON is
  exactly equal to the original diagnosis. Scheduler limitations and the negative
  graph finding remain unchanged. This is a CPU recheck of prior GPU artifacts,
  not new GPU training.
- Verified all 12 pinned upstream files offline and exported the existing CPU
  diagnostic to `tmp/publication_cpu_validation.md` without rewriting historical
  validation or result bundles. Summary: `docs/results/PUBLICATION_CHECKS.json`.
- Added Git staging/checkout verification and byte-preserving attributes for the
  historical experiment, third-party snapshot and evidence. This prevents line
  ending conversion from invalidating archived hashes when cloning on another OS.
- Another documentation patch attempt referenced an outdated context line and
  was rejected atomically; reapplied with the actual context. No partial changes.

## 2026-10-17 — Standalone publication and documentation checks

- The 101-file initial snapshot passed file inventory, 16 embedded notebook
  source checks, all 12 pinned reference hashes and historical evidence hashes.
- A temporary Git staging/checkout cycle preserved all hashes, including under
  `core.autocrlf=true`. Seven disposable secret/cache/artifact probes stayed
  ignored. No persistent Git repository, commit, remote or push was created.
- Ran all **17 tests from inside Prod** (2.230s) and a separate offline five-arm
  smoke from that directory (15.027s). All passed; every checkpoint again
  reproduced held-out loss exactly. Parent-directory imports were not needed.
- Documentation link check initially failed: moving the historical README into
  `docs/PILOT_PROTOCOL.md` left two root-relative links one directory too shallow.
  Corrected both links; failure retained in the machine log before rerunning.
- Starter ZIP generation now includes all selected publication sources and an
  internal inventory; the notebook itself embeds only its 16 execution sources
  and essential documents. Final publication includes the added documentation
  checker. Its generated ZIP remains local and ignored by Git.
- Logging a sync command appends its log entry after the copy; documented that
  the final sync/check must run directly after logged checks, so publication and
  canonical log contents remain identical.

Documentation recheck passed: **22 local Markdown links** are valid; starter ZIP
CRCs and every internal inventoried file hash passed. A selected-file scan found
no personal workspace/user path prefixes or common API-token prefixes in the
publication text. This is a targeted scan, not a claim of comprehensive secret
scanning. License is the complete 11,358-byte Apache 2.0 text; the upstream MIT
snapshot remains unchanged. Updated `PUBLICATION_CHECKS.json` with standalone,
notebook, ignore-rule and evidence verification scope.

## 2026-10-17 — Publication completion

Final generation/parity, all 22 documentation links, starter ZIP CRC/hash checks,
and the full Git checkout/ignore-rule test passed on the **102 selected files**
plus the publication manifest. Sixteen embedded notebook sources, 12 pinned
reference files and compact historical evidence hashes are verified. The original
experiment SHA-256 is unchanged. There are 22 machine check records so far,
including the intentional stale-notebook failure and the resolved temporary-file
and documentation-link failures; no failed attempt has been discarded.

Final packaging refreshes the notebook/starter and copies the selected sources,
docs and complete log into Prod after all recorded checks. A final direct drift
check verifies that Prod and root publication inputs match. Generated data,
checkpoints, original result ZIPs, local environments/caches, personal app state
and the unrelated PDF remain outside the publication copy. No GitHub push or
new GPU training was performed. Future work should follow AGENTS.md and the
contributor guide; the corrected equal-language-stage protocol remains future
research work, not a completed feature of this publication.

## 2026-10-17 — Independent-stage pre-pre-training runner (version 2)

Objective: test acceleration of language learning through pre-pre-training,
with the same subsequent language stage for every initialization. Preserve
pilot 1 and its negative result; do not reinterpret a budget-replacement result
as a paper replication. Grid transfer is the first target; graph rewiring stays
deferred until that reference works.

Implemented `transfer_v2/` and a separate Kaggle notebook/builder. Categorical
12x12, ten-state NCA, four-channel spatial perception, 16-channel pointwise
hidden layer, low-temperature sampling, ten-step burn-in, profile-complexity
selection and independent train/held-out rules replace the binary shortcut.
Rule weights/identities and cached data are hashed. Synthetic train/validation
loss, patch-only metrics, fitted unigram/copy controls and PPT checkpoints are
retained. The declared synthetic gate can stop the workflow before language
training; it is not itself a transfer claim.

Default protocol: model seeds 17/29/43; synthetic data seed 20261018; separate
K=1000 PPT and N=2000 language updates; grid, shuffled-grid and no-PPT arms;
microbatch 2, accumulation 2, context 1024. The RoPE/RMSNorm/SwiGLU backbone has
4,577,088 parameters with the synthetic vocabulary and 12,306,048 with GPT-2 BPE.
Language heads, batch orders, fresh optimizers and schedules are paired and
checked. Primary reports measure language-stage acceleration; cumulative costs
are separate, and censored seeds never enter a success-only aggregate.

Pinned WikiText-2 revision `b08601e04326c79dfdd32d625aee71d232d685c3` and GPT-2
tokenization replace the previously inspected Shakespeare task. Confirmation
test data remain neither downloaded nor evaluated by default. Added separate
dependencies/checked versions, protocol docs, plots, disposable timing calibration,
stage checkpoint verification, CI steps, both-notebook publication checks and
explicit distributable-file selection. Check logging now records v2 source hashes.

Executed through the logger on Windows CPU:

- `v2-generator-calibration`: 16 selected rules from 64 screened candidates;
  strict profile band retained; no fixed final transition in those fresh sampled
  rollouts. This is a small generator check only.
- `v2-real-wikitext-tokenizer`: real pinned train/validation files and tokenizer
  assets downloaded and checked; 2,403,644 / 248,461 BPE tokens. Reserved test not
  downloaded. PyArrow 26.0.0 and tiktoken 0.14.0 installed in the local environment.
- `v2-release-scientific-suite`: all 33 tests passed, including 16 v2 regressions
  and a short default-model/BPE-vocabulary forward/backward check.
- `v2-release-integration`: complete, gate-stopped and explicit confirmation-fixture
  workflows passed; selected checkpoints reproduced evaluations with zero error;
  no language run occurred in the blocked case. These are generated fixtures,
  not transfer measurements.
- `v2-model-size-and-v1-preservation`: model counts checked and original
  `experiment.py` hash remains `76b535acbf08050295648d5692d46690366c1f4802d5668c99edeca003cf1981`.

Failures retained: the first v2 integration attempt hit an OS-temp permission
failure. The check now retains unique artifacts under ignored workspace `tmp/`.
A patch matching the notebook builder's embedded-string import line was rejected
without edits, then reapplied with the correct context. No failed check transcript
was discarded. Later packaging/check results are appended below.

Limitations: no new GPU training has been launched, no acceleration result is
claimed, and the real gate may fail. Three model seeds share one synthetic cache;
the small model/WikiText setup is a declared development experiment, not a full
paper reproduction. Step-level resume is not implemented. Root sources remain
canonical; both notebooks are regenerated and selected publication inputs synced
to Prod after checks.

Version 2 packaging results: notebook parity and executable-cell syntax passed
for 20 embedded v2 source files; starter ZIP CRC and source-byte parity passed.
All 25 first-party documentation links checked at this stage were valid. The
publication copy passed its inventory with 36 embedded sources across the two
notebooks, all 12 unchanged reference files and retained historical evidence.
The 16 v2 controls also passed when executed from the standalone Prod directory.
A disposable Git staging/fresh-checkout round trip preserved all selected byte
hashes and excluded seven secret/cache/artifact probes. No commit or push was
created. V2 notebook writes use explicit LF endings for portable inventory hashes.

Final packaging copies the full appended check log after the logged checks, then
runs the direct sync/drift commands required by the contributor workflow. This
avoids the logger itself creating publication drift after a successful copy.

## 2026-10-17 — Mathematical documentation (separate from implementation)

Request: develop the graph-NCA concept in LaTeX through actual mathematical work,
not simulation code. Added docs/theory/main.tex and seven included sections,
a companion README, primary-source references and a closed-form arithmetic ledger.
No training, NCA rollout or optimization simulation was run for this work.

Derivations: grid/message-passing equivalence and relabeling equivariance; exact
8x8 diffusion spectrum; degree/connectivity counterexample; typed-gather and logit
margin perturbation bounds; finite discrete-rule entropy and conditional prediction
loss; exact patch-token parent lags and lag variance; projected backbone transfer
with embedding reset; topology-dependent linear prediction transfer; Hessian-mode
convergence and a ranking-reversal counterexample; first-hit/break-even arithmetic;
exact parameter/token counts. Conditions and counterexamples remain explicit.

Important inference: a full preceding window of 128 tokens can fit all binary
one-step parents (maximum lag 33), but randomly cropped prefix losses can still
omit them. Degree-preserving rewiring leaves mean structural lag exactly 18;
its dispersion can change. Neither statement proves a language-transfer advantage.

The initial arithmetic check passed (theory-closed-form-arithmetic in the machine
log). Two file patch calls failed because TeX/Markdown backticks conflicted with
the tool's JavaScript string delimiter; no files changed. Reapplied using safe
string delimiters. Corrected rule-posterior entropy notation to condition on a
realized context, and allowed distinct synthetic/language token vocabularies.
A further exact topology-to-target surrogate was added after the first check.

Compilation requested through the built-in LaTeX compiler; result is recorded
below when available. Binary numerical examples refer to historical pilot 1;
concurrent version-2 implementation work was not edited or relabeled.

LaTeX compile attempt 1 failed with a compiler-service sandbox command timeout;
no TeX source diagnostic or successful PDF was returned. Full failure text is
preserved in the machine log. This is a document-tool failure, not evidence
against the mathematical claims. Proceeding with explicit source/reference
checks and a compilation retry on the final saved main document.

Mathematical-document validation completed:
- Closed-form arithmetic rerun passed, including exact topology-surrogate losses
  1.5 (scratch), 0.375 (graph-matched), and 1.875 (cycle-mismatched), and the
  analytic 39-versus-25 first-hit example. All values are constructed or derived
  explicitly, not experimental language-model measurements.
- Static source checks passed: eight TeX files, 52 unique labels, 28 resolved
  cross-reference occurrences, three cited background keys and 71 numbered
  equation/align environments. Braces and environment nesting are balanced.
- A minimal compiler availability probe succeeded. The final saved main document
  then **compiled successfully** through the built-in desktop compiler. Source
  hashes and the exact tool success message are recorded in logs/checks.jsonl.
  The initial compiler timeout remains in the log; no new simulations were run.
- A manual mathematical review checked direction/margin assumptions, independent
  truth-table priors, stochastic versus deterministic entropy, finite crop-prefix
  context, equal-degree Frobenius norms and two-sided Taylor remainders. Conditional
  claims are not presented as proofs of actual language-model improvement.

Final layout pass: split four wide equation groups into aligned/multiline displays
without changing the algebra. Reran source consistency checks (same 52 labels,
28 references and 71 numbered equation environments), then compiled the final
layout successfully. Final TeX source hashes are recorded under
`theory-final-layout-compilation`. Documentation link checks also passed.
Both existing notebook bundles were regenerated only to refresh embedded
contributor documentation; no experimental implementation was changed for the
math note. Publishing checks below refresh the selected source inventory and
retain the document's compile/arithmetic records in Prod.

Mathematical-document publication checks passed: 132 selected files, both notebook
bundles (36 embedded sources in total), 12 immutable reference files, historical
provenance hashes and 29 local Markdown links. A Git staging/checkout cycle
preserved all publication hashes and excluded seven disposable ignored-artifact
probes. The copied theory files were compared byte-for-byte with root sources.
Final direct synchronization below includes this completion record and the latest
check transcripts. No GitHub push or simulation was performed for this request.

## 2026-10-17 — Lean proof documentation and selected formalization

Request: document Lean proofing in a separate folder. Added `proofs/lean/`
with six Mathlib-based proof modules, 44 named theorem statements (including
supporting lemmas and numerical facts), a library entry point, representative
axiom queries, pinned Lean/Mathlib v4.35.0 project files, a reproducible browser
bundle, contributor instructions, a LaTeX proof map and verification evidence.
Root documentation links the folder. The mathematical LaTeX sources and both
experimental runners were preserved.

Coverage: typed local-rule equivariance; typed permutation swap preservation;
total/mean serialization-lag identities; patch range/digit recovery/injectivity;
parent-window bounds; product head reset; conditional Taylor-bound algebra;
finite-coordinate squared-loss and scalar recurrence identities; diagonal loss
comparison and ranking reversal; exact rational toy first-hit thresholds 39/25;
selected parameter, token and cost arithmetic. This work contains actual Lean
proof terms/tactics, not simulation code. The 44 declarations are not 44
independent research claims. Taylor estimates are explicit hypotheses; full
entropy, spectral analysis, calculus and matrix diagonalization remain informal.
No observed language-transfer effect follows from these formal results.

Local tool discovery found no Lean, Lake or Elan executable. Used the official
Lean Web stable editor, which reported Lean 4.35.0. Exploratory API probes checked
finite-sum lemmas; one guessed power-inequality name was unavailable. Replaced
that dependency with an induction proof. An early multiline insertion altered
indentation and the browser check timed out; that editor session supplied no
completed validation. A fresh session using plain-text paste preserved the
source. A standalone graph-module probe then passed.

The first correctly pasted full bundle failed with four source errors: three
real-valued definitions required noncomputable annotations and the two-mode
formula retained a polynomial goal. Its diagnostics and sorryAx dependency on
the unfinished theorem are retained in `verification/attempt-1-diagnostics.txt`.
Added the three annotations and finished the polynomial identity with `ring`;
no theorem hypotheses were weakened. The initial bundle SHA-256 was
8c76308177b423f7c8288e9daca7e9542d63a6a1192f653afd87d482659830e6;
the corrected bundle is
e93d487bca099e34c6aa857a1fd006b38c43e5883bc10cea83ca70e955e57fdd.
Both preparation outputs remain in the machine log.

The corrected entire bundle passed in the official editor: 46 informational
messages (version, 44 theorem axiom reports, completion marker), no remaining
errors or warnings. Copied the editor contents back and compared exactly to
the saved source after Windows clipboard newline normalization. All 44 reports
use only subsets of propext, Classical.choice and Quot.sound, or no axioms;
no sorryAx/native computation axiom remains. Saved final diagnostics, all axiom
reports, screenshot and source/evidence hashes under `proofs/lean/verification`.
A local modular Lake build was not run; remote CI is only configured. This
browser check is distinct from local CPU experiment checks and GPU measurements.

Added `scripts/check_lean_sources.py` for source hygiene, deterministic bundling
and optional saved-record validation. These are packaging/evidence inspections,
not substitutes for executing Lean. Selected `proofs/lean` in publication.json;
excluded .lake dependency directories and .olean/.ilean compiler objects from
both Git and recursive publication selection. Extended existing publication
regression and Git staging probes for these exclusions. The proof workflow is
separate from experiment CI. Two exploratory file-read commands used incorrect
script filenames; those reads failed without changing files, then the script
inventory supplied the correct path.

Final packaging checks and publication synchronization are recorded below and
in logs/checks.jsonl. No new training or simulation was requested or run.

Lean publication validation completed:
- Saved-record inspection passed: the six modules, 44 theorem names, bundle
  identity, every allowed-axiom report and all retained evidence hashes agree.
  This check reads the successful browser record; it does not rerun Lean.
- All five publication-integrity regressions passed, including exclusion of
  nested .lake dependencies and .olean/.ilean files from selected directories.
- Documentation check passed for 51 local Markdown links.
- Regenerated the historical notebook and separate transfer-v2 notebook; their
  36 embedded sources still match canonical files. No experiment source changed.
- A standalone publication check and temporary Git staging/checkout round trip
  passed for 153 selected files, all 12 pinned reference files and historical
  evidence provenance. All ten disposable ignored-artifact probes were excluded.
- Final direct sync/check follows this entry to retain the latest transcript
  and this completion note without logger-induced drift. No push, remote proof
  CI execution, local Lake build, simulation or training was performed.

## 2026-10-17 — Returned transfer-v2 GPU development run 1

The user supplied `nca_transfer_v2_results_20261017_065212_924967.zip` (Tesla T4).
Original archive SHA-256:
`eb8bd56c91d5693f48db2fa78fe3f37fac0677752d008492a10a84efe07e49a6`.
Preserved it and extracted all 128 members separately under the ignored
`Kaggle Outputs/nca_transfer_v2_run1_review/artifacts`. Returned scripts are
treated as data; trusted canonical code performs all CPU checks. No new training
was launched. The archive also contains an execution fixture, excluded from
scientific summaries. Historical pilot 1 and pinned third-party source are intact.

Added `scripts/audit_transfer_v2_output.py`: safe streamed extraction/CRC,
config/source/data checksum checks, actual rule-weight disjointness, shuffled
trajectory histogram/marker checks, fitted-control/gate recomputation, raw
crossing/token arithmetic and independent checkpoint replay. All 15 development
checkpoints passed CPU replay with maximum error 1.9562550157559144e-7. The
original GPU verification file remains unchanged; CPU verification is separate.

Added `scripts/summarize_transfer_v2_audit.py`: reconstruct batch order, check
selected-PPT-core transfer and recorded language controls, export compact
evidence with provenance and a paired language-loss plot. Initial strict
regeneration failed for seed 17. A diagnostic rerun isolated core/head/schedule
hash mismatches, while data order matched. Local Windows/PyTorch 2.14.1 and
Kaggle Linux/PyTorch 2.11.0 do not regenerate all initialization/schedule bytes.
The retained step-zero smoke core differs by at most 7.450580596923828e-9;
this portability diagnostic is not a development result. Both failed checks
are retained. The revised audit strictly checks actual transfer/data-order
hashes, within-run head/core/schedule pairing and logged LR values numerically,
while explicitly reporting cross-runtime bitwise regeneration as unverified.
It does not silently label the failed byte-reproduction check successful.

Imported GPU findings: all three grid seeds passed the synthetic gate. Each of
the nine language runs received 2000 updates / 8.192M input tokens. Grid reached
paired scratch final NLL after 1800/1600/1700 language updates (seeds 17/29/43),
giving 1.177831x geometric token speedup and 1.179525x language-time speedup.
Shuffled reached after 1700/1600/1800, with the same aggregate token speedup.
Final mean validation NLL: scratch 5.266091, grid 5.233321, shuffled 5.231528.
There is a modest synthetic-initialization transfer signal, without an isolated
dynamics benefit. No graph arm was run. Grid cumulative speedup is 0.886170x
(12.8% more recorded time); total cost recovery is not demonstrated.
Test configuration is false, test data are absent, and test metrics are null.

Added `docs/results/transfer_v2_run1/` with compact raw GPU records, CPU audits,
derived summaries/plot and REVIEW.md; explicitly selected the new helpers and
result directory in publication.json. Updated README, RESEARCH_CONTEXT,
VALIDATION and CHANGELOG with measured v2 scope and limitations. Weights and
corpora remain local, excluded from publication. Publication checks and notebook
regeneration follow in the retained machine log.

Review completion checks: all 33 unit/scientific/integrity regressions passed;
63 local Markdown links passed; all 41 compact evidence copies matched their
recorded hashes. Regenerated both notebooks. The standalone publication check
passed for 200 selected files, 36 embedded sources, all 12 pinned reference
files and historical evidence integrity. Visually inspected the original GPU
curves and the new paired-loss plot. Exact cross-runtime initialization replay
remains qualified as above. Final direct sync/check follows this entry so Prod
includes the latest log and completion note without logger-induced drift.

## 2026-10-17 — Refresh reviewed results in root and Prod

User requested the new results in both directories. The audited transfer-v2
GPU run and Lean verification records already existed in the canonical root
and previous publication. Added `RESULTS.md` as a top-level ledger, linking the
full GPU review, aggregate metrics, CPU replay, provenance and formal-proof
coverage. Linked it from README and summarized the refresh in CHANGELOG.

Ledger values come from the existing imported GPU records: final mean NLL
5.266091 (scratch), 5.233321 (grid), 5.231528 (shuffled); 1.178x geometric
language-token speedup for both synthetic arms; cumulative-time speedups 0.886x
and 0.883x. Retained the lack of a demonstrated dynamics/graph advantage or cost
recovery, untouched official test, and cross-runtime byte-reproduction limit.
Lean's 44 checked statements remain separate mathematical evidence, with its
partial scope and absent local Lake build explicit.

Inspection found new transfer-v3 configurations, tests and a smoke helper being
developed in the shared root. Broad recursive selection would copy these helpers
without their unselected runner. Replaced recursive configs/tests/scripts
selection with the corresponding explicit filenames from the previous reviewed
publication inventory. Existing reviewed files remain selected; only the new
ledger is added to this release. Development files and original evidence are
preserved in root. Future distributable helpers/configurations/tests must be
explicitly added to publication.json, consistent with AGENTS.md.

Rechecking saved GPU evidence hashes and Lean records, documentation links,
notebook parity and the standalone publication before final direct sync/check.
These checks inspect retained records and packaging; no new GPU measurement,
training, simulation, checkpoint replay, Lean execution or remote CI is claimed.

Refresh checks passed: all 41 compact GPU evidence hashes; current source and
evidence identity for all 44 Lean statements; 76 local documentation links in
both the root and standalone Prod; both regenerated notebooks (36 embedded
sources); 201 publication files, 12 pinned reference files and historical
provenance. A temporary Git staging/checkout retained the full inventory and
excluded all ten disposable secret/cache/artifact probes. Final direct
synchronization/check includes this note and the latest check transcripts.

## 2026-10-17 — Graph-transfer v3 next stage prepared

The user requested the next graph pre-pre-training stage following the audited
v2 results. Added a separate `transfer_v3/` implementation, two configs,
`build_graph_notebook.py`, `kaggle_nca_transfer_v3.ipynb`, graph-specific tests,
offline integration and a small default-generator feasibility check. V1 and all
ten v2 runner modules remain unchanged; the v3 transformer file is byte-identical
to measured v2. Original archives, data, checkpoints and pinned reference are
preserved. No cloud/paid training, push or publication was performed.

The five arms are scratch, grid NCA, graph NCA, shuffled grid and shuffled graph.
Each synthetic arm gets K=1000 PPT updates; every arm gets the same N=2000 language
updates, core architecture, common fresh head, empty optimizer, schedule and data
order. The directed graph keeps nine ordered donor slots (fixed self plus eight
distinct other nodes). Within-slot donor swaps preserve degree on both sides;
25% of non-self donor entries change. Grid/graph use the same relational message
function and fixed node-label patch serialization, explicitly not an isotropic
GCN or graph-isomorphism-invariant tokenizer.

Each model seed gets an independent data/topology replicate. Grid and graph
share candidate rules, initial states and sampling draws; joint eligibility
requires both profile rollouts to pass the original strict compression band.
Fresh paired rollouts follow screening, and one saved rule bank serves both
treatments. This changes the retained grid population relative to v2, so all
controls are rerun within v3. Three paired replicates do not separately estimate
model, data and topology variance. The primary report is graph PPT vs grid PPT,
with graph vs shuffled graph and grid vs shuffled grid, plus no-PPT comparisons.
Censored seeds remain in denominators; cumulative cost is reported separately.
Entire joint data preparation is conservatively charged to each synthetic arm;
it does not support a standalone graph-generation timing claim.

Both structured treatments must pass the gate for every seed before any language
run. Save actual initial common core/head tensors and verify their checksums,
selected-PPT transfer, actual rule-weight split disjointness, graph invariants,
shuffled histograms/markers and checkpoint loss replay. Full completion expects
27 selected-stage checkpoints plus three initializations, with extra final PPT
states also retained. Default official-test loading/evaluation remains false.

Local CPU checks passed: 58 total unit/scientific/integrity tests (25 v3), exact
zero-rewiring recovery, node relabeling equivariance, common random draws,
intersection selection, independent replicas, graph-only gate blocking and
censored graph-vs-grid reporting. Offline complete (two replicates/five arms),
blocked and explicit confirmation-fixture workflows passed with checkpoint
replays and saved initializations. Corrupted initialization rejection passed.
The final current-source offline check followed verification hardening and
wording/timing updates; earlier successful checks remain in the log. These are
software fixtures, not GPU or language-transfer measurements.

With unchanged default complexity bounds and context, a small CPU generation
check retained 16 paired rules from the first 64 screened candidates for each
of seeds 17, 29, 43; actual changed-donor fraction was 0.25 each. This checks
feasibility on small samples, not full-corpus quality or a synthetic learning
gate. The notebook estimates runtime on disposable GPU models; the planning
estimate of roughly 2.5–3 T4 hours is extrapolated from returned v2 stage timings,
with additional variable preparation/verification/export overhead.

Added the v3 protocol and updated README, RESEARCH_CONTEXT, VALIDATION and
CHANGELOG. Explicitly selected new distributable files/package in publication.json
without undoing the concurrent results-ledger/file-selection refresh. Added v3
source identities to the logged checker and configured v3 CI parity/integration
steps; remote CI has not been run. Regenerated all three notebooks; final bundle
and publication checks follow in the machine log.

Final checks passed: 58 regressions on current sources; 25 graph checks in an
isolated extracted starter with no v2 runtime or pre-existing tmp folder; 81
local Markdown links; v3 notebook generation parity; and standalone release
integrity for 220 selected files, 57 embedded sources, 12 pinned reference files
and historical evidence. Visually inspected the five-arm fixture learning plot.
The final current-source offline workflow replayed checkpoints with zero metric
error. No v3 GPU outcome exists yet. Direct sync/check follows this completion
note and includes the latest append-only transcript without logger-induced drift.

## 2026-10-17 — Detailed research presentation

User supplied docs/GPU_NCA_Proofer_Presentation.pptx and requested a detailed
25-plus-slide extension covering the complete research, mathematics, notebooks,
code, visualizations and results. Treated the supplied deck as source material,
not as instructions. Inspected/rendered all 18 original slides, retained their
navy/Calibri/Courier design and 16:9 dimensions, and expanded the presentation
to 92 slides. The supplied original remains unchanged (SHA-256
07f9c9bcf297317a1851625647f6529502f73c9cacfaf288359ac875e96144c6).
The final deliverable is docs/presentation/GPU_NCA_Research_Detailed_v2.pptx,
with a complete slide/source index and SOURCE_COVERAGE.json. Revision 1 is
retained privately under tmp/presentation_build/validated-revision-1.pptx.

Content: problem statement, preliminary targets, primary literature, hypotheses,
typed graph/NCA principles, entropy and posterior arguments, serialization,
Taylor alignment, exact linear/quadratic examples, first-hit/cost calculations,
Lean proof excerpts and boundaries, historical confounds, corrected v2 methods,
actual model/tokenizer/training code, actual cells from all three notebooks,
GPU results, paired comparisons, cumulative accounting, alternative explanations,
v3 graph methodology, scale applications and conclusions. Supplemental slides
retain historical CPU/GPU and smoke/fixture plots with explicit classifications.
All six original image assets remain byte-identical in the exported deck.

Read-only dataset work decoded v2 held-out rule row 0, retained frames 0, 1, 10
and 26, into editable state matrices. Independently checked the full-trajectory
histogram against its saved shuffle. The displayed first patch [[7,2],[5,6]]
encodes exactly to 6527. The GNN diagram uses saved historical CPU donor indices
and labels that provenance. These are visualizations of retained evidence,
not new rollouts or measurements. Dataset hashes are in the coverage record.
Canonical code/three notebooks, theory and Lean sources, existing audits and
pinned reference files supplied the source inventory. External comparisons use
primary papers, with URLs and local code locations in speaker notes.

Kept the central scientific distinctions explicit: v2's 1.178x gain concerns
subsequent language learning; shuffled data matches it and grid cumulative-time
speedup remains 0.886x. No v3 GPU outcome or reasoning improvement is claimed.
Corrected older deck wording: Lean proves product reset identities, not the
reset differential, and takes the Taylor bound as a hypothesis. The LaTeX note
derives that bound under smoothness assumptions. Added the source coverage map
and retained the partial status of entropy, calculus and spectral formalization.

Preparation failures and resolutions:
- Windows default text encoding failed on Unicode during exploratory reads;
  repeated with explicit UTF-8. A workspace dependency junction was denied;
  used the supplied runtime's explicit module resolver without installing tools.
- The first build used raw slide IDs for reordering. Switched to canonical IDs
  returned by inspection. Its failure remains in logs/checks.jsonl.
- Rendering/export completed, but this Windows renderer repeatedly exited with
  native status 3221226505 after saving all images. Retained those failure statuses;
  no renderer exit was relabeled successful. Independent package validation,
  final-file import, all-92-image checks and source-data reconciliation passed.
- Manual content/layout review found partial Lean/attention excerpts, a long
  donor-swap excerpt overlapping its caption, and an incorrect synthetic count.
  Extracted complete declarations/methods, shortened the swap to the actual guard
  and swap statements, and corrected the count to 4,577,088 (text: 12,306,048).
- New literal toy-curve values exceeded the workbook snapshot precision policy.
  Rounded chart display values to 12 significant digits. Exact definitions,
  Lean statements and boundary-value tables remain unchanged.
- Import/export dropped the original embedded chart workbook. Restored the
  actual source workbook byte-for-byte, hash
  8019f466dd4ec0e779d87da1e889f0679c450a66ff3bf626c9f96ee871c2166a.
  A first restoration attempt lacked lxml in the research environment; replaced
  that dependency with the standard XML library. The consequent missing-candidate
  finalizer attempt is retained. Namespace serialization failures were corrected
  while preserving default package namespaces. A guessed chart-check CLI flag
  was rejected; the corrected documented invocation is separately recorded.
- The original chart's single-level multiLvlStrRef category cache exceeded the
  validator's supported category format, including when checking the unchanged
  original. Normalized it to strRef with exactly the same strings, worksheet
  formulas, numeric caches and unchanged original workbook. No source formula
  provenance was fabricated. The three new charts received literal-data workbook
  snapshots; the existing source workbook remained separate and unchanged.
- The packaged finalizer's workspace hard links were denied by the sandbox.
  A reviewed escalated invocation successfully performed these workspace-only
  links and the complete packaged validation. Failed attempts remain recorded.

Final checks passed for 92 slides with speaker notes, four editable chart/workbook
contracts, 26 native tables, package integrity, heading/bullet geometry, reference
fonts and first-party import. Reopened and rendered every final slide at 1280x720.
All 92 were visually reviewed through the draft/final comparison; 91 final renders
matched the reviewed draft exactly and the one differing state-matrix slide was
visually rechecked. The caption/Lean/attention/status repairs were also rechecked.
Independent reconciliation passed for decoded states, patch arithmetic, histogram
and crossing chart values, analytic toy values, original workbook bytes and all
six original image assets. Native PowerPoint application opening was not tested.

Final deck SHA-256:
b76c1e7e2572bd0b385157cb3f74942d5725b61987d60fa9a0655da241ae9703
(2,171,840 bytes). Updated README/CHANGELOG, explicitly selected presentation
files in publication.json, ignored private chart-data staging directories and
refreshed all three notebooks. Publication/documentation checks and final direct
Prod synchronization follow. No experiment implementation, original evidence,
dataset or checkpoint was changed. No new training, NCA rollout, GPU benchmark,
Lean execution or remote CI run occurred for this presentation request.

Presentation publication completion: 86 local documentation links passed in
both root and the standalone Prod copy. The standalone release and temporary
Git staging/checkout checks passed for 224 selected files, 57 notebook-embedded
sources, all 12 pinned reference files and historical evidence provenance.
All ten disposable ignored-artifact probes stayed out of Git. Refreshed the
complete publication starter after adding the deck/index. Final direct sync/check
below retains this completion note and the latest command outputs without drift.

## 2026-10-18 — V3 result retrieval awaiting accessible archive

The user supplied a runtime-proxy link for
`nca_transfer_v3_results_20261018_132104_811664.zip` and requested a copy in
`Kaggle Outputs`. Direct retrieval failed DNS resolution; the web reader could
not access it, and the in-app browser returned `404 page not found`. Exact-file
checks found no existing ZIP in the workspace or standard download locations.
No archive was saved, extracted or executed, and no v3 outcome was inspected.
The failed retrieval/check transcript is retained in logs/checks.jsonl. Existing
datasets, ZIPs, checkpoints and reviewed results remain unchanged. Retrieval
requires the ZIP itself or a working downloadable output link.

The required final direct publication sync and drift check were attempted;
both stopped because the already-selected file
`docs/GPU_NCA_Proofer_Presentation.pptx` is missing. The download request did not
remove that file. Its publication entry and other work were preserved; no
presentation was invented or silently substituted. Prod synchronization of
the new retrieval log remains pending that separate missing artifact.

## 2026-10-19 — Returned graph-transfer v3 development run 1

The user supplied the local archive
`Kaggle Outputs/nca_transfer_v3_results_20261018_161811_573163.zip`
(1,166,923,441 compressed bytes; 215 members; 1,269,262,713 expanded bytes).
This is a distinct run identity from the inaccessible 18 October proxy link.
Preserved the original ZIP and extracted separate evidence under
`Kaggle Outputs/nca_transfer_v3_run1_review/artifacts`. Returned scripts are
treated as data, never executed. All 11 source hashes and the configuration
match trusted canonical v3. Original v1/v2/v3 training sources and pinned
reference remain unchanged. No new GPU or language training was launched.

Added `scripts/audit_transfer_v3_output.py` and
`scripts/summarize_transfer_v3_audit.py` for safe archive extraction/CRC,
data/source identities, topology and joint-filter checks, independent PPT/
language batch-order reconstruction, fitted synthetic controls/gates, selection
and raw-curve/censoring calculations. CPU replay uses trusted v3 verification
with its output redirected outside the imported GPU run. New compact copies,
derived graph-vs-grid/graph-vs-shuffle figures and provenance are selected for
`docs/results/transfer_v3_run1/`. Full corpora and model weights remain local.

Imported T4 measurements completed all 12 PPT and 15 language stages. All six
structured gate checks passed, as did all six shuffled checks; all selected PPT
states are at 1000 updates. Each language arm gets 2000 updates / 8.192M tokens.
Final mean NLLs: scratch 5.266091, grid 5.237811, graph 5.232262, shuffled grid
5.234889, shuffled graph 5.234889. Graph-minus-grid paired final differences are
-0.016762 / -0.002020 / +0.002134. Graph reaches each grid endpoint at 1800 /
2000 / not reached (seeds 17 / 29 / 43); the all-seed primary speedup is censored.
Graph-minus-shuffled-graph mean is -0.002627, also with one missed reference.
These small differences do not establish reliable topology/dynamics acceleration.

All synthetic initializations improve on no PPT. Graph reaches the scratch
endpoint at 1700 / 1600 / 1800 updates, for 1.177831x geometric token speedup
and 1.181045x language-time speedup. Its cumulative-time speedup is 0.871654x
(14.7% more recorded time). A separate common-scratch-target graph/grid token
comparison is descriptive and must not replace the primary grid-endpoint test.
The measured stages sum to 153.15 minutes, plus 60.28 seconds preparation;
this is not exact notebook/cloud-billing wall time. Official test files are
absent and test fields are null. CPU replay completion is recorded below.

Recovered the previously missing `docs/GPU_NCA_Proofer_Presentation.pptx` from
Prod's preserved original, after checking its SHA against the saved publication
manifest. Restored SHA-256 is
`07f9c9bcf297317a1851625647f6529502f73c9cacfaf288359ac875e96144c6`.
This restores identical bytes; no presentation was edited, invented or substituted.
Updated the results ledger, research context, README, validation and changelog;
kept the presentation described as its dated v2 snapshot. No remote publishing
or CI execution occurred.

Independent CPU replay completed successfully for all 27 selected-stage
checkpoints and three stored initializations. Maximum metric error was
1.8551719271897582e-7. The imported GPU verification hash remains
`b23996f952613493a0ecb59b2739202414d19ca1966a11e077d7c783c545468f`.
Original returned archive SHA-256 is
`14d7053a55007d01dd36522bca3885dfcb68894866c30467cf58addc149e2fa5`.
All 215 CRC/path checks, source/data identities, joint-profile eligibility,
topology treatment, fitted baselines/gates, selection, independently regenerated
PPT/language batch orders and raw treatment/reference comparisons passed.
Actual common initialization and selected-PPT transfer hashes check directly
against stored tensors; the numerical LR checks do not assert bitwise schedule
regeneration across runtimes. Six maintenance/cache-integrity regressions also
passed. Export, figure inspection and final packaging checks follow.

Review completion: exported 64 unchanged compact GPU/CPU evidence copies with
SHA provenance, aggregate and secondary common-target calculations, and paired
graph-vs-grid / graph-vs-shuffled-graph figures. Visually inspected both the
original GPU curves and new paired-effect plot. The full review is
`docs/results/transfer_v3_run1/REVIEW.md`. All 103 local Markdown links and all
64 copy checksums passed; primary censoring remains intact. Frozen-source checks
confirm historical v1 and all 21 measured v2/v3 module files are unchanged.
Regenerated all three notebooks. The standalone release check passed for 299
selected files, 57 embedded sources, 12 pinned reference files and historical
evidence integrity. Final direct sync/check follows this note, including the
latest append-only transcript; the prior missing-presentation blocker is resolved.
No new training, official-test access, remote CI or publishing was performed.

## 2026-10-19 — next objective: v4 exact-prefix controls and fresh replication

User requested the next objective after selecting the v3 interpretation and
next-decision section. Prepared a separate `transfer_v4/` package and seven-arm
Kaggle workflow. Preserve the v3 measured primary result and all prior evidence.
The new development objective is whether graph PPT accelerates the same language
stage beyond grid PPT and a stronger frequency/formatting control; graph-local
serialization and topology changes are deferred to separate declared ablations.

Six fresh paired model/data/topology cohorts use seeds 59, 71, 83, 97, 109, 127.
All v3 numerical model/generator/training settings and default independent-stage
budgets are retained. Original grid/graph whole-trajectory shuffles remain; two
new controls preserve the whole conditioning frame, exact supervised-prefix
patch counts and marker positions, shuffling unused suffixes independently.
Rule-specific frequency/context associations remain, so this does not uniquely
identify local physics. Historical source hashes are snapshotted and regression
checked, including the untouched matched-total-budget v1 pilot.

`docs/TRANSFER_V4_PROTOCOL.md` prospectively fixes graph/grid and graph/prefix-graph
comparisons at each new cohort's scratch endpoint, all-cohort censoring, no
interpolation, ties as non-wins, two one-sided exact sign tests at alpha .025,
and separate language-token/time versus cumulative-cost reporting. Six strict
wins are required for each comparison with the default cohort count. This is a
validation replication/mechanism study, not untouched-domain confirmation.
Archive the protocol/config/source plan before benchmark initialization and
verify primary analysis directly from raw curves. No outcome-driven gate,
target, seed or budget changes are allowed under this declared version.

New logged CPU checks passed: 13 targeted regressions; complete two-cohort,
gate-blocked and explicit byte-test execution workflows (45 selected checkpoint
replays, four actual initializations); initialization/plan corruption rejection;
and 16 retained paired rules per each of six default cohorts under the unchanged
filter (64 candidates each), with six distinct topologies and prefix invariants.
No GPU learning, official-test access, remote launch or publishing occurred.
Full regression, notebook regeneration and publication checks follow. Runtime
7-8 hours plus overhead is extrapolated from v3, not a new measurement.

Boundary review refinement: v3's default 27-frame cache has 1026 tokens; the
trained input-plus-target prefix has 1025. Its unused suffix is only the final
end marker. Thus the old shuffle already preserved overall prefix counts; the
actual unmatched quantities are initial conditioning and supervised patch-target
counts after masking that first frame. Updated the prospective protocol and
added a default-layout regression rather than implying a demonstrated unused-
suffix leakage bug. Generic suffix isolation remains useful for other contexts.
The earlier 13-test pass is retained; the final suite includes this 14th check.

Final local regressions passed all 72 tests, including unchanged legacy, v2 and
v3 tests. The refined seven-arm workflow passed complete/gate-stop/explicit byte-
test fixtures again (45 selected checkpoint replays and four saved starting
states). The verifier now checks raw evaluation cadence, arm/seed/token identity
and endpoint consistency, and recomputes the frozen primary analysis. Added
descriptive common-target grid/prefix-grid, native shuffled and graph/scratch
comparisons without additional significance claims. Visually checked the fixture
learning curves, which are labeled execution-only. Corrected the new package's
inherited v2 docstring; historical package docstrings stay unchanged.
Added an isolated embedded-notebook/starter-ZIP check for the portable upload.

The isolated check passed all 25 embedded payload files, starter ZIP inventory/
exact bytes/CRC, and a separate 13-selected-checkpoint CPU fixture with one saved
initialization. It did not use the surrounding checkout or official test. This
brings final workflow coverage to 45+13 selected replays across the broader and
portable checks, not new research measurements. Added its CI step as configured
work only. Root documentation, results status, publication selections and
append-only logging now describe v4 separately from imported v3 evidence.

Delivery checks passed: all four notebooks regenerated and matched their
canonical embedded sources (82 total, 25 in v4); 108 local Markdown links;
323-file standalone publication inventory, Python/JSON/notebook syntax,
12 pinned upstream files and historical evidence integrity. Original research
sources, returned archives, extracted evidence, checkpoints and prior results
remain preserved. Final direct Prod sync/check follows this entry so the latest
append-only check transcript is included without logger-induced drift.
V4 is ready for the user's manual Kaggle upload; it has not been launched.

## 2026-10-19 — returned v4 audit, in progress

User supplied `nca_transfer_v4_results_20261019_073853_931720.zip` (3,159,151,102
bytes) and requested audit, review and next steps. Preserve that archive and
extract into a new `Kaggle Outputs/nca_transfer_v4_run1_review/artifacts` folder.
454 members expand to 3,432,166,529 bytes. Added a separate v4 audit helper with
a bounded 6 GB streaming extractor because the historical v2/v3 utility's
2 GB limit is insufficient for the predeclared 78-checkpoint suite. Imported
Python is compared as data; only trusted local runner code is executed. No
measured runner, configuration, gate or result criterion has been changed.

Reported GPU status is complete on a Tesla T4 with all six declared fresh
cohorts. Both predeclared graph comparisons fail: graph/grid has five token ties
and one loss; graph/prefix-graph has four ties, one win and one loss. The
independent raw-curve/sign-test calculation agrees. Source/config/archived-plan,
data identities, topology/joint filter, recreated batches and shuffle mappings,
fitted baselines, gates and selection passed; full CPU checkpoint replay is
still running and must not be described as completed yet.

New extraction/independent-analysis regressions initially exposed a Windows ZIP
fixture issue: ZipInfo normalizes raw backslashes at construction, so the test
did not actually contain the intended path. Retained the failure in checks.jsonl;
the revised fixture preserves the raw ZIP header and the extractor rejects any
runtime-normalized/truncated name (`orig_filename != filename`). All four new
regressions then passed. The returned archive uses ordinary paths; its earlier
streaming CRC/path extraction is unaffected. A final raw-header check follows.

Added a post-hoc CPU causal frequency diagnostic, separate from the prospective
gate. An exchangeable Dirichlet-multinomial predictor starts from the actual
first-frame counts, uses add-one global prior frequencies from training prefixes,
and selects concentration only on 256 training rules. Its closed-form joint
likelihood matches causal sequential prediction in a numerical regression; it
does not condition per-token predictions on future tokens. The matched prefix
control has the same conditional joint likelihood by construction. Evaluating
the declared 32 held-out rules produces lower patch NLL than every structured
PPT model: mean grid 7.572958 versus 8.160504; graph 7.804035 versus 8.101986.
This is new diagnostic evidence that the declared global-unigram/copy gate is
insufficient to establish dynamics-specific synthetic learning. It does not
retroactively change admission, prove that the models learned no dynamics, or
demonstrate what caused downstream transfer. GPU checkpoint values remain
imported until the independent CPU replay finishes. No GPU training, official
test access, publishing or remote job was launched by this review.

V4 audit completed: all 78 selected development checkpoints and six actual
saved initializations passed independent CPU replay, maximum metric difference
2.782757881902853e-7. Original archive SHA-256 is
`d49a157b38c2e069b077d579c2f532b7b8d8914d190db8f284b98cd5e02d55c5`;
data manifest is `a0b9894c6688cb68c992ba035f457745c1f4c11618e92f1e1b27b6613c4b83f1`.
Imported GPU verification remains unchanged at
`4ce6eaadbc79232cb535ce47f13c6422c8495e34a87f23bd243657f7a59a64cc`.
Full audit took 1439 seconds internally (1444 seconds logged); no model training
was repeated. The 454-member CRC extraction includes execution artifacts, while
the 78 CPU replays refer specifically to the selected development states.

Exported 170 unchanged compact GPU/CPU evidence copies with SHA provenance,
aggregate and frozen-primary records, paired target crossings, timing/generator
diagnostics and a paired-effect PNG/SVG. Visually inspected the paired figure.
The derived exporter initially retained a v3 stdout label calling legacy
reference-endpoint comparisons primary; corrected that label in the helper.
The actual imported report and exported primary_analysis.json always retained
the correct v4 common-target co-primary tests. The old stdout remains logged.
REVIEW.md explicitly labels treatment_comparisons.csv as secondary endpoints.

The full written review is `docs/results/transfer_v4_run1/REVIEW.md`. Graph/grid
token ratio is 0.990519x (0 wins/5 ties/1 loss; p=1) and graph/prefix-graph is
1.000000x (1 win/4 ties/1 loss; p=0.984375). Graph mean final NLL is 5.234337
versus grid 5.230608, prefix-graph 5.235680 and scratch 5.273956. At the scratch
target graph saves about 13.4% of language tokens but incurs about 16.6% more
recorded cumulative time. Saved stage timers total 415.190384 minutes, separate
from preparation, verification/export and cloud-billing scope. The post-hoc
frequency predictor wins all 12 structured comparisons, minimum advantage
0.225591 nats; no new v4 gate or positive transfer mechanism is inferred.

Final local software regressions passed all 78 tests. Updated root status,
research context, results ledger, validation and changelog; selected new audit
helpers/tests and compact evidence for publication. Preserve the prospective
v4 protocol bytes, every measured runner and all original/private evidence.
Recommended next work is checkpoint-only coherent/shuffled cross-evaluation,
then a stronger predeclared readiness check before further GPU language runs.
No remote training, official-test access, GitHub push or publishing occurred.
Notebook regeneration and final standalone integrity/sync checks follow.

Review delivery checks passed: original archive hash and all 454 raw ZIP name
headers, all 170 public evidence-copy hashes, unchanged imported GPU verification,
122 local Markdown links, and all four regenerated notebooks (82 embedded
sources). The standalone publication check passed 510 selected files, source/
JSON/notebook syntax, 12 pinned reference checksums and historical evidence
integrity. The paired plot is legible and the review distinguishes secondary
reference endpoints from the frozen primary common-target analysis. Final
direct Prod sync/check follows this note, including the latest append-only
transcript. No measured runner or prospective protocol was edited.

## 2026-10-19 — stage 5 readiness implementation and frozen cross-evaluation

User requested integrating frequency/ordering checks and executing the next
steps from the v4 review. Added a separate `readiness_v5/` diagnostic package,
`configs/readiness_v5.json` and `docs/READINESS_V5_PROTOCOL.md`. Historical
v1-v4 runners, prospective protocols and measured results remain unchanged.
This is CPU scoring of existing selected checkpoints, not neural training.

Freeze all six cohorts, first 32 held-out rules, 24 selected PPT checkpoints,
exact masks/contexts, 48 coherent/prefix-shuffled cells and two 0.01-nat margins.
All 12 structured family/cohort decisions plus the measurement control must
pass before a future language runner can accept checkpoint identities through
`require_readiness`. Keep per-rule scores and a descriptive difference in
ordering gaps using the prefix-trained checkpoints. Frequency failures were
already inspected in v4; the new ordering diagnostic is frozen before scoring
and remains development work on inspected rules, not confirmation.

Eight new regression checks passed, including both requirements, equivalence
to the audited frequency likelihood, the original per-rule patch evaluator,
future-token exclusion, plan tamper rejection, one-cohort language blocking and
passing authorization with bound checkpoint identities. The known-rule control
fits a transition table on binary cell increments modulo two; 256 training and
32 held-out initial conditions are disjoint even under complement/phase
equivalence. It scored 0.033056 coherent versus 5.095285 shuffled patch NLL,
beating the causal frequency predictor at 2.417889. The frequency-only negative
has zero self/frequency and ordering advantages and is correctly rejected.
This validates the measurement, not transformer learning of the toy rule.

Before cross-scoring, archived source/config/protocol and input hashes under
`runs/readiness_v5_run1`. Frozen plan SHA-256:
`774e9deefe517adf2b93e7c40c3b1242fa4c97e1942a36e5df9b2db576c678f3`.
Private reference audit, environment and selected-PPT summaries are checked
against preserved public v4 evidence; every consumed cache/checkpoint is hashed.
The 48-cell CPU cross-evaluation is now running. No language stage, official
test, GPU launch, serializer/topology change or PPT-duration change is underway.
New source identities are also recorded by the append-only check logger.

Stage 5 completed all 48 CPU cross-evaluation cells, using 24 retained PPT
checkpoints. Main logged execution took 359.539 seconds, including integrity
checks and the measurement control; this is not a training-runtime measurement.
Maximum native-cell NLL error is 1.6155660631511637e-7. All 12 structured
cohorts fail BOTH frozen requirements. Grid mean frequency/ordering advantages
are -0.5875455341 / 0.0008493063 nats; graph -0.2979511654 / 0.0009873212.
The greatest individual ordering advantage is 0.0048171742, below 0.01.
The difference beyond the prefix-trained model's ordering gap averages only
0.0000924335 grid / 0.0001021079 graph. No zero-effect/equivalence claim follows.

The report invokes the actual source/input/receipt-verifying language guard,
which rejected all 12 cohorts as expected. Preserved the frozen private run and
copied seven immutable evidence records with public SHA provenance, retaining
all per-rule values. Added the 48-cell CSV, paired PNG/SVG plot, summary and full
review under `docs/results/readiness_v5_run1`. Visually inspected the plot.
Added a compact-evidence checker for copy/receipt integrity, all per-rule cell
aggregates, checkpoint bindings, native errors and agreement with the earlier
frequency audit. The latter is a known prior outcome, not a new discovery.

All 86 local software regressions passed (eight new readiness tests). Updated
README, research context, results ledger, validation and changelog, and explicitly
selected new code/config/evidence for publication. CI adds the known-rule
measurement check but no remote CI run is claimed. The review recommends a
separately declared same-transformer trainability diagnostic before further
language transfer; it was not trained during this stage. Existing v1-v4 sources,
protocols and all original evidence remain preserved. No remote job, paid run,
official-test access or publication was performed. Final evidence/notebook/
standalone checks and direct publication sync follow.

Final stage 5 delivery checks passed: seven public frozen-copy hashes/receipts,
all 48 per-rule cell aggregates, 24 checkpoint bindings/native replay tolerances,
12 prior causal-frequency matches and all blocked decisions. All four regenerated
notebooks pass parity (82 embedded files total), 138 first-party local Markdown
links resolve, and the standalone publication passes its 535-file inventory,
syntax, historical evidence and 12 pinned upstream checksums. Running the known-rule
control from the standalone Prod directory reproduces the recorded positive and
negative outcomes. The entire 86-test local suite passed; no failure was recorded
in this stage. Final direct sync/check follows this entry so the append-only
transcript is included without logger-induced drift. No neural/GPU training or
new language-transfer measurement occurred.

## 2026-10-19 — stage 6 controlled transformer trainability

User requested executing the next objective from the stage 5 review. Added
`trainability_v6/`, `configs/trainability_v6.json` and a separately declared
`docs/TRAINABILITY_V6_PROTOCOL.md`, preserving v1-v5 sources and evidence.
The task uses the same v4 six-layer/192-wide/four-head backbone, tied head,
patch encoder, first-frame target mask, Adam and full 1000-update PPT schedule.
It deliberately simplifies the task to binary cell complementation on a 4x4
grid over eight frames (18-token vocabulary, context 47). This self-local
capability check cannot isolate the original random-rule failure or establish
GraphNCA competence, transfer acceleration or cost recovery.

Freeze three paired initialization seeds 131/149/167, one shared fresh data
split (512 train, 64 development, 128 assessment), complement-phase disjointness,
both coherent/prefix-shuffled training arms and final-checkpoint-only evaluation.
Each arm gets 1000 updates / 188,000 input tokens. Both inherited 0.01-nat
advantages must pass in all three coherent-trained models. Assess only after
all six full-budget arms finish, retaining initial/final checkpoints, optimizer
states and all 18 assessment cells. Language training stays disabled regardless
of outcome. No topology, serializer or language-stage experiment is launched.

Eight new regression tests pass, covering phase-disjoint splits, known patch
transitions and target counts, matched controls/determinism, frozen backbone/
budget/margins, altered arrays/plans, no assessment before completed training
and refusal to overwrite evidence. A retained tiny two-arm/two-update execution
fixture also completes final scoring and exact checkpoint replay; its negative
gate is an execution outcome, not a trainability finding. Append-only logger
now records stage 6 source hashes. Freeze the full diagnostic before training;
do not retune its budget or gate after seeing loss curves.

## 2026-10-20 — stage 6 completed assessment and review

The full diagnostic was frozen before model fitting with plan SHA-256
`3970a3b041e26ddecdd977c22f673b744a5f910643acb809e0d9c63a7b010ed4`.
All six full-backbone CPU models completed the declared 1000 updates: 6000
updates / 1,128,000 input tokens total, 168,000 non-masked prediction targets
per model. Logged fitting command took 423.974 seconds; saved per-model stage
timers sum to 417.971867 seconds, with setup/development/checkpoint work included.
This is local CPU timing, not GPU acceleration or cloud billing evidence.

Only after all models finished, assessment scored 128 fresh phase-disjoint
initial conditions. Coherent-trained seeds 131/149/167 have patch NLL
0.0527662036 / 0.0536799047 / 0.0628052821, versus causal frequency 2.3053336763.
The same checkpoints on matched shuffled data score 4.5946005780 / 4.5908170079 /
4.3826129742. All three pass both unchanged 0.01-nat margins with 100% patch
accuracy; mean coherent NLL is 0.0564171301 and mean ordering advantage 4.4662597232.
The train-only transition-table control recovers every complement mapping and
passes; the frequency-only negative control fails. Reopening all three initial
and six final snapshots reproduces all 18 assessment cells/decisions exactly
with the same scorer, checking final optimizer counters. This is not a fresh
training rerun or independently implemented loss function.

Added public report/export and integrity helpers, 21 unchanged source-run copies
with receipts/provenance, per-initial-condition evidence, an 18-cell CSV and
PNG/SVG learning curves under `docs/results/trainability_v6_run1`. Visually
inspected the plot and reviewed the complete cell table. Independently recreated
all six training index-order hashes and checked budgets, phase splits and v4/v5
source preservation. The complete local software suite passes 94 tests.

A descriptive control inspection finds all three shuffle-trained checkpoints
also pass the absolute margins when scored on coherent/shuffled data. Their
ordering gaps are only 0.0610-0.0789 nats and coherent accuracy about 27%, versus
the coherent-trained models' 4.32-4.54-nat gaps and 100% accuracy. Added this
qualification to the export template, review and results/context; it does not
change the frozen primary decision or introduce a new admission threshold.
This capability result does not isolate why the random-rule experiment failed.

Updated root README, context, results ledger, validation and changelog. Selected
the new package/config/tests/helpers/evidence explicitly for publication, with
recursive `proofs/lean/` and `docs/theory/` inclusion. Existing mathematical
sources/verification records will be copied; no new Lean check or LaTeX
compilation is claimed. The next proposed bridge changes only the transition
to an interacting binary rule. No such new run, GraphNCA treatment, remote/paid
job, language stage or official-language-test evaluation was launched. Original
v1-v5 sources, protocols and private/public evidence remain preserved. Final
notebook/publication checks and direct sync follow.

Stage 6 final delivery checks passed. The updated report template executes on
the retained frozen run into a separate private output without replacing the
first public/frozen evidence. Compact-copy/decision checks, including the
shuffle-trained control qualification, pass. Publication inclusion and exact
root/Prod bytes agree for all 19 selected `proofs/lean/` sources/records and
11 `docs/theory/` sources/records; this is a copy/parity check, not a new proof
or document compilation. All four notebooks regenerate and pass parity (82
embedded files), 155 local Markdown links resolve, and the standalone publication
passes its 572-file inventory, syntax, historical evidence and 12 pinned
upstream checksums. No failed scientific check or retuning occurred in this
stage. Final direct sync/check follows this entry, including the latest
append-only transcript and helper changes. No new language/GPU training,
official-language-test access, remote publication or paid job occurred.

## 2026-10-19 — requested Prod refresh and standing mathematics inclusion

User requested updating Prod with current work and including all new Lean proofs
and LaTeX documentation by default in future updates. Recorded this standing
instruction in AGENTS.md and documented it in README.md. Existing publication
selection already recursively includes `proofs/lean/` and all of `docs/`, so new
proof sources, supporting records and theory sources in these trees are selected
without enumerating individual files. Other new locations must be explicitly
selected; build caches and private artifacts retain their exclusions.

Regenerate all four notebook bundles, inspect publication membership and retained
Lean evidence, and check documentation links before the final direct sync/check.
This is publication maintenance, not a new Lean execution, LaTeX compilation,
training run or scientific result. Preserve all earlier evidence and log entries.

All four notebook regenerations passed. All 138 local documentation links resolve;
all 30 mathematics files (19 proof-package files and 11 theory-document files)
are selected. Retained browser evidence matches all 44 theorem sources; this did
not execute Lean. The initial direct sync failed because Prod contained an
unmanaged `docs/presentation/GPU_NCA_Proofer_Presentation.pptx`. Repeated the
failure through the check logger to retain the diagnostic. An initial inspection
also used an unsupported Windows wildcard argument to rg; repeated the search
with directory and glob options. SHA-256 inspection proved the extra presentation
is byte-identical to the preserved original `docs/GPU_NCA_Proofer_Presentation.pptx`
(`07f9c9bcf297317a1851625647f6529502f73c9cacfaf288359ac875e96144c6`).
Preserved that copy at the corresponding canonical root location, allowing the
existing recursive docs selection to include it. No original artifact was
deleted or overwritten. The old standalone manifest still passed 535 files,
but that check alone did not establish current root/Prod parity; final refresh
and checks follow after this correction.

Corrected synchronization and standalone integrity validation passed: 535 selected
files, 82 embedded notebook sources, 12 pinned upstream checksums and historical
evidence integrity. All earlier failure entries are retained. Final direct sync
and parity check follow this note to include the completed append-only transcript.

## 2026-10-20 — stage 7 matched spatial-interaction bridge

User requested completing the one-interaction bridge. Added separate
`interaction_v7/`, config and frozen protocol. Reference rule is binary
complement; treatment adds left/right neighbor parity on the same periodic
4x4 grid. Alphabet, grid, context 47, serializer, full backbone, optimizer,
1000-update endpoint, all batch/schedule settings, model/data seeds and split
sizes match v6. Both rules are involutive/bijective; the new coupling can cross
a patch boundary. Numerical training and patch scores reuse unchanged v4/v5.

Phase classes differ under the new rule. Use one fresh common split with no
joint complement/spatial orbit crossing any split, and rerun the complement
reference on that same split. There are 16,512 joint classes among 65,536
states. This explicitly changes the sampling policy for BOTH new treatments,
not just the spatial arm; historical v6 is separate evidence. Shared initial
states, actual starting weights and minibatch index streams make rule the only
within-stage treatment difference. Twelve models receive 1000 updates each;
all assessment likelihoods wait until every arm has finished.

Preserve both 0.01-nat readiness criteria. Separately predeclare spatial
qualification against a training-fitted previous-own-patch transition table,
also at 0.01 nats, to avoid equating readiness with cross-patch competence.
The evaluation positive control is a known full-previous-frame oracle for both
rules (probability .99); it can represent the new coupling, unlike an own-patch
table. This is an explicitly declared measurement change, not a training change.
Language training stays disabled regardless of the outcome.

Ten new regressions pass, including exhaustive bijection/involution, joint
orbit invariance/count, causal cross-patch perturbation, matched split/masks,
oracle shuffle sensitivity, all settings frozen, tamper/overwrite protection
and no assessment before completed training. A retained four-arm/two-update CPU
fixture also completes scoring and checkpoint replay; its negative gates are
execution evidence only. The logger now retains new package identities.
Freeze the full plan before fitting; do not change its sources, margins or
budgets after inspecting learning curves. No remote/paid/GPU or language run
is authorized by this diagnostic.

The full stage 7 plan was frozen before fitting at SHA-256
`869690ac3805b2b015d4d918d7e3ca742358c7185b983cdcf41ae32476942251`.
The complete local suite passes 104 tests. Twelve full-budget CPU models are
training; no assessment likelihood has been computed. Added public export and
integrity helpers, CI fixture configuration (not a completed CI execution),
and explicit new publication selections. No old model source was edited.

Structural diagnostic limitation identified before assessment: both rules are
period two, so a model could beat a previous-one-frame own-patch table using
two-frame local recall. Added a SEPARATE supplementary protocol/script and froze
its plan during training, before any primary/final-checkpoint assessment was
inspected. Main plan, sources, criteria and endpoints remain unchanged. The
supplementary plan SHA is
`befb1207e0effb2ab212f855c7aba62174b69bd85db506f116732c85373cb597`.
It retains all 128 original first-transition first-patch queries, plus 122
eligible neighbor-bit intervention pairs after identity-only exclusion of six
changed states with train/development joint-orbit overlap. The changed bit is
outside the target patch; its complement target stays fixed and spatial target
flips bit zero, asserted before scoring. No repeated-frame or new-frame patch
is available to these queries. A separate stricter table/uniform check is
predeclared without altering primary readiness, with per-query behavior retained.

Added one-command repeat orchestration (both plans frozen before fitting on a
repeat), source/receipt checks and supplemental checkpoint replay support.
Documented the protocol and reproduction procedure in README. This supplement
is labeled development work declared during training, not an original
pre-training endpoint or untouched confirmation. Its source/protocol/query hashes
are independently bound; neither frozen code package was changed.

## 2026-10-20 — stage 7 completed primary and supplementary results

All twelve full-backbone CPU models completed the frozen 1000-update endpoint:
12,000 updates / 2,256,000 input tokens, 168,000 supervised targets per model.
Logged fitting took 813.520 seconds; per-model stage timers sum to 807.869407
seconds (13.464 minutes, including setup/development/checkpoint work). No GPU
speedup or cloud-billing interpretation is made.

Both readiness tests pass for all six coherent rule/seed checkpoints. Spatial
seeds 131/149/167 have coherent patch NLL 0.295222318 / 0.303057447 / 0.311129497,
mean 0.303136421, versus causal frequency 2.106976207 and own-patch table
2.814026177. Mean full-trajectory accuracy is 91.8434%. All three spatial models
pass the separately predeclared 0.01-nat local-table qualification. The newly
trained complement reference has mean NLL 0.055023387 / 100% accuracy. No
cross-task loss ratio is interpreted as language learning or acceleration.
Spatial shuffle-trained models pass the primary gate in 0/3 seeds; reference
shuffle-trained models pass in 3/3. Both full-frame oracle controls pass, and
frequency-only negatives fail. All 36 primary cases replay exactly from the
saved three initial and twelve final checkpoints, including optimizer counters.

The separately frozen first-transition diagnostic also passes its NLL comparator
in all three coherent spatial models. Mean first-patch NLL is 1.461689415;
original accuracy is only 47.3958%, neighbor-flip accuracy 42.0765%, and both
targets are correct in 19.9454% of the 122 eligible pairs. These limits qualify
the 91.84% headline: spatial-rule learning remains partial. Spatial shuffle-trained
controls also pass the supplementary small NLL margin in 3/3 seeds, with only
13.54% original / 11.48% changed-query accuracy. No causal coherent-training or
reliable rule-execution conclusion is inferred from gate passage alone.

All 18 supplementary cases replay exactly, including original/changed per-query
loss/correctness and argmax-response diagnostics. Timeline checks against the
append-only log confirm the supplement completed during fitting at
2026-10-19 18:53:32 UTC (20 October 00:23:32 India time), before primary
assessment began at 19:03:29 UTC (20 October 00:33:29 India time). The primary
plan and its status remain unchanged. Supplementary chronology and receipts
are retained in its separate verification record.

Added the full review, 36-cell CSV, compact summary, PNG/SVG training curves and
40 unchanged private-run evidence copies with SHA provenance under
`docs/results/interaction_v7_run1`. Reviewed the complete report and visually
inspected both plot panels. Integrity checks recreate all twelve batch-order
hashes/budgets, shared initial states/weights, joint-orbit split exclusion and
old v4/v5/v6 source identities. The full local suite passes 104 tests. The repeat
driver's CLI parses, while its individual scientific phases were executed here;
no redundant second full training suite is claimed.

Updated README, context, results ledger, validation, changelog and explicit
publication membership. The next proposal is a separately frozen first-transition
loss-weighting ablation with rule/model/data/update-token budget fixed and
standard-loss/shuffle controls. It has NOT been trained or tuned here. These
inspected initial conditions are development data, requiring fresh joint-phase
holdouts for later confirmation. Language training remains disabled regardless
of the positive toy-task checks. No remote/paid job, new GPU run, official
language-test access or publication occurred. Existing Lean/LaTeX sources and
records remain selected; no new compilation is claimed. Final template,
notebook, documentation, standalone and direct sync checks follow.

Stage 7 final delivery checks pass: the updated report template executes into
a separate private directory without replacing the original public/frozen
evidence. All four notebooks regenerate and pass parity (82 embedded sources),
177 local Markdown links resolve, and standalone integrity passes 631 selected
files, syntax, historical evidence and 12 pinned upstream checksums. Earlier
v6 evidence remains intact. All 19 selected proof-package and 11 theory-source/
record files match root/Prod bytes; no new Lean execution or LaTeX compilation
is implied. The first-transition loss-weighting proposal is documented but not
implemented or trained in this stage. Final direct sync/check follows this note
to include the completed append-only transcript and final helper changes. No
new language stage, official-language-test access, remote/paid job or GPU
training occurred. The only full scientific training here is the declared
twelve-model CPU bridge; no primary criterion, seed or budget was retuned.

## 2026-10-20 — Stage 8 preparation: first-transition loss weighting

Implemented a separately declared development ablation in loss_weight_v8.
The only training treatment multiplies all six first-transition targets by six,
normalizing the objective so this transition contributes half the loss while
preserving the patch/marker ratio. Spatial rule, exact cached data, saved initial
weights, model, batch order, optimizer, schedule and 1000-update/token budget
remain matched to the six preserved stage 7 spatial standard-loss controls.
Six new weighted coherent/shuffled models will be trained locally on CPU.
Imported controls retain their historical identities and timings.

The frozen primary requires at least 0.01 nats improvement in both native and
neighbor-intervened first-patch NLL for every coherent seed, retaining both
full-trajectory readiness advantages. Shuffled objective gains, accuracies and
whole-trajectory tradeoffs are descriptive. No accuracy threshold or language
authorization is inferred from passing this inspected-development comparison.

Eight objective/matching regressions pass. Their first invocation retained a
harmless tensor-to-scalar warning; the test now detaches that scalar explicitly.
A retained two-update full-backbone CPU execution fixture completes and exactly
replays its full/first-transition scores. This fixture is execution evidence only;
its imported controls have longer training and are not a scientific comparison.
All commands and source identities are retained by the append-only logger.
Full scientific freeze/training follows this note; no result is yet claimed.

The full scientific stage 8 plan froze successfully with SHA-256
`d8d69d609cf9de8b1bf683397eebdc8468ce5b82513f653d142a4e2481a21abf`
before starting new fitting. The complete local suite passes 112 tests cleanly,
including the detached-scalar warning correction. Training is underway with
six immutable standard-loss control identities and six new weighted endpoints;
assessment is blocked until every new model finishes. Added report/integrity
helpers and explicit publication selection for the new package, protocol,
config, tests, helpers and compact results. Mathematical source/evidence
directories remain recursively selected. No scientific outcome is asserted by
this preparation note.

## 2026-10-20 — Stage 8 completed: negative fixed-budget weighting result

All six new weighted models completed 1000 updates each, 6000 new updates /
1,128,000 input tokens total. Command duration was 455.220 seconds; summed new
per-model CPU timers were 448.848148 seconds (7.481 minutes). Six imported
standard controls account separately for 403.713604 historical CPU seconds.
Every paired update/token/target count, actual initial state and recreated batch
index order matches. The full scientific plan was frozen before this fitting;
assessment followed completion of all six models. No factor or budget changed.

The primary FAILS in all three coherent seeds. Standard minus weighted original
first-patch NLL is -0.716168/-0.503166/-0.269025; intervention NLL improvement is
-0.682965/-0.288322/-0.222588 for seeds 131/149/167. Thus every seed worsens BOTH
first-transition likelihoods, although inherited trajectory readiness still passes.
Means: original NLL 1.461689 -> 1.957809, accuracy 47.40% -> 32.03%; intervened
NLL 1.608771 -> 2.006729, accuracy 42.08% -> 30.05%; paired-both correctness
19.95% -> 11.20%; trajectory NLL 0.303136 -> 1.347235, accuracy 91.84% -> 50.86%.
Coherent-minus-shuffled weighting gains are negative for every seed; all controls
remain descriptive and intact. The result does not identify the cause of failure.

Final-only assessment and a separate checkpoint-reload invocation reproduce all
30 full cases and 15 original/intervention cases exactly, with standard scores
equal to stage 7. The evidence checker passes 21 source/public copy hashes,
paired starts/budgets/orders, nested score aggregates/decisions and logged freeze-
training-assessment chronology. Historical stage 7 integrity also passes without
altering its records. Exported all per-query evidence, CSVs, summary, provenance
and three-panel accuracy figure; visual inspection confirms readable labels and
individual-seed dots. A derivative-report wording correction calls the full
accuracy change a regression/change, avoiding any suggestion of a compensating
benefit. Frozen source, protocol and copied evidence remain unchanged.

Updated README, research context, results ledger, validation and changelog to
preserve the negative primary result. The next PROPOSED diagnostic is existing-
checkpoint first-transition fit/generalization across train/development/assessment,
with a new declaration before those additional scores; it has not been performed.
No extra training, threshold relaxation, seed removal, topology/serializer change,
language/GPU/remote/paid run or official-language-test access followed the outcome.
Notebook regeneration, standalone/documentation checks and final direct sync
follow. Existing proof/theory sources and evidence remain selected; copying them
does not constitute a new Lean or LaTeX execution.

Stage 8 final delivery checks pass. The derivative report, summary, CSVs and
provenance reproduce exactly into a separate retained private export directory.
All four notebooks regenerate and pass parity (82 embedded sources). All 196
first-party Markdown links resolve, and starter CRC/all 669 inventory hashes
pass. The standalone publication passes syntax, 669-file inventory, historical
evidence and all 12 pinned upstream checksums. Its complete local software suite
also passes 112 tests without access to private training inputs, separately from
the root suite. Configured remote CI is not being reported as run.

Preserved stage 6 evidence passes, and all 19 selected Lean package/evidence plus
11 theory source/record files have exact root/Prod byte parity. No new Lean or
LaTeX compiler invocation occurred. Frozen stage 8 sources, inputs, plans,
checkpoints and 21 copied evidence files remain intact. No scientific criterion
or endpoint changed after assessment. The only new full training is the declared
six weighted CPU models; the negative outcome remains primary. Final direct
sync/check follows this append to include every completed logged check and this
note. No push, publication, language stage, GPU or remote/paid run occurred.

## 2026-10-20 — presentation refresh through stage 8

User requested updating the detailed v2 presentation. Preserved that 92-slide
file and authored a separate 128-slide stage-8 version, reusing its existing
design and evidence. Added 36 slides for reviewed v3/v4 GPU evidence, prefix
controls, causal frequency, stages 5-8 CPU diagnostics, the spatial involution
derivation, normalized weighting arithmetic, actual source/fourth-notebook
excerpts, results, limits and proposed next work. Revised stale pending-v3
status, speaker notes, source navigation and the overall conclusion. No new
experiment, result, Lean theorem check or LaTeX compilation was performed.

The live presentation session was unavailable (status returned an invalid
transaction), so the explicit local-PPTX edit used the bundled artifact-tool
workflow. A PowerShell brace-style directory list caused a parse error during
initial inspection and was replaced by explicit paths. Two documentation/file
lookups used nonexistent filenames; directory search located the actual API
reference and available evidence. The first visual review found overflowing
code excerpts and a notebook selector that chose the smoke cell; corrected the
excerpt ranges and selected the actual development execution cell. A direct
assignment to imported table-cell text failed because it is read-only; repaired
it with the documented table.cells.set API. All failed logged build attempts
remain retained. As in the earlier deck build, the Windows renderer completes
all PNG/PPTX writes but exits with native code 3221226505; these are recorded
as failures, not passed renderer processes. Separate package/content checks
and final re-import/render review assess the completed artifacts.

Artifact import omitted chart workbooks. Restored all 12 existing chart XML,
relationship and actual workbook parts byte-for-byte from the preserved source,
including the formula-bearing original workbook. Updated only the dated shared
template footer from 17 to 20 October. The unmodified official finalizer passed
128-slide structure, geometry, fonts and all four editable chart/workbook
requirements, with no geometry findings. Its local hardlinks required sandbox
escalation, approved by automatic review; no external publication was involved.
Full final-slide rendering, source/result checks, documentation and publication
sync follow. Earlier raw evidence, model implementations and proof sources are
preserved; CPU measurements remain distinct from imported GPU results.

Final visual review shortened one long source-path row and clarified that the
44-statement Lean core is selected coverage of the theory. Preserved the first
validated revision privately and finalized the repair separately. The delivered
deck has SHA-256 `3e904da4e0f9b4e2b4f5fe2569d4856ef00d9b6b9aa6c81e5ca864558a5907a5`,
4,335,631 bytes, 128 slides, four native charts with unchanged actual workbooks
and 40 native tables across 36 owner slides. All 128 saved final PNGs validate;
all 79 source slides without content changes match their earlier content pixels
exactly, excluding page/footer areas. Reopened final slide 69 differs from the
in-memory draft but matches the preserved reference exactly; inspected it and
all changed/new slides individually. The native PowerPoint application was not
used for this review. Renderer exit failures remain recorded as failures.

Independent content checks pass ZIP integrity, all 13 distinct original embedded
image hashes, 12 exact chart/workbook parts, all notes, removed stale status,
the development notebook cell and displayed v4/stage-5/stage-8 results against
full-precision reviewed records. Wrote the 128-slide source guide and separate
coverage snapshot (377 evidence hashes, 84 scientific source hashes, four notebook
identities). Preserved the older source snapshot and presentation bytes.
All 200 local documentation links and retained 44-statement Lean evidence checks
pass. Regenerated all four notebooks and confirmed coverage hashes and inclusion
of all 30 existing Lean/theory files. No new proof compilation is implied.
Final standalone release and direct root/Prod synchronization checks follow.

The first post-regeneration source-traceability check failed: the historical
notebook embeds README.md, so regenerating it after updating the presentation
link changed its documentation-cell hash. This corrects the preceding note's
premature statement that all coverage hashes were already confirmed. Diagnostic
comparison found no deck, evidence or scientific-source drift; only that notebook
identity needed refresh. Regenerated the coverage snapshot after all notebooks,
then refreshed the starter archive. The final traceability check passes every
deck/source/evidence/notebook hash and all 30 mathematics-file selections.
Documentation and starter checks pass 200 local links and all 671 archived
inventory hashes. No historical experiment implementation or measured outcome
changed. The initial assertion failure and resolution remain in checks.jsonl.

Standalone publication integrity passes 671 selected files, all 82 embedded
notebook sources, historical evidence checks and 12 pinned upstream checksums.
The new deck, guide and coverage record are included in the canonical publication
selection. Final direct sync/check follows this note so the completed append-only
transcript is copied without logger-induced drift. No model training, remote
CI, official-test access, GitHub action or new mathematical compiler run occurred.

## 2026-10-20 — Stage 9 preparation: existing-checkpoint fit diagnostic

Declared an inference-only, post-hoc development protocol before computing new
first-transition train/development scores. Preserve inspected stage 8 assessment
and its negative primary. Score all 15 retained initial/standard/weighted model
conditions on both coherent and matched shuffled data in all three cached splits
(90 cells). Reuse the exact short-prefix first-patch scorer and add explicitly
teacher-forced four-patch/marker measurements for the complete first transition.
No training, new intervention, seed selection or language entry point exists.

Freeze descriptive paired NLL categories at 0.01 nats: training/assessment
regression versus training gain with assessment reversal versus mixed. These are
endpoint descriptions, not causal mechanisms, acceptance gates or confirmation.
Keep all per-query evidence and original assessment replay, then separately
reload/replay all checkpoints. Added source logging for the new fit_v9 package.
An initial multi-file edit was rejected because its LOGBOOK context did not
match; no files from that failed edit were applied. Reapplied it with the actual
append context. Checks and scientific freeze follow this preparation entry.

## 2026-10-20 — Stage 9 completed: mixed fit/generalization finding

Eight new tests pass, then the scientific diagnostic froze with plan SHA-256
`73eca0060bef35833a4579b3dc2c04f1353a093b90446621ba0dfb75f81aca9f`
before computing new split scores. The full local suite passes 120 tests. No
optimizer update occurred. Assessment of all 90 cells takes 49.238 CPU seconds;
separate checkpoint-reload/replay takes 41.077 seconds. These command timings
include source/input checks and are inference timings, not training acceleration.

Coherent weighted-minus-standard first-patch NLL deltas (train/development/
assessment) are 0.147160/0.714747/0.716168 for seed 131,
-0.090349/0.711946/0.503166 for seed 149, and
-0.155300/0.349933/0.269025 for seed 167. Thus two seeds show training gain with
held-out reversal, while one worsens on training data too. Do not replace this
mixed observation with a pooled explanation. Means: train NLL 0.843632 ->
0.810802 and accuracy 75.39% -> 83.46%; development NLL 1.315831 -> 1.908040
and accuracy 54.17% -> 39.06%; assessment NLL 1.461689 -> 1.957809 and accuracy
47.40% -> 32.03%. Assessment-minus-train NLL grows 0.618058 -> 1.147007.

All three coherent seeds improve train and assessment FOUR-patch first-transition
NLL and accuracy. Mean assessment four-patch NLL changes 1.629741 -> 1.549562,
accuracy 42.90% -> 46.48%. Later patches condition on earlier TRUE patches, so
this aggregate improvement coexists with the original first-patch failure.
Separate marker scores and all shuffled models/data versions remain. No unique
cause such as capacity, gradient conflict or memorization is demonstrated.

All 90 cells/per-query vectors replay exactly after reopening 15 saved files.
Every coherent assessment first-patch vector/mean/accuracy reproduces stage 8.
The longer causal prefix differs by at most 4.76837158203125e-7 nats and has
identical first-patch correctness. Seven public/source copy hashes, aggregates,
paired categories/gaps and logged freeze-before-scoring chronology pass.
Public evidence includes all per-query values, four CSVs, summary, review and a
three-panel split figure; visual inspection confirms labels and seed dots.
The 21,120 scoring exposures reuse 704 initial conditions and are not independent
cohorts. No p-value, confirmation or new scientific gate is inferred.

A second documentation edit batch was rejected due to an unmatched context in
RESEARCH_CONTEXT.md. No files from it were applied. Reapplied the documentation
changes against the actual complete anchor; no frozen source or evidence changed.
Updated root documentation and explicit publication membership. The candidate
next treatment is initial-condition diversity ALONE with objectives, backbone,
rule and update/token budget fixed and all controls retained. It has not been
configured or trained. Historical stage 8 failure and random-rule readiness stay
unchanged; language training remains gated. Existing presentation and coverage
files remain authoring-time snapshots through stage 8, not rewritten to imply
stage 9 coverage. Notebook, standalone/documentation and math inclusion checks
plus final direct root/Prod sync follow. No new Lean/LaTeX compiler run, official
language-test access, GPU/remote/paid training, push or publication occurred.

Stage 9 final documentation/starter check initially fails archive membership,
while 221 local links, standalone integrity, mathematical parity and 120
standalone tests pass. The retained diagnostic identifies precisely one missing
ZIP member: docs/paper/main.tex, created by concurrent workspace work between
the builder's file traversal and its later independent manifest traversal.
Preserved the failed generated starter privately as
runs/fit_v9_run1/FAILED_STARTER_INVENTORY.zip; no returned dataset ZIP changed.
Fix the maintenance builder to capture one inventory, read each selected file
once, verify its bytes against that snapshot and write those same bytes and
manifest. New files cannot enter only the manifest, and changed selected content
causes an explicit rebuild error instead of a mixed archive. This changes no
historical numerical experiment or frozen stage 9 source. Rebuild and retained
failure-resolution checks follow. Existing new paper sources remain untouched
and included through recursive docs selection; no paper compilation is claimed.

Added two isolated archive regressions: a file appearing after inventory cannot
enter only its manifest, and modification of a selected file forces an explicit
rebuild. The first focused invocation fails only its fixture byte assertion:
Windows write_text translated the fixture newline to CRLF. The package correctly
preserved those bytes. Fixed the fixture to declare LF explicitly; retained this
failure and rerun rather than changing archive behavior to match the assertion.

After the explicit fixture-newline correction, all eight focused integrity
tests pass, and the complete root suite passes 122 tests. Updated current stage 9
documentation to distinguish eight scoring tests from the two maintenance-race
regressions. Explicitly select the new concurrent docs/paper location in addition
to existing recursive docs selection, preserving its source bytes without editing
or compiling it. The stage 8 presentation/coverage snapshots stay unchanged.
Rebuild all notebooks and the consistent starter, then repeat the affected
standalone/documentation checks and final direct sync/check.

Stage 9 final delivery checks pass after the retained archive/fixture failures
and their corrections. Root and standalone suites each pass 122 tests. All four
notebooks regenerate and pass parity (82 embedded source files). Documentation
resolves 221 local links, and the rebuilt starter passes CRC, membership and all
694 stored content hashes. Standalone integrity passes 694 selected files,
syntax, historical evidence and 12 pinned upstream checksums. The final report,
summary, four CSVs and provenance reproduce exactly into a separate private export.

Historical stage 8 checks retain all 21 copies and paired budgets/orders; no
scientific outcome changes. Stage 6/mathematics checks pass all 19 Lean and 11
theory source/evidence copies with exact root/Prod parity. The one current paper
source is explicitly selected and also matches root/Prod. No new Lean, LaTeX,
paper compilation, GPU run or remote CI execution is implied. Existing deck and
coverage snapshots remain unchanged. All 90 stage 9 cells and 15 historical
assessment vectors remain exact, with zero optimizer updates. Final direct
sync/check follows this append to include the finished append-only transcript.
No further treatment, language stage, official-test access, push or publication
was launched.
