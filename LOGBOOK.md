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
