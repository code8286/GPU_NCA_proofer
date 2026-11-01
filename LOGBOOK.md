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

## 2026-10-21 - Research paper completed (20 October stage 9 snapshot)

Prepared the author-requested paper with Soumyajit Deb, Vellore Institute of
Technology (VIT), and the expressly supplied correspondence email. Preserved the
reference PDF untouched and used its A4 academic structure only. Added the
editable docs/paper project, canonical manuscript, readable transcription,
eight source-grounded figures, numerical checks, validation notes and source
map (504 retained evidence/source hashes). Selected docs/paper explicitly for
publication; linked it from README and summarized it in CHANGELOG. Generated
output/ remains ignored and excluded from the distributable source bundle.

Final local PDF: 23 pages, 28 numbered vector equations, 16 tables, eight figures;
SHA256 2fccf60a428add4c9493c20e4d277f3ab57a196c79819c8a3f0949a0c70380cf.
All pages were rendered and visually checked; content verification passed 388
prose windows, numbering, byline, references and page bounds. The final LaTeX
project separately compiled successfully in the desktop compiler on 21 October.
The delivered PDF was separately typeset from the same canonical content;
LaTeX compilation is not represented as its export. Exact arithmetic checks
passed, including the stage 9 position-wise NLL decomposition. This is a new
written algebraic inference, not a new experiment or formal Lean theorem.

Retained authoring failures and resolutions: optional fitz was unavailable
(use pypdf/Poppler); console/Lean source decoding required explicit UTF-8;
MathText required braced commands; lattice row coordinates required corrected
integer-division precedence without donor-data changes; PDF stream-order text
checking required spatial reading order with page furniture removed; crowded
plot annotations were repositioned. Earlier failed check entries remain intact.

Regenerated all four versioned notebooks. Documentation checks passed 240 local
links and starter CRC/all 708 inventoried hashes. Static checks matched all 44
Lean statements across six modules to retained verification records; no new
Lean execution or recompilation of the older theory project is claimed. No model
training, new checkpoint scoring, official-test access, push or remote job was
performed. Final publication integrity and root/Prod parity checks follow.

Paper final packaging checks passed: 709 publication files, 82 embedded sources,
12 pinned upstream files and historical evidence integrity. Every source-map
hash and final PDF identity matched. Exact root/Prod parity passed for all 19
Lean files, 11 theory files and 16 paper sources/figures. These checks preserve
existing proof evidence; they do not execute Lean. Final direct sync/check below
includes these completed log entries and this append.

## 2026-10-21 — Stage 10 preparation: nested initial-condition diversity

Declared the next single-variable study before fitting: expand 512 original
training initial conditions to 4096 while preserving exact original rows and
shuffles, the fixed known spatial rule, both objectives and their factor-one/
six kernels, model, initial states, Adam/schedule and 1000-update exposure.
Exclude all old train/development/assessment and neighbor-flipped assessment
joint phases from new additions; no old eligible intervention becomes leaked.
Keep twelve original 512-condition endpoints as imported verified controls and
fit twelve new larger-cache models. No language or remote/paid run is included.

Freeze first-patch AND intervention NLL improvement >=0.01 for every coherent
seed, plus unchanged full-trajectory readiness, separately for standard and
weighted objectives. The study-wide primary requires both strata. Report failures,
all shuffled controls and costs without retuning. This is inspected development
data, not confirmation or evidence of language/GNN acceleration.

Ten focused tests pass, including a tiny isolated four-arm/two-update execution
fixture with mocked provenance (software evidence only). Sampling uses the same
RNG seed and draw count but a different bound; do not claim identical examples
across cache sizes. Freeze expected index digests and actual distinct coverage.
Fixed anchor/extension fit probes distinguish cache membership from actual draw
exposure, with null metrics for empty subsets. Retain raw score vectors in
per-seed evidence files under the existing public size limit. Logger now records
the new diversity_v10 implementation identity. Full freeze and CPU fitting follow;
no new scientific outcome is asserted by this preparation entry.

The full stage 10 plan froze with SHA-256
833219dd13b85713c34af5b649cfd2693939b1b766e3a3d900933685ac231101
before new fitting. The full local suite passes 132 tests. Independent reconstruction
reproduces all 4096 coherent/shuffled trajectories, nested original rows, held-out
phase exclusions and all 122 eligible interventions. Frozen actual-diversity
predictions are 2532/2533/2558 distinct rows for seeds 131/149/167, versus
512/512/511 at the original cache size. All four new arms within each seed must
match their larger-cache draw trace. Source/input identities remain frozen while
training proceeds. Added export/integrity helpers and explicit publication
membership without modifying any frozen experiment source. No assessment result
has yet been inspected, and no criterion or training setting was changed.

## 2026-10-21 — Stage 10 completed: objective-specific diversity benefit

All twelve new larger-cache models finish 1000 updates each: 12000 new updates /
2256000 input tokens, versus twelve preserved historical control budgets. The
training command takes 795.593 seconds; new per-model timers sum to 788.816615
seconds (13.147 minutes), while imported controls account separately for
852.561752 historical seconds (14.209 minutes). These are CPU measurements across
different invocations/concurrent checks, not controlled throughput or GPU speed.

The frozen study-wide primary FAILS. Weighted loss passes all three coherent
seeds; standard passes only 167. Standard first/intervention NLL improvements
are -0.236307/-0.180254 for 131, -0.675943/-0.458504 for 149, and
0.162120/0.250459 for 167. Weighted improvements are 0.669642/0.754802,
0.152022/0.175839 and 0.713782/0.758982 respectively. All six larger coherent
models retain trajectory readiness; no seed, objective, threshold or endpoint
was removed or retuned.

Weighted means (512 -> 4096): original NLL 1.957809 -> 1.445994, accuracy
32.03% -> 45.57%; intervention NLL 2.006729 -> 1.443522, accuracy 30.05% ->
43.99%; paired correctness 11.20% -> 24.32%; full accuracy 50.86% -> 57.56%.
Standard means: original NLL 1.461689 -> 1.711733, accuracy 47.40% -> 27.86%;
intervention NLL 1.608771 -> 1.738204, accuracy 42.08% -> 25.96%; paired
correctness 19.95% -> 5.19%; full accuracy 91.84% -> 90.33%. The larger weighted
arm still has lower first/full accuracy than the original smaller standard arm;
this cross-objective comparison is descriptive. Do not promote the weighted
stratum into a general diversity or language-acceleration claim.

Larger shuffled controls also pass coarse readiness (3/3 standard, 2/3 weighted)
with weak first-patch accuracy. Preserve that limitation and all calibrated
train-only frequency priors, actual distinct exposure and drawn/undrawn probes.
All 54 full, 108 fit and 27 intervention cases replay exactly from 27 initial/
final files in a separate invocation. Assessment takes 77.899 seconds; replay
76.296 seconds. All imported score vectors match stages 8/9. Forty-two copied
evidence hashes, nine raw per-seed files, nested aggregates/exposure masks,
paired decisions and freeze-training-assessment chronology pass. No raw evidence
file exceeds 1.45 MB. Public compact evidence supports accounting audits; neural
replay requires the retained private arrays/checkpoints.

The final derivative template reproduces the review/summary/CSVs/provenance into
a separate retained private directory. Visual inspection prompted extra NLL plot
headroom so legends cannot cover seed dots; the first render is preserved
privately, with data and all 42 copied evidence records unchanged. Root status,
results, validation, context, changelog and explicit publication selection are
updated. The full local suite passes 132 tests. No new language/GPU/paid run,
official-test access or extra model fitting followed these outcomes. Any next
confirmation needs a separate frozen objective-specific hypothesis on fresh
joint-phase-disjoint data with both objective strata/controls retained. Existing
manuscript/presentation authoring snapshots and mathematical evidence remain
unchanged. Final notebook, distribution, math inclusion and direct sync checks
follow; copying prior sources is not a new compiler execution.

Final historical notebook regeneration initially fails at the publication guard:
a concurrent generated PDF exists under docs/paper/output/pdf. The artifact is
preserved; no output directory or paper source is moved/deleted. Add an explicit
publication excluded_directories entry for docs/paper/output and validate these
exclusion paths with the existing safe-path guard. Recursive sources/curated
figures under docs/paper, proofs/lean and docs/theory remain selected. Explicitly
selecting a forbidden PDF still fails; the global artifact ban is not relaxed.
Previously published paper build-output membership is empty, so no existing Prod
source/evidence is removed. The focused integrity suite passes ten checks,
including two new regressions for source retention/output exclusion and unsafe
exclusion rejection. The failed regeneration and correction remain in checks.jsonl.
No frozen scientific source, data, model or decision changed.

Stage 10 final delivery checks pass. Root and standalone suites each pass 134
tests, including ten new diversity/exposure checks and two later publication-
policy checks. Repeated tiny mocked-provenance software fixtures in these suites
are distinct from scientific fitting; no additional full-budget scientific model
was trained after assessment. All four notebooks regenerate and pass parity
(82 embedded sources). All 259 first-party local Markdown links resolve; starter
CRC/membership/all 770 recorded content hashes pass after the retained PDF-output
policy failure and correction. Standalone integrity passes 770 selected files,
syntax, historical evidence and all 12 pinned upstream reference checksums.

The final review, summary, four CSVs and provenance reproduce exactly from the
derivative template. All 42 scientific evidence copies and nine raw vector files
remain unchanged. Existing 19 Lean and 11 theory files plus all 16 paper sources,
figures and records have exact root/Prod parity. Generated paper output is excluded
without deleting or compiling anything. No new Lean/LaTeX/paper compilation or
remote CI execution is implied. The final direct sync/check follows this append
to include every completed check transcript and this note. The objective-specific
weighted pass, failed standard stratum and failed study-wide primary stay intact.
No criterion, seed, objective, cache size or endpoint was changed after results;
no further study, language stage, official-test access, paid/remote GPU job,
push or publication was launched.


## 2026-10-21 — Stage 11 confirmation declared, before fitting

User requested the next study. Added a frozen fresh-phase confirmation of stage 10’s weighted diversity finding; retain standard loss and both prefix-shuffled controls. All 24 models will be fitted anew from three fresh paired starts at the unchanged 1000-update budget. Reserve assessment intervention partners and exclude all controlled-binary historical phase classes from stages 6–10. Weighted all-seed confirmation and broader both-objective status are distinct; preserve stage 10’s failed primary. No results or training claimed at declaration. Sources/config/protocol and private receipt precede fitting; language remains gated.

## 2026-10-21 — Stage 11 frozen and fitting; evidence helpers prepared

The frozen plan is f66ad56b5ef509c188f33d92cefc4032ee6979e658aff7259bf8685a49720a4a.
The fresh-data reconstruction passes before training: 4926 historical joint
classes excluded, exact nested coherent/shuffled arrays, 4096/64/128 fresh
splits and all 128 reserved intervention pairs. Three new seeds 179/193/211
share one new split; all 24 models are new fits. Distinct anticipated exposure
is 512/512/512 from the small cache and 2561/2558/2564 from the larger cache,
each from 4000 draws. Actual digests will be checked after fitting.

Seven focused regressions pass before freeze, including a tiny eight-arm
execution fixture with mocked provenance, separate from scientific fitting.
The full local suite passes 141 tests. Fixed-budget CPU fitting is in progress;
no assessment outcome is claimed. Added export/review helpers and explicit
publication selection for the new package, config, tests and compact evidence.
The helpers parse; completed evidence will be checked after full checkpoint
replay. Historical numerical sources and outcomes remain unchanged. The weighted
confirmation primary was declared prospectively; it does not replace stage 10's
failed both-objective primary. No language, remote/paid run, push or publication.

### Stage 11 frozen inputs and pre-assessment checks

Seven focused regressions and the full 141-test local suite pass. An isolated
8-arm/two-update tiny fixture checks execution with mocked provenance; it is not
scientific training evidence. The scientific plan froze at
`f66ad56b5ef509c188f33d92cefc4032ee6979e658aff7259bf8685a49720a4a`.
A separate full input reconstruction confirms 4926 excluded historical joint
classes; fresh 4096/64/128 train/development/assessment classes; 128 reserved
neighbor intervention pairs; exact nested shuffles; unchanged rule/model/budget.
Frozen actual exposure predictions are 512 small-cache rows for every seed and
2561/2558/2564 larger-cache rows for 179/193/211, from 4000 draws per model.
All 24 CPU models are in progress, not yet assessed. Added checkpoint-replay,
public export, independent score/decision accounting and chronology checks;
new distributables explicitly selected, with private arrays/weights excluded.
No source in the frozen scientific package or historical stage was changed.

During stage 11 export preparation, an audit finds a historical stage 10 derived
CSV defect: flattening the readiness dictionary overwrites the all-criteria
passed field. The original nested JSON, review and primary decision are correct;
standard seeds 131/149 still fail and the broad primary stays failed. Preserve
the CSV/template and add docs/STAGE10_CSV_CAVEAT.md rather than silently rewrite
historical evidence. Stage 11 uses distinct passed/readiness_passed columns and
an independent CSV-versus-JSON check. A focused regression for readiness passing
while the complete comparison fails passes. The first patch attempt found an
incorrect text context and made no change; the narrower patch succeeds. No
frozen numerical source or trained parameter changes. Static Lean source/evidence
record checks also pass; this inspects saved records and does not rerun Lean.

Before assessment, the stage 11 report template explicitly explains the fixed
rule's two-update recurrence and why short-prefix/intervention probes remain
separate from full-trajectory readiness. This is a mathematical property already
exhaustively checked in stage 7, not a new measured mechanism or decision rule.
No numerical protocol/source change or new model fitting is introduced.

## 2026-10-21 — Stage 11 status-write interruption and frozen recovery

The original training command fails with Windows PermissionError replacing
status.json.tmp after 21 complete models and 800 updates of the next fit. Its
1519.694013-second failed command/transcript remains logged. No assessment or
partial final checkpoint exists. Preserve all completed endpoints and the
unfinished metrics/status; no seed, loss, budget, data or criterion is changed.
The private continuation correctly stops without assessment on that failure.

Declare docs/CONFIRMATION_V11_RECOVERY.md and a separately frozen operational
helper. Its only train-function edits accumulate verified completed rows and
skip those 21 keys. A new AST-equivalence regression proves the entire remaining
numerical body is identical after reversing those two control-flow edits.
Retry PermissionError only around status writes, never optimizer steps. Verify
all existing checkpoint/optimizer/budget/order/start identities, preserve the
unfinished folder after resolved-path checks, and restart that fit from its
original initialization plus the two remaining fits. Freeze the helper, exact
transformed source and interruption evidence before restarting. Compare its
first 800 metrics exactly against the interrupted prefix afterward.

Final endpoints still total 24000 updates; actual cost must include the discarded
800 updates /150400 input tokens /134400 targets (24800 /4662400 /4166400 total).
The partial recorded timer is a lower bound, with failed command time retained.
Source/config/original plan remain frozen and no outcome has been inspected.

## 2026-10-21 — Reconciled concurrent recovery preparation; no duplicate fitting

Attempt-I was already recovering this same run when Attempt-II prepared an
alternative helper under the same root filename. The alternative freeze fails
at the retained-model-count guard (more than 21 endpoints already exist), before
any mutation or new fit. Preserve its failed logged check and unused driver/
protocol under runs/confirmation_v11_unused_recovery_preparation. The first
alternative declaration above is a preparation record, not the executed recovery.

Read the active task's recorded source and restore the actual executed driver
exactly to its frozen SHA-256
099c0a5895a64ea636055d768845c83455fb4d5802ce7a3f221870db8efb6e61.
Its execution_recovery.json/receipt, completed-folder guard and JSON-only retries
remain authoritative. No original scientific package/source, plan, checkpoint or
criterion changes. Adapt the AST audit to the actual single guard and explain
the unused alternative separately. Both task records and discarded 800-update
cost stay retained; only one set of remaining scientific fits runs. Avoid
launching assessment or further fitting while the active task completes it.

## 2026-10-21 â€” Stage 11 completed, assessed and replayed

The weighted primary confirmation passes (3/3 paired seeds); standard passes 1/3. The broader both-objective criterion fails.
All 24 new scientific models complete the frozen 1000-update endpoint, totaling
24000 updates /4512000 input tokens /4032000 targets. Per-model timers sum to
1652.946594 seconds (27.549 CPU minutes).
This excludes 800 discarded updates /150400 inputs /134400 targets after the
Windows status-write interruption. Actual fitting totals 24800 updates /
4662400 inputs /4166400 targets. Preserve the partial 68.244014-second timer as a
lower bound, the failed 1519.694013-second invocation and successful recovery.
Generation, assessment, replay and software checks add costs; no controlled
throughput speedup is inferred. The original one-shot continuation stops on the failed training command.
Attempt-I completes one journaled recovery and stops before assessment; Attempt-II
restores the exact executed helper checksum, then performs separate logged
assessment, replay, export and accounting. Its unused alternative recovery
freeze stops before any fitting. These are operational fixes, not treatments.

All 54 full, 108 fit and 27 intervention cases replay exactly from 27 initial/
final files. All 128 fresh pairs remain eligible and all 71 evidence copies,
per-query aggregates, masks, decisions, independent decision arithmetic,
CSV-versus-JSON checks and freeze-training-assessment chronology pass. No seed,
threshold, objective, factor, cache size or endpoint changed. Nine stage 11 and three recovery checks
and the full root suite pass 146 tests. Earlier stage 10 failed primary and
all historical raw/derived evidence remain intact with the CSV caveat documented.

Updated canonical status/context/results/validation/changelog from actual scores;
private arrays, weights/optimizer states remain preserved. Declare a separate bridge toward interacting categorical rules, changing one task property at a time and retaining both objectives and controls.
No further scientific fitting, language/official-test access, remote/paid job,
push or publication follows these results. Final distribution/standalone and
math inclusion/parity checks follow; these do not imply new Lean/LaTeX runs.

Stage 11 derivative export reproduces the review, summary, provenance and four
CSV tables exactly in a separate retained private directory. The scientific
figure is visually inspected: all seed dots, legend entries and axes remain
visible. All four notebooks regenerate (historical full starter last) and pass
source parity; all 283 local Markdown links and the starter's CRC/membership/
866 recorded content hashes pass. Root tests pass 146, including the CSV collision
and actual recovery AST/serialization audits. No new fitting follows assessment.

The exact executed recovery helper is restored under its original path and
preserved under the separately named 6950 copy; both match the frozen driver
checksum. The other task stopped after 24 endpoints and its independent recovery
verification. This task completes the original frozen assessment and separate
checkpoint replay: all 189 cases agree exactly. Preserved original failure,
unused alternative preparation failure, recovery and extra discarded compute
remain explicit. The weighted prospective primary passes 3/3; standard 1/3,
broader both-objective criterion fails. No historical primary is relabeled.
Standalone distribution and retained mathematical-source inclusion checks follow.

## 2026-10-21 — Stage 11 final verification and distribution

Root and standalone suites both pass 146 tests. All 71 evidence copies,
189 checkpoint-replayed cases, independent paired decisions, distinct CSV gates,
recovery identities/800-step prefix and discarded-cost arithmetic pass. Weighted
confirmation passes all three seeds; standard only seed 211, so the broader
both-objective criterion fails. Weighted native/intervention accuracy changes
48.44%/40.89% to 56.51%/47.66% on the new split. Do not relabel stage 10's failed
primary, pool its development evidence or infer a language/GNN speedup.

Final endpoint stage timers sum to 1652.946594 seconds (27.549 minutes), excluding
the discarded partial stage's recorded 68.244014-second lower bound and other
pipeline costs. Actual fitting uses 24800 updates /4662400 inputs /4166400 targets;
only 24000 /4512000 /4032000 contribute to the 24 fixed final endpoints. The
original failed command, successful recovery and failed unused preparation remain
in the append-only log with original model/data/source identities preserved.
No new scientific fitting occurs after the assessment.

All four notebooks regenerate after final status/validation/changelog updates.
Generation parity and 82 embedded-source identities pass. All 283 first-party
local Markdown links and the starter's CRC/membership/866 content hashes pass.
Standalone publication integrity checks 866 selected files, historical evidence
and all 12 pinned upstream checksums. Full recursive source/evidence inclusion
and exact root/Prod parity pass for 19 Lean, 11 theory and 16 paper files;
generated paper output stays excluded and locally preserved. These are packaging
and retained-evidence checks, not a new Lean/LaTeX compilation or remote CI run.

The final direct unlogged sync/check follows this append, including all completed
check transcripts and this record. No language/official-test access, remote/paid
training, topology/serializer/budget retuning, push or publication occurred.
Next is a separately frozen categorical-rule bridge with controls/readiness,
not an automatic language stage on the original unready random-rule checkpoints.


## 2026-10-21 — Resumed handoff audit after coordinated completion

On resuming, the other session had already completed stage 11 assessment,
checkpoint replay, export and distribution. Do not rerun fitting or silently
start another scientific treatment. A new logged accounting audit passes all
71 public/private evidence identities, 189 retained scoring cases, paired
objective decisions, recovery identities/prefix and discarded-cost arithmetic,
and frozen chronology. Weighted remains 3/3, standard 1/3 and broad criterion
failed. This is a new CPU accounting check of existing evidence, not new model
training, a new neural checkpoint replay or a GPU/language measurement.
Regenerate distribution snapshots and sync the append-only audit record;
no scientific configuration, checkpoint, source or outcome is changed.

Resumed delivery checks pass: all four notebooks regenerated; direct notebook
source parity, 283 local links and starter CRC/866 content identities pass.
Recursive mathematical-source/retained-evidence inclusion and exact root/Prod
parity pass for 19 Lean, 11 theory and 16 paper files; no compiler is executed.
The following final direct sync/check includes this note and every logged check.
No fitting, assessment, neural replay or outcome retuning occurred on resume.


## 2026-10-30 — Stage 12 OOD implementation declared before predictions

Read the requested private reply and its surrounding thread using the connected
requested account. The actionable advice is explicit out-of-domain testing,
with CLRS-Text as inspiration. Preserve only the relevant reply privately under
runs/advice_20261030; do not copy correspondence, contacts, signatures or CV into
public files, and send no reply. The public methodological basis is CLRS-Text
arXiv:2406.04229v1 and its source pinned to release v2.0.1 commit
d1c2ad7af8437c7536cf329d9cef8fdf93184d9d, including Apache-2.0 notices.
The initial GitHub lookup failed DNS in the sandbox; a read-only network retry
succeeds, and the seven-file immutable snapshot is retained. OpenReview browser
verification blocked direct reading; the official arXiv paper and source were
read instead. These are reference-access failures, not experimental outcomes.

Added a separately declared OOD evaluation layer, with local BFS/Dijkstra
distance-output adaptations explicitly distinguished from official CLRS-Text.
Size, topology-family and joint shifts have separate cells; node relabeling,
edge ordering and guaranteed answer-changing cuts retain paired correctness.
Independent graph oracles, graph-isomorphism-invariant conservative exclusions
including interventions, context/format/coverage checks and fixed source identity
are required. Official prompt/reference JSON exports have a separate strict
final-string adapter. No fitting command is provided.

Declare a tiny historical zero-shot calibration BEFORE neural predictions:
v4 final language cohort 59, five paired arms, 32 graph/task/control prompts
each, unchanged tokenizer/context and no new training. It is a task-format
calibration, not a graph-specific acceleration test or clearance of failed
original random-rule readiness. A future matched ID task-adaptation stage and
fresh OOD confirmation need separate frozen protocols and stronger readiness.
No task or seed will be selected after scores. Language/official-test access,
remote/paid jobs, publication and email sending remain outside this stage.

Stage 12 preparation: all 14 new graph/oracle/scoring/context regressions pass. A documentation edit encountered Windows default-text decoding on an existing UTF-8 file; continue with explicit UTF-8. The frozen scientific package has not yet been executed or changed after a receipt. No model outcome has been inspected.


### Stage 12 completed measurements and verification

The scientific plan froze at
55b705756dcec4c546928a6c3a52e66314a2e455ad2026f2819a5b3642c824c4.
The first tokenizer attempt failed DNS; priming a default cache in the network-
enabled process did not make that cache visible to the sandbox. A second
pre-freeze attempt also failed. Before any receipt or neural score, switch to
one explicit private workspace cache; prime it and freeze offline successfully.
No input/model/criterion changed after the successful freeze. Failed attempts
occur before creating the run directory and remain in the append-only log.

All 14 new tests and the full local 160-test suite pass. Independent Floyd-Warshall
agrees on 5760 labels; native/invariant/cut control results and topology split
exclusions pass. Five retained final v4 language models, cohort 59, each return
0/32 valid and correct answers, with complete context coverage. All 160 outputs,
token IDs and stop reasons replay exactly in a separate checkpoint invocation.
This is an observed zero-shot task-format floor; do not interpret it as an OOD
reasoning gap, graph no-effect/equivalence, new training or learning acceleration.
Per-model prediction timers total 5.160180 seconds including checks/loads, not
isolated kernels; preparation/control/replay costs are separate in logs.

Export 13 byte-identical evidence copies, summary, per-cell CSV and review. The
independent evidence/decision/CSV/chronology/privacy checker passes. Added an
independent dataset-generator CLI requiring no historic weights or email access,
and runnable usage for local/official-export scoring. Informal mathematical
control arguments are new docs/theory source, not a new Lean/LaTeX execution.
Explicit publication membership retains all proof/theory/paper sources and pins
the seven-file Apache reference. Private correspondence, raw datasets, weights
and caches remain excluded. Original outcomes/readiness failures are unchanged.

### Stage 12 final distribution checks

The root and standalone suites both pass 160 tests. The checkpoint-independent
dataset CLI reproduces all frozen train/development/assessment JSONL bytes.
Seven CLRS-Text reference files/license hashes verify; the older 12 reference
checksums and historical evidence remain unchanged. All 13 exported identities,
independently verified zero-shot formatting floor, 160 free-generation replays,
paired controls and chronology pass. All four notebooks regenerate, and source
parity plus 304 local links/starter content checks pass. Publication integrity
checks 907 selected files and 82 embedded sources. Mathematical-source/evidence
recursive membership and parity pass for 19 Lean, 12 theory and 16 paper files;
no new Lean/LaTeX compilation or remote CI.

Update the final distribution snapshot after these validation notes, then run
final direct sync/check after the logged checks so the append-only transcript
remains in parity. No scientific source, configuration, checkpoint, gate or
outcome changed after prediction; no new fitting or email reply occurred.

## Stage 13: prospective matched ID adaptation, 30 October 2026

The user authorizes the next ID task-interface/trainability study. Declare a
bounded historical continuation only: all five v4 cohort-59 final language models,
same full inherited model/head policy and 600 answer-only supervised updates,
common batches/optimizer/schedule. Freeze source/data/order/protocol before fit.
Retain 0/100/300/600 development generations, train/development NLL and all model/
optimizer checkpoints. Require every arm and ID task/size cell to meet 80% exact
match and pooled 95% valid format at fixed final endpoints; otherwise skip OOD.
This does not admit the failed random-rule configuration to a new language or
confirmatory graph study. No original evidence, topology, tokenizer, model or
historical language budget is changed. New private weights/data stay excluded.
Meaningful code/config/publication/logger changes are recorded here; eight new
pipeline tests precede scientific fitting. Publication/review helpers follow.

### Stage 13 freeze and execution

Eight new regressions pass, including full loss/all-parameter gradient equivalence,
causal padding, prompt/EOS masks, full-model optimizer updates, common order,
schedule boundaries and censored/four-cell decisions. Freeze succeeds at
9e18f7f90def47c453f29847244d7715ebab48cdbf0d267a23bbde66957d5b5e.
Each arm's fixed exposure is 562855 unpadded inputs, 96016 supervised answer/EOS
targets and 691560 padded positions. Scientific fitting begins only after this
receipt. No scientific source/config/protocol edits follow the freeze.

Add report/export, independent JSON correctness/chronology/privacy audit and a
runnable usage guide while fitting is active. The training NLL probe is explicitly
the declared first 64 examples (only eight-node graphs); the full dev NLL has
both sizes. Document this composition mismatch rather than claiming a matched
pooled NLL generalization gap. All task/size generation cells remain separate.

### Stage 13 integrity wording clarification during fit

Preserve the original frozen protocol, code and receipt. Its literal assertion
that prior assessment is not read is too broad: verify_ood hashes every prior
input, including assessment bytes, for identity. Only ID train/development are
parsed into stage 13 examples; no OOD prompts/labels reach fitting/selection/new
predictions. Add a separate explicit clarification and correct the derived report
wording. Prior assessment was already inspected development evidence; no untouched
confirmation claim was made or enabled. No scientific edit or retuning occurs.

### Stage 13 scientific fitting completed

All five fixed-budget fits complete in one successful invocation, 1681.873 seconds
command wall time. Total 3000 new CPU adaptation updates; each arm exactly 562855
unpadded input tokens / 96016 answer-EOS targets / 691560 padded positions. Actual
source/input/order/schedule checks pass at fit start and end. All final models
score 0/128 ID exact answers. Valid-vector rates are scratch 99.21875%, grid 100%,
graph 100%, prefix-grid 98.4375%, prefix-graph 98.4375%. Every task/size cell has
zero exact match, every prompt is covered, all 80% crossings remain censored.
Calibration fails and OOD is explicitly skipped with zero new OOD predictions.
Do not treat low NLL or high format validity as learned shortest-path execution.

CPU update timers total 1172.576786s and development generation/NLL timers total
495.862688s; IO/hashing/construction and full command overhead remain separate.
No new PPT/language/GPU measurement occurs; historical costs remain imported.
Start separate same-code replay of all 20 checkpoints /2560 generations and the
full regression suite. Add a three-panel scientific curve artifact and strengthen
the evidence audit to compare every CSV/summary field and independently recompute
all per-update token/target/padding counts from actual sampled IDs. These derived
report/helper changes do not alter frozen scientific code, data or endpoints.

### Public verifiability during stage 13 replay

The complete local suite passes 168 tests. Add a separately selected weight-free
public score-reproduction helper and usage: regenerate declared ID dataset bytes,
check frozen SHA-256 identities and rescore all 2560 exported generations/cells.
It needs no historic/private checkpoints, tokenizer downloads or correspondence.
Neural/NLL replay remains separate and still requires retained private weights.
This derived helper does not change the frozen calibration or any source included
in its plan. Document final scientific results in canonical status/results/context.

### Stage 13 completed evidence and interpretation

Separate replay passes all 20 checkpoints and 2560 generated answers/IDs/stops,
NLL and score vectors, optimizer update counts and initial/final full-model
identities (541.315s command wall time). Export 32 byte-identical evidence copies,
summary, 80-cell CSV, three-panel curve PNG and review. Visual inspection confirms
readable labels, frozen target lines and overlapping zero-exact curves. Evidence,
chronology, independent JSON parse, all CSV/summary fields and independently
recomputed actual per-update exposure checks pass. The public-only reproduction
helper regenerates exact ID dataset hashes and all 2560 answer scores without
private weights/data/cache/network. No failed scientific fit or recovery occurs.

Update canonical README/results/context/validation/changelog. The completed
study learns syntax but misses the exact task criterion for every arm/cell;
no graph effect/no-effect or acceleration follows. Next proposed work is a
separately frozen task/size-matched train/development generated-error audit of
these existing checkpoints before another treatment or fitting budget. Original
random-rule readiness, categorical bridge, binary confirmation and prior negative
results are preserved. Full distribution validation and final direct sync follow.

### Stage 13 final distribution checks

Both root and standalone suites pass 168 tests. The standalone publication also
recreates the exact ID dataset bytes and all 2560 answer-score/cell records using
public files only, without original private run folders or weights. All 32 stage
13 evidence copies, 20 checkpoint replays, independent JSON accuracy counts,
80 derived cells and actual per-update exposures pass. Prior stage 12's 13 copies,
160 predictions, controls and all 19 combined pinned reference/license hashes
remain valid. All four notebooks regenerate; notebook parity, 321 first-party
Markdown links, starter CRC and all 955 inventoried hashes pass. Standalone
release integrity checks 955 selected files and 82 embedded sources.

Recursive inclusion and root/Prod parity pass for 19 Lean files, 12 theory files
and 16 paper files, including saved proof evidence. No new Lean/LaTeX compiler
or remote CI execution is claimed. No original data/checkpoint/returned ZIP/source
is removed, retuned or relabeled. No email, push, publication or remote/paid GPU
is launched. The only scientific work is the declared 3000 CPU adaptation updates;
original stronger synthetic readiness still fails and new OOD scoring is skipped.

Regenerate the final starter snapshot after these validation notes, run its
logged content check, then directly sync/check without logging either command
so the final append-only log stays in root/Prod parity. No scientific source,
configuration, endpoint or gate changed after the freeze.

## Stage 14: matched seen/held-out checkpoint audit, 31 October 2026

User authorizes the proposed next diagnostic. Declare no new fitting: five fixed
stage 13 final models, 128 training examples selected by outcome-blind base-ID
hash ranking, all 128 development examples, 32 per task/size cell. Freeze probes,
source, protocol and endpoint hashes before new metrics. Verify original training
exposures for every selected ID. Measure free generation, unconditional source/
one-hop/multi-hop correctness (invalids count wrong), signed/first-node errors,
and matched-cell teacher-forced answer/EOS and digit/other token scores.
Retain all controls, prior negative endpoints, original 80% criterion, failed
random-rule readiness and OOD gate. Add eight meaningful selection/accounting/
read-only/forward-equivalence tests before freeze. No prior scientific source or
weights are edited. Add source-group logging; derived report helpers follow.

### Stage 14 freeze and derived reporting

Eight new pipeline tests pass. Freeze before new metrics succeeds at
190ab24f24655072f8ebfe9f309d61135920aab57dd7c7611628db6fef57e393.
Start one five-checkpoint evaluation invocation; there is no fitting command.
Add explicitly selected report, independent decimal-vector/hop/token accounting,
CSV/chronology audit and public-only regenerated-probe/error reproduction helpers,
usage and a descriptive plot. Raw probe caches/weights stay excluded. No frozen
scientific source/config/protocol or earlier evidence is edited after freeze.

### Stage 14 completed primary diagnostic

All five fixed models finish one inference-only invocation, wall time 335.533s.
Every arm has 0/128 exact answers on BOTH probes. Training node accuracy ranges
29.55-32.89% versus development 18.95-21.60%. Training teacher-forced digit-bearing
token accuracy ranges 35.06-39.61%, versus 24.36-28.09% development; OTHER-token
accuracy is 100% training and 99.92-100% development. Each selected training
example has genuine earlier exposure; the new study performs zero scientific
optimizer updates. Existing development outputs reproduce stage 13 exactly.
All arms receive the frozen training-probe-competence-not-established flag.

The conclusion is a seen-example competence deficit WITH a held-out difference,
not a unique cause or a claim that generalization is irrelevant. Initiate separate
five-checkpoint replay of 1280 answers/teacher/error metrics. All 176 root tests
pass, including eight new pipeline checks. Derived reporting adds conditional
formatting versus digit-token ranges; no scientific code/data/protocol changes.
The next proposal is a separately frozen one-factor small-cache overfit control,
retaining original cache reference, architecture, codec, objective, budget and
all arms from the same pre-adaptation starts; no such fit is launched here.

### Stage 14 verified evidence and interpretation

Separate replay succeeds for all five checkpoints, 1280 generations/token IDs/
stop reasons and every teacher-forced, node-error and interpretation metric
(406.364s command wall time). Weights/checkpoint hashes and step-600 optimizer
states are unchanged. Export 22 exact copies, 40-cell CSV, summary, review and
scientific bar plot; visual inspection passes. Independent decimal-vector and
Floyd-hop accounting verifies exact/node/source/hop rates, malformed cases,
signed/first-node errors, teacher token partition/NLL sums, all CSV/summary values
and frozen chronology/privacy. Public-only probe bytes/independent labels and all
1280 generated-error scores reproduce without private weights/data/cache/network.
No scientific source/config/protocol edits follow the freeze, and no new fit is
launched. Initial AST helper parsing was repeated through the logger for the
retained verification record. Update canonical status/results/context/validation
and changelog. No causal mechanism, graph effect/no-effect or acceleration is
inferred; seen-example competence remains weak with a held-out difference too.
Full distribution checks and final direct sync follow.

### Stage 14 final distribution checks

Both root and standalone suites pass all 176 tests. Standalone public-only
regeneration reproduces exact matched probe bytes/IDs, independent labels and
all 1280 generated-error scores without private models/data/cache/network.
Independent accounting passes 22 exact evidence copies, 40 CSV cells/summary,
all signed/first-node/source/hop errors, teacher token partitions/NLL, flags,
original exposures and freeze-before-score chronology. Prior stage 13's 32 copies,
20 checkpoints/2560 prediction receipts and negative outcome stay intact.

All four notebooks regenerate. Source parity, 335 first-party local links and
starter CRC/all 992 content hashes pass. Standalone integrity checks 992 selected
files/82 embedded sources, original reference hashes and historical evidence.
Recursive mathematical source/evidence inclusion and root/Prod parity pass for
19 Lean files, 12 theory files and 16 paper files; no new Lean/LaTeX compiler
or remote CI execution is claimed. Original source/data/checkpoints/returned
ZIPs are preserved. New scientific optimizer updates, new OOD predictions,
remote/paid GPU jobs, emails, pushes and external publications: zero.

Regenerate the final starter snapshot after these notes, complete its logged
content check, then run final direct sync/check after all logged commands so
root/Prod logs remain in parity. No frozen scientific source, config, protocol,
probe, endpoint, target or interpretation rule changed after the freeze.

## Stage 15: small-cache overfit positive control, 31 October 2026

User authorizes the next controlled fit. Declare a nested 16-example cache (four
base graphs/size, both tasks) with the same outcome-blind salt/ranking as stage 14.
Reuse all five original stage 13 complete step-0 states and exact numerical
helpers/configuration: 600 updates, batch8, same optimizer/LR/loss/tokenizer/model.
Changing the cache implies repetition and token/length differences; fix 4800
presentations and report actual tokens, not an equal-FLOP claim. Each selected
example gets exactly 300 presentations. Reaggregate the verified 512-cache final
reference on these SAME 16 examples plus full original development, with imported
costs/updates distinct. Cache80% per cell requires 4/4 with this discretization.
Keep OOD/readiness gated regardless of fitting success. Add eight meaningful
selection, sampler, original-update-equivalence, reference-subsetting and frozen
criterion tests; source-group logging follows the new package. Freeze before fit.
No earlier scientific source/evidence or checkpoint is edited.

### Stage 15 freeze and execution

Eight new pre-freeze tests pass, including exact numerical update equivalence to
stage 13. Freeze succeeds at
2efa24ef3342b417e796078bdca5d1743e202bdee6e6bac532d51bd54cf05c65.
Each new arm's fixed exposure is 571200 unpadded inputs, 96000 answer/EOS targets,
708488 padded positions and 300 presentations/example. The historical reference
has 562855 /96016 /691560 respectively: record the inherent token differences,
not an equal-FLOP budget claim. Start one new five-arm fitting invocation.

Add explicitly selected derived report, 140-cell/independent-vector/token/budget
accounting audit, public-only data/order/reference/score reproduction, usage and
scientific plots while fitting proceeds. Preserve all private arrays/checkpoints,
old raw references and prior negative results. No frozen numerical source,
configuration, selection, target or endpoint is changed after freeze.

### Stage 15 completed primary fitting

All five fits complete one successful invocation, 1455.195s command wall time:
3000 NEW scientific CPU task-adaptation updates. Every arm gets 16/16 final
cache answers and 4/4 per cell. Observed pooled cache80% crossings are 300,100,
100,100,300 for scratch/grid/graph/prefix-grid/prefix-graph; no interpolation or
speed ratio. Imported same-cache references remain 0/16. All development scores
remain 0/128 exact, with validity 51.56-66.41%, node accuracy 8.61-12.34% and
macro-cell NLL 2.5892-2.8112: all worse than the 512-cache reference's continuous
metrics. Preserve BOTH fitting success and held-out regressions.

New CPU update timers total 1215.787310s, evaluation timers 217.229853s; imported
reference timing is not newly measured. All 184 root tests pass. Start separate
20-checkpoint/960-generation and teacher-score replay, imported reference
reconstruction, optimizer/initial-state/order/token-budget verification. Add
explicit same-development continuous-reference comparison in derived reporting,
retain the one-factor training protocol/criterion unchanged, and update canonical
status/results/context. No additional fit or OOD run is launched.

### Stage 15 completed replay and evidence

Separate replay passes all 20 checkpoints/960 new predictions and teacher scores,
initial identities, per-parameter optimizer counts, actual sampler/LR/token budgets,
final criteria/crossings and imported references (225.213s command wall time).
Export 52 byte-identical copies, summary, 140-cell CSV, review and scientific plot;
visual inspection confirms readable axes/labels, observed cache curves/targets and
zero held-out bars. Independent vector/cell/token/budget accounting, all CSV and
summary values, frozen source/chronology/privacy pass. Public-only selected-data/
labels/order/reference reconstruction and all 960 generated scores pass without
private weights/data/tokenizer cache/network. Preserve raw reference and new
private weights/optimizer states. No failure/recovery or post-freeze scientific
edit occurs. Update canonical status/results/context/validation/changelog with
BOTH fitting success and worse held-out behavior. Distribution checks follow.

### Stage 15 final distribution checks

Root AND standalone suites each pass all 184 tests. Standalone public-only
reproduction verifies selected data/labels/order, original reference subsets,
960 generated-answer scores and frozen flags without private weights/data/cache/
network. Fifty-two exact evidence copies, 140 cells/summary, 20 checkpoint replay
receipts, original starts/numerical configuration, actual 300 presentations per
example and all token/optimizer/chronology/privacy accounting pass. Stage 14's
22 evidence copies/1280 prediction receipts and negative outcomes remain intact.

All four notebooks regenerate. Starter CRC, all 1059 inventoried content hashes,
349 local Markdown links and standalone 1059-file/82-embedded-source integrity
pass, with original references and historical evidence retained. Recursive
mathematical source/evidence inclusion and root/Prod parity pass for 19 Lean,
12 theory and 16 paper files. No new Lean/LaTeX compiler or remote CI is claimed.
No original source/dataset/checkpoint/ZIP is deleted, changed or relabeled.
There are 3000 new CPU task-adaptation updates and 3000 imported reference updates;
no new PPT/language/GPU/OOD/official-language-test work, emails, pushes or external
publication. The fitting-only pass does not replace held-out/readiness failures.

Regenerate the final starter snapshot after these notes, run its logged content
check, then directly sync/check after all logger writes so root/Prod logs remain
in parity. No frozen scientific source/config/data/target/endpoint changed after
freeze. Do not extend or retune this completed study.

## 2026-10-31 — stage 16 retained-output error audit, before analysis

User authorizes the next study from stage 15's recommendation. Implement a
separately frozen descriptive audit of the two fixed update-600 variants on the
same 16 cache and 128 development cases, all five arms. Reuse retained raw
generations and teacher counters; there are no new model forwards or fits.
Predeclare all four task/size cells plus pooled, explicit invalid/valid-vector
denominators, unweighted hop roles for both tasks, signed/first-error counters,
token-micro digit/other scores and paired small-minus-reference counts. Keep
the original cache-only pass, failed full-ID and readiness/OOD gates unchanged.

The first logged pre-freeze unit invocation failed on one TEST fixture: its
new direct edge used weight 12 outside the generator's 1..9 range. Correct the
fixture to weight 8 and the corresponding truth; scientific data/code are not
retuned. The resolution invocation passes all 19 targeted audit/integrity tests.
The failed transcript is retained. No diagnostic result had been computed.

The append-only log was already 4,912,176 bytes before this stage. Extend only
the exact selected logs/checks.jsonl publication limit to 10,000,000 bytes so
the history can remain intact. Keep all other 5 MB limits and artifact rules;
test exact-path acceptance and both limit rejections. Logger records new stage
source identities. No original source, evidence or private artifact is changed.

### Stage 16 frozen execution and initial verification

Plan 279bd3b7ea385de061d9ec8c02c5a87b329c8c2d4b1bdd15a9a1c225aa357702
was frozen before the new analysis. The logged analysis took 13.676 seconds;
separate semantic verification took 13.699 seconds. New fitting, model forwards,
generations and optimizer updates are zero. All 1440 retained endpoint outputs
match imported semantics using a different parser/all-pairs hop algorithm;
aggregation is shared code and teacher counters are imported. All 100 diagnostic
and 50 paired CSV rows reproduce. Counter/teacher/hop/first-error partitions,
paired marginals, exact rendered review, privacy and recursive math inclusion
checks pass. No prior protocol, source, data, checkpoint or returned ZIP changed.

Small16 held-out valid-vector node-micro accuracy is 16.83–19.81%, and its
teacher-forced digit token-micro accuracy 19.18–20.57%. Source and one-hop errors
persist as well; formatting alone cannot account for poor prediction. Conditional
metrics select each model's own valid cases and do not isolate formatting
causally. All arms retain their cache-only fitting pass and failed ID/readiness
endpoints. No graph equivalence, acceleration, cost recovery or unique cause is
claimed. Intermediate64/fixed600-cache development is a separately frozen next
proposal, not a new fit executed in this study.

Implementation disclosure: the inherited generate() recreates the entire
deterministic bundle, including unused assessment/OOD rows, before ID selection.
Thus the frozen blanket "No OOD data" wording is too broad. Preserve it and the
executed sources; record the scope deviation in the regeneration note and review.
No assessment/OOD row is predicted/scored/used for selection or given to any
model; no new model forward occurs. No official language test, remote/paid job,
email, push or external publication occurs. Original eligibility gates persist.

### Stage 16 final verification and distribution

Root and standalone suites each pass all 193 tests. Public-only standalone
replay independently parses/hop-scores all 1440 retained outputs without private
weights/data, tokenizer cache or network. Root and standalone counter/CSV/review
accounting pass. The stage 15 preservation check passes its 52 exact evidence
copies, 20/960 existing neural replay receipts, original numerical/source/start/
reference identities, 300 presentations per selected example, token budgets,
140 CSV rows and unchanged decisions. It does not rerun those neural forwards.

All four notebooks regenerate. Starter CRC/1077 content hashes and 365 local
Markdown links pass. Standalone inventory passes 1077 selected files, 82 embedded
sources and pinned-reference/historical integrity checks. Inclusion/parity
checks retain 19 Lean, 12 theory and 16 paper source/evidence files. No new Lean,
LaTeX, remote CI, GPU, scientific fitting, email or publication is claimed.
The append-only log exceeds 5 MB and remains intact under its tested exact-path
10 MB bound. Other limits, private-artifact exclusions and all gates stay intact.

Regenerate the final starter after these notes; after its logged integrity
check, finish direct unlogged sync and --check to retain root/Prod log parity.
No stage 16 frozen source/protocol/input/endpoint is changed after its freeze.

## 2026-10-31 — stage 17 preparation, before any new fit

User authorizes the intermediate64-cache/600-update development study. An
unfinished local cache_v17 draft/config/protocol exists with no frozen run or
active Python process. Preserve its initial five-file snapshot privately under
tmp/cache_v17_initial_draft, then complete it before freezing. No previous frozen
source/protocol/evidence is edited. A read-only CIM process inventory was denied
by the host; ordinary Get-Process shows no Python training process. The first
large patch did not apply because a draft status string differed; no partial
mutation or scientific execution occurred. Apply the corrected patch.

New64 uses the same salted nested ranking: common16 <= cache64 <= old train128
<= train512. Retain all five original complete stage13 step0 starts, fresh
optimizers, exact numerical training helpers/config, decoder and criteria.
Each example receives exactly75 presentations; all arms share full actual order.
Imported512 is reaggregated onto cache64/common16/dev128; imported16 onto only
common16/dev128. Never fill its unmeasured extra48 cases with zeros. New common16
metrics are subsets of final cache64 outputs/counters, with no extra forward.

Primary cache fit requires >=13/16 in all four cells for every arm, >=95% pooled
validity and complete coverage. Full dev criterion remains separate. Preserve
600 updates, fixed0/100/300/600 checkpoints, no best-model selection or retuning.
Plan 3000 new CPU task-adaptation updates; 6000 reference updates are imported.
No new PPT/language/GPU/OOD/official-test work, email, push or paid job. Public
reconstruction may recreate unused synthetic OOD rows in memory, explicitly
declared in protocol; only the three declared ID probes enter new fit/scoring.

### Stage 17 freeze and fit launch

All14 targeted new/inherited tests pass before freeze, including exact numerical
update equivalence, nested-selection/missing-reference failures and integer
criterion boundaries. Freeze succeeds with plan
6b52c649bdfab7fd8d43d11a308e62606a0e285a10f9210db2a502a629195028.
The fixed per-arm accounting is561300 input tokens,96000 supervised targets and
692880 padded positions, with75 presentations per selected example. The logged
five-fit command is launched after that freeze; scientific sources/config/data/
targets stay unchanged. Public report/replay/accounting helpers are prepared
separately while fitting runs. All historical data/checkpoints/ZIPs are preserved.

All199 root regression tests pass (19.478 seconds command wall time), including
the six new cache-scope/boundary checks. Those tiny fixtures are not scientific
benchmark fits. The regression invocation briefly overlaps the final prefix-
graph fit; report observed CPU costs without controlled-throughput ratios.
No frozen scientific source/configuration/criterion is edited during fitting.

### Stage 17 fixed fit endpoint; checkpoint replay launched

The five-fit command exits0 in1611.880 seconds (about26.9 minutes). Every arm
gets64/64 correct cache vectors, including16/16 in every cell, and0/128 exact
held-out vectors. All five observed pooled80% cache crossings are at300 updates;
these are coarse descriptive points, not graph-specific learning-speed ratios.
New64 development node-example means are scratch12.8125%,grid12.5391%,graph17.0898%,
prefix-grid12.1680%,prefix-graph18.7109%. The matched graph prefix control exceeds
graph; none meets held-out competence. Actual cost sums:1163.269 seconds CPU
updates and403.728 evaluations, excluding provenance/IO/replay. No target,
checkpoint, arm, training duration or input is selected/changed from these results.

Launch the separately logged replay invocation after fit exit0. It will reopen
all20 checkpoints and reproduce1920 new generations/teacher scores, common16
subsets, actual optimizer/order/token identities and both reference extractions.
That replay is pending here; no completed verification is inferred from launch.

Derived review interpretation retains continuous improvements: new64 held-out
answer/EOS NLL is2.0393–2.1876, lower than imported16 in every arm, but imported512
has both lower NLL and higher node accuracy in every arm. New64 node changes
versus16 are mixed; prefix-grid is slightly lower. Keep all15 variant/arm rows,
all220 task/size cells and fixed zero exact endpoints, not only favorable scores.
A direct syntax-only AST inspection of the derived renderer was repeated through
the required logger together with the other two derived tools; all parse. No
new scientific forward or source/config/target change is made by this edit.

### Stage 17 replay, exported evidence and independent checks complete

The separate replay exits0 in562.925 seconds: all20 checkpoints,1920 new
generations/teacher scores, final/common16 decisions, actual optimizer/order/LR/
token counts, original complete starts and both reference extractions reproduce
exactly. This is same-code replay, not independent neural implementation or a
second fit. Export72 byte-identical copies under docs/results/cache_v17_run1,
plus220-cell CSV, summary, review, plot and provenance. No private weights,
optimizer states, order arrays or raw cache data are distributed.

Independent decimal-vector/hop/sign/cell/token accounting, actual75 exposures,
source/config/start/reference identities, all220 rows/review and frozen log
chronology/privacy checks pass. Public data/label/order/both-reference/1920-score
reproduction passes without private weights/data, tokenizer cache or network.
The plot is visually inspected: all recorded curves/legends/axes, pooled80% line,
coincident-curve disclosure and same-dev128 three-variant node bars are correct.
Keep all metrics, controls, original failures and gates. Append descriptive
results to canonical research/validation/result docs; no frozen science changes.

### Stage 17 standalone and final distribution verification

All199 standalone regression tests pass. Public-only standalone reconstruction
also passes cache64/common16/dev128 data/labels/order, both references and1920
new generated-answer scores/decisions without private weights/data, tokenizer
cache or network. All four notebooks regenerate;377 first-party Markdown links,
starter CRC and all1164 inventoried content hashes pass. Root/Prod recursive
source/evidence inclusion/parity passes for19 Lean,12 theory and16 paper files.
No new Lean/LaTeX compiler, remote CI, language/PPT/GPU/OOD run or official
language-test evaluation is claimed. No email, push or external publication.

Update the final validation notes and regenerate the final starter snapshot;
after the last logged integrity check, directly sync and check Prod so logger
appends remain in parity. Preserve all original results, checkpoints, datasets,
ZIPs and pinned sources. The fitting-only pass does not replace full-ID,
categorical/random-rule readiness or graph-transfer confirmation requirements.

## 2026-10-31 — stage 18 exploratory ID acceleration, before fitting

User explicitly authorizes the proposed exploratory acceleration experiment.
This supersedes the next-development-audit proposal for THIS new study only;
no earlier frozen protocol, eligibility bit or endpoint changes. Use five original
complete stage13 step0 models after identical historical language stages. Keep
the architecture/codec/objective/AdamW/600 updates fixed across arms. Use a fresh
ID resample and shared order seed2026103118, train512/dev128, no assessment rows.
This is one inherited initialization cohort, not independent confirmation.

Primary semantic target: every task/size cell's unconditional NON-SOURCE node
accuracy exceeds its frozen source-aware train-marginal predictor by10pp,
pooled validity>=95% and full coverage. Freeze the deterministic baseline and
its same-case targets before any neural fit. Exact vectors and the historical
80% criterion remain separate, not retrospectively relaxed. Seven evaluations
at0/100/200/300/400/500/600, no early stopping; report actual updates/tokens/time,
right-censoring, final regressions and sustained-crossing sensitivity. Graph
must be compared against BOTH grid and prefix_graph, with scratch/prefix_grid.

Bind all original v4 language/PPT cost summaries and the data manifest. Original
hardware is verified as Tesla T4 with CUDA; new fitting is CPU. Report components
and mixed-hardware accounted sums, never controlled end-to-end throughput or
automatic cost recovery. Full paired preparation charges remain conservative
as in v4. No new PPT/language/GPU/OOD run, official test, paid job, email or push.

All16 targeted new/inherited tests pass before freeze: semantic shortcut/invalid
failures, per-cell and format requirements, first versus sustained crossing,
proper censoring/zero-denominator ratios, cost ledger and numerical updates.
Read-only discovery included a Windows rg wildcard error and a missing optional
historical summary path; use the actual retained manifests and language/PPT
summaries. Several draft patches with placeholder context did not apply; no
partial mutation, failed scientific fit or discarded data occurred. All final
scientific sources/config/protocol will be frozen before the first update.

### Stage18 freeze and matched fit launch

Freeze exits0 in40.243 seconds with plan
1477e7887baea612119c09717242061d000c7c1a59df44407a3c8f6e24767076.
Shared per-arm training exposure is562823 input tokens,96024 supervised targets,
696256 padded positions over600 updates. Frozen baseline non-source scores are
BFS8:96/224,BFS10:129/288,Dijkstra8:14/224,Dijkstra10:16/288; primary targets
are those fractions+0.10 in EVERY cell, plus95% validity/coverage1.
Launch the logged five-fit command only after freeze exit0. All scientific
sources/config/data/targets remain unchanged. No new model outcome informed the
baseline or target. Prepare derived export/replay/accounting tools separately.

The4480 per-case results are expected to exceed the ordinary5MB publication
bound (stage17's1920-case file was4,008,358 bytes). Preserve exact raw metadata
with a bounded12MB exception for ONLY docs/results/accel_v18_run1/results.json;
all weight/corpus/cache exclusions and other5MB limits remain, with log10MB
unchanged. Add exact-path/boundary regression checks; this packaging policy does
not alter frozen scientific inputs, fitting or targets. Private raw data/weights
remain excluded. No target adjustment, model selection or extra updates occur.

### Stage18 fixed exploratory fit endpoint; replay launched

All five matched fits finish with exit0 in2561.106 seconds (42.685 minutes).
All five arms are right-censored at600 updates; none of the140 observed semantic
cell scores reaches its frozen individual target either. There is no finite
time-to-target ratio, no infinite speedup and no imputed attainment at the cap.
Every final exact-vector score is0/128. One exact answer occurs only at scratch
update500 and is retained along with the final regression. Do not select it.

Final whole-node example means: scratch20.5859%,grid21.6602%,graph20.7813%,
prefix-grid21.5234%,prefix-graph20.6055%. All final vector-validity rates are100%.
New CPU timers sum1375.383 training seconds and1095.985 evaluation seconds;
actual command wall includes provenance/setup/IO. Imported T4 costs remain
separate; fixed arm order and differing hardware prevent controlled end-to-end
throughput claims. No threshold, data, optimizer setting or budget is changed.

Launch the separate35-checkpoint/4480-prediction replay after fit exit0; it is
pending at this entry. Start the full regression suite after scientific fitting,
so those tiny execution fixtures do not overlap the new training timers.
The9,285,187-byte raw results file fits its exact-path12MB publication bound;
no original data/result/checkpoint is dropped or compacted to fit a size cap.
Derived reports disclose that CPU elapsed excludes model/optimizer construction,
while command-wall logs retain shared setup/teardown without a fabricated per-arm
allocation. Original weaker v4 PPT gate flags are retained but do not replace
the later failed stronger synthetic-readiness check.

All208 root regression tests pass after scientific fitting. A separate private
score preflight initially fails BEFORE executing its Python body because nested
f-string quotation is mangled by PowerShell. Preserve that exit1/transcript;
replace the invocation with single-quoted path components and string concatenation
under a new resolved label. No scientific source, checkpoint, target or dataset
changes; the full neural replay remains active and independent of this command.

The resolved private score preflight passes all4480 independently parsed outputs,
semantic/cell and teacher-counter aggregation plus imported cost extraction.
No neural forward or fit is performed by this score-only check. While full neural
replay remains pending, update canonical RESULTS/RESEARCH_CONTEXT with explicit
pending status, fit endpoints, right-censoring and costs; do not claim public
export or completed checkpoint verification before its actual receipt.

## 2026-11-01 — stage18 neural replay and public evidence

The separate neural/teacher replay exits0 in970.423 seconds: all35 checkpoints,
4480 predictions, initial full tensor/optimizer identities, actual order/LR/token
counts, baseline/target/censoring/contrasts and imported costs reproduce exactly.
Original clock observations are retained; this is not a second fit or independent
neural implementation. All earlier pending documentation is updated only after
this receipt. No extra optimizer update or eligibility change is introduced.

Export60 byte-identical copies,140-cell CSV, summary, review, plot and provenance.
The bounded9,285,187-byte JSON retains all cases; weights/optimizer arrays/caches
remain private. Full-budget spending is separately labeled and contains no
misleading reached=True field for censored target crossings. Historical weaker
PPT gate flags are explained; costs and token sums are not equal-FLOP/$ claims.
Public-only ID data/labels/order/train-marginal/cost/4480-score reconstruction
passes without weights/private data/tokenizer cache/network. Source-aware
semantic decisions also agree with an independent decimal parser. The plot is
visually inspected: four task/size cells, all five curves/seven recorded points,
gray marginal baselines, black +10pp targets, common0..1 axes and correct labels.
No cell reaches its target; no best point, altered threshold or omitted arm.

### Stage18 independent/public checks and final distribution

The evidence audit passes60 exact copies,35/4480 replay receipts, full numerical
starts/token/order/LR/optimizer budgets, baseline/cost/target/censoring/140-cell
and source/chronology/privacy checks. Standalone also passes all208 tests and
public-only ID data/order/baseline/cost/4480-score reconstruction. The sole failed
check remains the retained shell-quotation preflight and its passing resolution;
no scientific fit/replay fails or is restarted. Original source/data/weights/
checkpoints/ZIPs and all older negative endpoints remain unchanged.

All four notebooks regenerate;393 local links and starter CRC/all1239 inventoried
hashes pass. Standalone publication verifies1239 files and82 embedded sources,
with pinned-reference/historical integrity. The stage17 preservation audit
passes72 copies,20/1920 old replay receipts,220 cells, references and fixed
decisions, without reexecuting those neural forwards. Recursive root/Prod
membership/parity passes19 Lean,12 theory and16 paper files, including prior
verification records. No new Lean/LaTeX compiler, remote CI, GPU/PPT/language/OOD,
official language test, paid job, email, push or external publication occurs.

Regenerate the final starter after these final notes; after the last logged
integrity check, directly sync and --check Prod to preserve log parity. All
scientific stage18 files/config/protocol/data/targets stay frozen throughout.
This exploratory all-censored result does not claim equality, zero graph effect,
universal failure or positive acceleration; old confirmation/OOD gates persist.

## 2026-11-01 — stage19 checkpoint-only fitting/generalization diagnostic

User requests continued work from stage18's preserved all-censored endpoint.
Choose the unresolved seen-example fitting question before spending another
training budget. Freeze all five original stage18 update600 checkpoints, a
new outcome-blind balanced128-example SEEN subset of its512 training examples,
and its full128 development reference. The new salted base-ID ranking keeps both
tasks per graph and32 examples/cell, without outcome/label/difficulty selection.
Use the same train-fitted marginal predictor on each probe's own cases, preserving
the original held-out baseline/targets exactly. Keep old exact calibration separate.

Import all600 update records per arm and predeclare six100-update blocks plus
first50/last50 summaries, actual exposure and pre-clip gradient rates. Per-arm
interpretation distinguishes seen-probe target failure from a held-out gap after
seen competence; do not infer a unique optimization/representation cause. New
work is640 seen predictions/teacher scores from five unchanged endpoints, then
separate replay. The640 held-out outputs are imported, not new neural evaluation.
New optimizer/PPT/language/GPU/OOD/official-test/remote/paid work iszero.

All13 targeted new/inherited tests pass before freeze: balanced outcome-independent
selection, every selected case's9/10 actual presentations, fixed trace blocks,
nonfinite/missing trace failures, probe-specific semantic baselines, descriptive
categories and unchanged target/censoring logic. Keep all previous sources,
protocols, negative endpoints, datasets, weights, ZIPs and confirmation gates.

## 2026-11-01 - separate scientific-transfer foundation

User requests implementation in a separate directory within root. Create
scientific_transfer with an explicitly declared steady constant-property bridge,
not a silent transcription of the CEJA manuscript. Audit energy Pr factors,
similarity definitions, mixture properties, flux scaling and shooting versus
collocation. Original PDF remains unchanged/outside distribution; retain its
hash. Neural training and manuscript replication remain disabled. Five-arm
matching is draft pending training-interface/budget freeze. No historical
runner/checkpoint/protocol/result changes. Current stage18 is all-censored.

Implement reference solves, tighter/longer verification, tuple-level split
identities, source/config receipts, retained failures, denominator-aware metrics,
supplied-guess correction and immutable outputs. Extend append-only logger
source identities; explicitly select new directory in publication.json. Keep
runtime solutions under ignored runs. No new model training or physical claim.

Dependency preflights find SciPy absent in local/bundled Python and local pip
absent. UV restricted-network DNS fails; authorized network escalation installs
SciPy1.17.0 into local venv. All failures/resolution retained in checks.jsonl.
First ten-test check has one failure: linear interpolation adds 0.000234 profile
error despite ~3.6e-9 wall agreement. Replace interpolation with cubic Hermite
using RHS derivatives; keep the 1e-4 criterion unchanged. Failed test retained.
A documentation patch with a wrong logbook heading applied no changes; use an
append entry. Validation outcomes follow after execution.

### Scientific-transfer pipeline verification complete

All10 new numerical/integrity tests pass. Existing root regression suite passes
213 tests (including its own tiny training fixtures, not scientific fits).
Eight train/four development reference cases pass the unchanged1e-4 agreement
criterion; maximum profile discrepancy9.623207e-8, wall2.194326e-9. Four analytic
supplied guesses correct to their references exactly. They equal the ordinary
baseline guess: pipeline verification only, no neural speedup. Retain full raw
artifacts under runs/scientific_transfer/dev1 and compact public source/input/
output hash receipt and review under docs/results/scientific_transfer_foundation.
No assessment/extrapolation solutions, new neural scientific updates, GPU,
PPT/language/OOD/official-test measurement or remote/paid run. Initial setup and
interpolation failures/resolutions stay in the append-only log. Regenerate all
four notebooks and final-sync selected root sources/evidence to Prod, including
all Lean/theory records; copying them claims no new proof/LaTeX compilation.

### Scientific-transfer packaging retry

All four notebook build commands were executed; v2/v3/v4 pass. The first v1
starter packaging detects logs/checks.jsonl changing during packaging and exits1;
retain that failure and rebuild a stable snapshot. Documentation check passes
399 first-party local links. No scientific data/result is changed by packaging.

### Scientific-transfer final distribution

V1 packaging retry passes; preserve original failed attempt. Source-bound plan
integrity and recursive publication selection pass for41 scientific/Lean/theory
files. All four notebooks regenerated; no proof/LaTeX compiler run is claimed.
After these final notes and the last logged check, regenerate the starter and
run direct sync/check so logger appends cannot leave root/Prod log drift.
No external publication, push, paid run or learned scientific result occurs.

### Stage19 frozen execution and independent verification

Freeze exits0 in59.019 seconds with plan
959921b6a2852f0075ef2ca677c82af6dcf15134359c490beeb3360bd616359e.
All five fixed endpoint seen evaluations exit0 in265.287 seconds; summed new
neural evaluation timers151.558 seconds. Every arm fails its seen semantic
target AND held-out target, with0/128 exact vectors on both. Seen whole-node
accuracy23.48–28.65%, versus20.59–21.66% held out; gold-prefix seen digit accuracy
34.0–37.7% remains weak. First50 mean losses3.038–3.320 fall to0.834–0.863 in the
last50 updates. These results leave seen-probe fitting unresolved and do not
identify a unique cause or guarantee that more updates repair generalization.

All213 root tests pass. Separate same-code replay exits0 in213.673 seconds,
reproducing five unchanged endpoints/640 new predictions/teacher scores and
fixed decisions/traces. Export36 exact copies plus40-cell/30-block CSVs, summary,
review, plot and provenance. Public-only reconstruction passes ID probes,
baseline identities,1280 new/imported generated semantic scores,3000 actual
trace rows and all tables/decisions without private weights/data/cache/network.
Private accounting passes exact copies, actual encodings/tokens, optimizer/count
receipts and frozen source/chronology/privacy. Teacher logits still require
weights. Bar palette inspection confirms all four model/marginal series and
the visual trace plot retains all blocks; no scientific result is altered.

Another session has added a separate scientific_transfer foundation and selected
it for distribution while this diagnostic ran. Preserve its code, evidence,
logbook entries, dependency/test/failure receipts and publication entries; do
not overwrite its work. Ask whether editing remains active and continue frozen
stage19 verification independently. Shared files remain stable and its own final
log marks verification complete. Package the current root with that work retained,
checking snapshot/parity rather than claiming its tests as newly run here. No
frozen stage19 or original science input is changed. Final mathematical inclusion
still covers all Lean/theory/paper sources/evidence without a new compiler claim.

### Stage19 final distribution verification

Standalone also passes all213 root-style regression tests and public-only ID
probe/baseline/1280-answer/3000-trace/40-cell/30-block reconstruction. Standalone
inventory passes1303 selected files,82 embedded sources and pinned-reference/
historical integrity. All four notebooks regenerate;411 local links and starter
CRC/all1303 content hashes pass. Stage18's4480 output/baseline/cost/target/
censoring reproduction stays unchanged; no old neural inference is rerun by
that preservation check. Root/Prod recursive inclusion/parity passes19 Lean,
12 theory and16 paper files, including prior verification records. No compiler,
remote CI, GPU/PPT/language/OOD/official-test run, paid job, email or push occurs.

Preserve the other session's separately selected scientific_transfer package,
docs/receipts/test/failure evidence and all logbook entries. Its files remain
stable while this package's snapshot/inventory/parity checks pass; do not relabel
that separate scientific verification as newly executed here. No frozen study
input, model or endpoint is overwritten. Finish with final starter regeneration,
logged integrity check, then direct unlogged sync and --check to retain parity
after all logger writes. Stage19 adds zero scientific optimizer updates and
does not loosen any earlier target or confirmation/OOD condition.

## 2026-11-01 - next scientific-transfer numerical milestones

User requests implementation of next work according to the five milestones.
Read context/audit and current separate benchmark. Use PDF skill to render and
visually inspect manuscript pages9-13, preserving original PDF and all earlier
source/data/results. Search exact title/number finds no exact public author code
or reference; related papers are not exact manuscript substitutes. Withdraw the
suspected Eq11/20 Pr mismatch: visual fraction bar covers the full numerator,
so the two printed forms agree. Earlier concern came from text extraction.
Confirm printed density(1-1)rho_f fails zero loading; note printed momentum
geometry/dimensional/magnetic problems without silently choosing author intent.
Faithful reproduction remains blocked; milestone1 is only partially resolved.

Declare steady_hybrid_v2 BEFORE numerical checks: independent steady local-pole
boundary-layer derivation, final-volume Zn/TiO2 mixture policy, constant-property
viscosity, explicit sequential Maxwell conductivity, conservative cubic radiation
and named dimensionless wall outputs. Table1 inputs are imported/unvalidated,
not new material measurements. No unsteady/heat-source/global-CFD claim. Add
independent Chebyshev/root Homann plus spectral thermal solver, not solve_bvp or
initialized from its solution. Freeze four-case material/tolerance/mesh/domain
and independent49/65-node1e-5 wall checks before execution. Nine new focused
unit tests pass. No neural fitting, checkpoint import, threshold relaxation,
reserved data or old eligibility change. Later milestones3-5 remain pending
separate neural freeze/ID competence and full heterogeneous stage-cost accounting.

### Scientific-transfer hybrid numerical checks complete

Frozen plan11b? See authoritative PLAN.json SHA rather than a manual identifier.
All four declared v2 cases pass base/tolerance/mesh/domain checks: maximum
profile2.4200894622e-9 and wall4.0103629306e-9 under unchanged1e-4 criterion.
Independent49/65 Chebyshev/root+linear-thermal references pass1e-5 agreement:
max fpp error2.2594370819e-9, thermal8.0424511495e-10. Full internal numerical
check timer0.2718832 CPU seconds excludes startup/imports/IO/packaging. All19
scientific tests and213 existing root regressions pass, including prior test
fixtures only, not new scientific neural fitting. No failed scientific check
is discarded. Export two byte-identical freeze/result files plus provenance
and review under docs/results/scientific_transfer_v2, selected recursively by
the existing docs publication policy. Original v1 artifacts and old negative
neural endpoints remain unchanged. Keep manuscript audit partial and faithful
replication disabled; stages3-5 neural/ID/OOD/full-cost work remain pending.
Regenerate all four notebooks and final direct Prod sync/check after logged
checks; recursively preserve all proof/theory/paper sources/evidence without
claiming new Lean/LaTeX compilation, CI, remote training, paid run or publication.

### Scientific-transfer v2 final source distribution checks

Plan identity, clarifying the provisional phrase in the prior entry:
2db65ce93f2863043b79f9148a6d8cede9d05a77decf24feda0abf2a54494155.
Public exact-copy/plan/source identities pass and418 local Markdown links pass.
All four notebook builds pass. No new neural science, reserved outcome, compiler,
paid/remote run or external publication is claimed. After final logged recursive
selection check, rebuild the starter and run direct Prod sync/check to avoid
append-only logger drift. Preserve earlier PDFs/renders, ZIPs, sources/evidence.
