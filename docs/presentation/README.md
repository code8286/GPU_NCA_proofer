# Detailed research presentation

[Open the 92-slide PowerPoint](GPU_NCA_Research_Detailed_v2.pptx). The supplied [original descriptive deck](../GPU_NCA_Proofer_Presentation.pptx) is preserved.

The main narrative runs through slide 76. Slides 77–92 provide source navigation, proof coverage, additional derivations and retained diagnostic plots. Speaker notes identify the sources, code locations and interpretation for every slide.

## Coverage

| Slides | Subject |
| --- | --- |
| 1–8 | Problem, preliminary targets, literature, hypotheses and present results |
| 9–25 | Typed GNN/NCA dynamics, topology visualization, entropy, serialization and actual tokenizer/masking code |
| 26–40 | Backbone transfer, Taylor derivation, linear and quadratic examples, first hits, costs and Lean proofs |
| 41–58 | Historical methodology, corrected v2 controls, actual dataset frames, histograms, transformer code and notebooks |
| 59–67 | Measured v2 language gains, paired losses, cumulative costs, verification and alternative explanations |
| 68–76 | Prepared v3 graph experiment, actual v3 code/notebook, larger-scale applications and conclusion |
| 77–92 | Documentation, complete slide index, proof map, references, further derivations and plot appendix |

Four native charts and 26 native tables (including the decoded state matrices) remain editable. Code excerpts and displayed mathematics are editable text. Existing plots and the Lean verification screenshot remain image evidence. The repository retains all three complete notebooks and the full LaTeX/Lean sources.

## Interpretation

The measured v2 gain concerns subsequent language learning after synthetic pre-pre-training. Grid and shuffled initialization both show about 1.178× geometric language-token speedup. Grid cumulative-time speedup is 0.886×. The presentation does not describe these as faster individual updates, a demonstrated dynamics/graph advantage or recovered synthetic cost. V3 GPU outcomes remain pending.

The entropy and spectral arguments remain informal. Lean checks a selected 44-statement core, with the Taylor bound supplied as a hypothesis. Product reset identities are formalized; its differential is not. The presentation separates exact numerical toy examples from imported GPU measurements and execution fixtures.

## Traceability

[SOURCE_COVERAGE.json](SOURCE_COVERAGE.json) records source hashes, all three notebooks, code line ranges, sampled dataset identities, the figure inventory and slide-by-slide sources. The inventory captures authoring-time research sources. Binary datasets and weights remain local; the deck contains selected decoded state values and compact reviewed evidence. Dependency caches and unrelated administrative material do not supply research claims.

The four chart workbooks retain their data. The original cumulative-token workbook is copied unchanged, including its formulas. The three new charts have workbook snapshots of their displayed literal data. Toy-curve display values use 12 significant digits, while the manuscript and Lean statements retain the exact definitions.

## Complete slide index

| Slide | Topic | Primary source references |
| --- | --- | --- |
| 1 | GNNs in Neural Cellular Automata | README.md; RESULTS.md |
| 2 | Research map | RESULTS.md; RESEARCH_CONTEXT.md; docs/theory/main.tex |
| 3 | Problem statement and research question | RESEARCH_CONTEXT.md; METHODOLOGY_AUDIT.md |
| 4 | Preliminary targets and present status | docs/results/transfer_v2_run1/REVIEW.md; RESULTS.md |
| 5 | Literature review | https://distill.pub/2020/growing-ca/; https://arxiv.org/abs/1704.01212; https://arxiv.org/abs/1706.03762; https://arxiv.org/abs/2302.13971; https://arxiv.org/html/2603.10055v1 |
| 6 | Reference study and this development experiment | https://arxiv.org/html/2603.10055v1; sources/nca-pre-pretraining-reference/reference.json; configs/transfer_v2.json |
| 7 | Hypotheses and falsifiable comparisons | RESEARCH_CONTEXT.md; docs/TRANSFER_V2_PROTOCOL.md; docs/TRANSFER_V3_PROTOCOL.md |
| 8 | Measured v2 transfer and cumulative cost | docs/results/transfer_v2_run1/aggregate.json |
| 9 | Theory: dynamics, graph and transformer | docs/theory/sections/01_problem.tex; docs/theory/sections/02_dynamics.tex |
| 10 | GNN visualization from a saved topology | data/cpu_check/grid_edges.npy; data/cpu_check/graph_edges.npy; experiment.py |
| 11 | Actual message-passing code | transfer_v2/nca.py:36-40; SHA-256 4b7bbf83f70614d2feca5619a68f4313a18172900d0fb3397d14d96f0d0bc52f |
| 12 | Grid convolution equals typed message passing | docs/theory/sections/02_dynamics.tex: Proposition grid equivalence; transfer_v2/nca.py:convolution_logits; tests/test_transfer_v2.py |
| 13 | Relabeling equivariance and swap invariants | docs/theory/sections/02_dynamics.tex; proofs/lean/GraphNCA/Graph.lean |
| 14 | Exactly solvable diffusion surrogate | docs/theory/sections/02_dynamics.tex; docs/theory/numerical_checks.json |
| 15 | Degree preservation does not guarantee mixing | docs/theory/sections/02_dynamics.tex |
| 16 | Exact perturbation and a logit-margin condition | docs/theory/sections/02_dynamics.tex; docs/theory/numerical_checks.json |
| 17 | Information bounds for deterministic trajectories | docs/theory/sections/03_information.tex |
| 18 | Rule uncertainty and stochastic transitions | docs/theory/sections/03_information.tex |
| 19 | Contextual prediction and the shuffled control | docs/theory/sections/03_information.tex; transfer_v2/nca.py:shuffled |
| 20 | Serialization: exact accounting and parent lags | docs/theory/sections/04_serialization.tex |
| 21 | A decoded categorical patch from the actual dataset | Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/data/development/nca_validation.npy, rule row 0, frame 0; transfer_v2/nca.py:encode_frames |
| 22 | Actual 2 × 2 patch-token encoding | transfer_v2/nca.py:76-81; SHA-256 4b7bbf83f70614d2feca5619a68f4313a18172900d0fb3397d14d96f0d0bc52f |
| 23 | Parent visibility and crop boundaries | docs/theory/sections/04_serialization.tex; proofs/lean/GraphNCA/Serialization.lean |
| 24 | Mean lag is fixed while dispersion can change | docs/theory/sections/04_serialization.tex; proofs/lean/GraphNCA/Graph.lean |
| 25 | Actual synthetic target mask | transfer_v2/nca.py:157-164; SHA-256 4b7bbf83f70614d2feca5619a68f4313a18172900d0fb3397d14d96f0d0bc52f |
| 26 | Transfer mathematics and an exact convergence example | docs/theory/sections/05_transfer.tex; docs/theory/sections/06_numerical.tex |
| 27 | Actual stage-boundary transfer code | transfer_v2/model.py:92-97; SHA-256 4394c7c226ec72113925da5fb9c50949361d652c5d47f54926e399a5b8f8c6b7 |
| 28 | Taylor derivation for backbone transfer | docs/theory/sections/05_transfer.tex: eq:displacement and prop:alignment; proofs/lean/GraphNCA/Transfer.lean |
| 29 | Worked gradient-alignment calculation | docs/theory/sections/05_transfer.tex; docs/theory/numerical_checks.json |
| 30 | Exact topology-to-target surrogate | docs/theory/sections/05_transfer.tex; proofs/lean/GraphNCA/LinearSurrogate.lean |
| 31 | A six-node numerical topology example | docs/theory/sections/05_transfer.tex: exact topology surrogate; docs/theory/numerical_checks.json |
| 32 | Quadratic convergence and ranking reversal | docs/theory/sections/05_transfer.tex; proofs/lean/GraphNCA/Quadratic.lean |
| 33 | Closed-form toy loss curves | docs/theory/sections/06_numerical.tex; proofs/lean/GraphNCA/Numerics.lean |
| 34 | Exact numerical first hits | docs/theory/sections/06_numerical.tex; proofs/lean/GraphNCA/Numerics.lean |
| 35 | Break-even derivation and numerical costs | docs/theory/sections/06_numerical.tex; proofs/lean/GraphNCA/Transfer.lean |
| 36 | Parameter and token arithmetic | docs/theory/sections/06_numerical.tex; transfer_v2/model.py; configs/transfer_v2.json |
| 37 | Lean 4: verified coverage and explicit boundaries | proofs/lean/PROOF_MAP.md; proofs/lean/verification/status.json |
| 38 | An actual Lean equivariance proof | proofs/lean/GraphNCA/Graph.lean:19-24; SHA-256 72cf666ca49daa9f75f780b596c4b1bc8418d0540443025d945f81d52bef6d67 |
| 39 | An actual Lean first-hit proof | proofs/lean/GraphNCA/Numerics.lean:60-66; SHA-256 bcca26b251d6a5c31c0abd4ce12870572cd46351bae064a8d5a3283872b94b02 |
| 40 | Formal proof failures, corrections and trust | proofs/lean/verification/README.md; proofs/lean/verification/attempt-1-diagnostics.txt; proofs/lean/verification/final-diagnostics.txt |
| 41 | Historical pilot 1 and its exposure confound | docs/results/pilot1/REVIEW.md; METHODOLOGY_AUDIT.md |
| 42 | Historical pilot: language-exposure alignment | docs/results/pilot1/language_aligned_curves.png; docs/results/pilot1/prepretraining_diagnosis.json |
| 43 | Methodology across three versions | docs/PILOT_PROTOCOL.md; docs/TRANSFER_V2_PROTOCOL.md; docs/TRANSFER_V3_PROTOCOL.md |
| 44 | V2: identical language-stage controls | configs/transfer_v2.json; docs/results/transfer_v2_run1/REVIEW.md |
| 45 | Dataset preparation and rule separation | transfer_v2/nca.py:generate_split; transfer_v2/data.py; docs/results/transfer_v2_run1/REVIEW.md |
| 46 | Saved dataset sizes and screening outcomes | docs/results/transfer_v2_run1/REVIEW.md; Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/data/development/manifest.json |
| 47 | Actual NCA states from the retained v2 dataset | Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/data/development/nca_validation.npy, row 0; transfer_v2/nca.py:encode_frames |
| 48 | Structured and shuffled data at fixed markers | Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/data/development/nca_validation.npy; Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/data/development/shuffled_validation.npy; transfer_v2/nca.py:shuffled |
| 49 | Actual full-trajectory token histogram | Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/data/development/nca_validation.npy; Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/data/development/shuffled_validation.npy |
| 50 | Actual transformer attention and feed-forward code | transfer_v2/model.py:42-48; SHA-256 4394c7c226ec72113925da5fb9c50949361d652c5d47f54926e399a5b8f8c6b7 |
| 51 | Actual optimizers for the two stages | transfer_v2/training.py:76-79; SHA-256 fa34eb401bacf01490191bbcfa4729db95e4db2fed95f0de68a15337a36abfe5 |
| 52 | Actual gradient accumulation in the runner | transfer_v2/training.py:86-98; SHA-256 fa34eb401bacf01490191bbcfa4729db95e4db2fed95f0de68a15337a36abfe5 |
| 53 | Historical notebook: preserved pilot execution | kaggle_gnn_nca_pilot.ipynb, cell 25; notebook SHA-256 e47cc0c3e505354f64c4d21aaf2c3d3d114ea3ea52d8d064dab6e491da9f458d |
| 54 | V2 notebook: actual development execution | kaggle_nca_transfer_v2.ipynb, cell 29; notebook SHA-256 80c23b8cdceeb9dd607e92779e095357bb2c088daeb3c196dba1402de773199b |
| 55 | V2 notebook: actual result packaging | kaggle_nca_transfer_v2.ipynb, cell 33; notebook SHA-256 80c23b8cdceeb9dd607e92779e095357bb2c088daeb3c196dba1402de773199b |
| 56 | V2 training configuration | configs/transfer_v2.json; docs/TRANSFER_V2_PROTOCOL.md |
| 57 | Synthetic prediction passed the declared gate | docs/results/transfer_v2_run1/gpu/ppt_diagnostics.png; docs/results/transfer_v2_run1/gpu/grid_nca_17/ppt_summary.json |
| 58 | Gate definition and interpretation | transfer_v2/training.py:quality_gate; docs/TRANSFER_V2_PROTOCOL.md; docs/results/transfer_v2_run1/REVIEW.md |
| 59 | V2 language-stage result at paired reference loss | docs/results/transfer_v2_run1/aggregate.json; docs/results/transfer_v2_run1/paired_language_gain.png |
| 60 | Full curves and the shuffled comparison | docs/results/transfer_v2_run1/gpu/learning_curves.png; docs/results/transfer_v2_run1/comparisons.csv |
| 61 | Per-seed language updates to the scratch endpoint | docs/results/transfer_v2_run1/comparisons.csv |
| 62 | Speedup calculation and final perplexity | docs/results/transfer_v2_run1/comparisons.csv; docs/results/transfer_v2_run1/aggregate.json |
| 63 | Paired final losses and their uncertainty | docs/results/transfer_v2_run1/comparisons.csv; docs/results/transfer_v2_run1/REVIEW.md |
| 64 | Cumulative accounting and synthetic-stage cost | docs/results/transfer_v2_run1/aggregate.json; docs/results/transfer_v2_run1/comparisons.csv |
| 65 | Timing scope and input-token accounting | docs/results/transfer_v2_run1/REVIEW.md; docs/results/transfer_v2_run1/comparisons.csv |
| 66 | Independent replay, provenance and limitations | docs/results/transfer_v2_run1/cpu_verification.json; docs/results/transfer_v2_run1/stage_control_audit.json |
| 67 | Analytical survey of alternative explanations | docs/results/transfer_v2_run1/REVIEW.md; docs/theory/sections/03_information.tex; METHODOLOGY_AUDIT.md |
| 68 | V3: the next graph-versus-grid experiment | docs/TRANSFER_V3_PROTOCOL.md; transfer_v3/graph.py; VALIDATION.md |
| 69 | V3 code: common random draws for graph and grid | transfer_v3/graph.py:65-73; SHA-256 a0787e8a15f5ff058107260c5f762615cdc832848c2bfd484c927f7860022ee2 |
| 70 | V3 code: a degree-preserving donor swap | transfer_v3/graph.py:49-56; SHA-256 a0787e8a15f5ff058107260c5f762615cdc832848c2bfd484c927f7860022ee2 |
| 71 | V3 notebook: actual five-arm execution cell | kaggle_nca_transfer_v3.ipynb, cell 30; notebook SHA-256 3b32f4030cec46df0970b11c58e16bac0af86e8c9b14872365f3ae1a242adfb7 |
| 72 | V3 design and confirmation requirements | docs/TRANSFER_V3_PROTOCOL.md; VALIDATION.md; tests/test_transfer_v3.py |
| 73 | Comparison with the external evidence | https://arxiv.org/html/2603.10055v1; docs/results/transfer_v2_run1/REVIEW.md; docs/results/pilot1/REVIEW.md |
| 74 | Applications at larger scale | docs/theory/sections/06_numerical.tex; docs/results/transfer_v2_run1/REVIEW.md; docs/TRANSFER_V3_PROTOCOL.md |
| 75 | Research decisions supported by the mathematics | docs/theory/sections/07_conclusions.tex; docs/TRANSFER_V3_PROTOCOL.md |
| 76 | Conclusion: observed transfer and open mechanisms | RESULTS.md; docs/results/transfer_v2_run1/REVIEW.md; proofs/lean/PROOF_MAP.md |
| 77 | Documentation, proofs and publication workflow | AGENTS.md; publication.json; CONTRIBUTING.md; LOGBOOK.md; CHANGELOG.md |
| 78 | Source and code navigation | experiment.py; transfer_v2; transfer_v3; build_notebook.py; build_transfer_notebook.py; build_graph_notebook.py |
| 79 | Mathematical documentation coverage | docs/theory/main.tex; proofs/lean/PROOF_MAP.md |
| 80 | Proof module and axiom inventory | proofs/lean/GraphNCA.lean; proofs/lean/verification/axiom-reports.txt |
| 81 | Evidence hierarchy and retained failures | LOGBOOK.md; docs/results/transfer_v2_run1/REVIEW.md; proofs/lean/verification/README.md |
| 82 | Primary literature and local evidence | https://arxiv.org/html/2603.10055v1; https://distill.pub/2020/growing-ca/; https://arxiv.org/abs/1704.01212; https://arxiv.org/abs/1706.03762; https://arxiv.org/abs/2302.13971 |
| 83 | Appendix: posterior prediction and rule inference | docs/theory/sections/03_information.tex |
| 84 | Appendix: comparing graph and grid remainders | docs/theory/sections/05_transfer.tex: eq:graphvsgrid; proofs/lean/GraphNCA/Transfer.lean |
| 85 | Appendix: historical verified CPU learning curves | runs/cpu_verified/curves.png |
| 86 | Appendix: original pilot GPU learning curves | Kaggle Outputs/gnn_nca_results_1_review/artifacts/runs/pilot_20261016_214947_390021/curves.png |
| 87 | Appendix: earlier local CPU diagnostic | runs/cpu_check/curves.svg |
| 88 | Appendix: local smoke execution diagnostic | runs/smoke/curves.svg |
| 89 | Appendix: returned pilot smoke diagnostic | Kaggle Outputs/gnn_nca_results_1_review/artifacts/runs/smoke_20261016_214947_390021/curves.svg |
| 90 | Appendix: returned v2 language execution fixture | Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/runs/smoke/learning_curves.png |
| 91 | Appendix: returned v2 synthetic execution fixture | Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/runs/smoke/ppt_diagnostics.png |
| 92 | Appendix: v3 local five-arm execution fixture | tmp/transfer_v3_check_ed51a18898ec4460bc1a79ac2ac40da4/run_complete/learning_curves.png |
