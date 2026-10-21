# Detailed research presentation

[Open the current 128-slide PowerPoint](GPU_NCA_Research_Detailed_v8.pptx), updated 20 October 2026 through research stage 8. The [earlier 92-slide version](GPU_NCA_Research_Detailed_v2.pptx) and [original descriptive deck](../GPU_NCA_Proofer_Presentation.pptx) remain preserved.

The main narrative runs through slide 112. Slides 113–128 provide documentation, proof coverage, further derivations and historical diagnostic plots. The new measured-evidence chapter is slides 73–108. Speaker notes and the complete index below identify the sources.

## Coverage

| Slides | Subject |
| --- | --- |
| 1–8 | Problem, hypotheses, literature and current status, with the original v2 transfer result |
| 9–25 | NCA/GNN dynamics, actual topology, information bounds, tokenizer and dataset patch |
| 26–40 | Transfer derivations, quadratic examples, exact first hits and selected Lean proofs |
| 41–67 | Historical methodology, actual v2 code/notebooks/data views, results and cumulative cost |
| 68–72 | Completed v3 design, paired topology code and actual notebook execution |
| 73–78 | Research progression, definitions and reviewed v3 GPU outcomes/plots |
| 79–89 | V4 controls, actual code/fourth notebook, primary comparisons, cost and causal frequency |
| 90–100 | CPU readiness, complement trainability, spatial bridge, involution derivation and neighbor intervention |
| 101–108 | Weighted objective, actual code, failed stage 8 result, controls and proposed next diagnostic |
| 109–112 | External comparison, larger-scale applications, mathematical implications and current conclusion |
| 113–128 | Documentation, proof coverage, source navigation, further mathematics and retained historical plots |

Four native charts retain their actual workbooks and formulas unchanged. Forty native tables include the decoded NCA state matrices. Code excerpts and equations remain editable text. Ten additional recorded figures cover v3/v4 and stages 5–8. Existing dataset views, GNN visualization, historical plots, all four complete notebooks and mathematical sources remain available.

## Interpretation

V2–v4 show language-stage transfer relative to scratch, with comparable shuffled gains and unrecovered synthetic cost. V3 has a censored primary graph/grid crossing; both frozen v4 graph-advantage tests fail. These runs do not demonstrate a graph-specific acceleration mechanism.

Stage 5 rejects the original random-rule checkpoints under stronger frequency/ordering checks. Controlled complement learning succeeds in stage 6. Stage 7 shows partial first spatial prediction despite high trajectory accuracy. Stage 8’s fixed sixfold first-transition weighting worsens original and intervened NLL in every coherent seed. These are local CPU development diagnostics. Language training remains disabled.

The [LaTeX note](../theory/README.md) and [Lean coverage map](../../proofs/lean/PROOF_MAP.md) retain their prior verification scope. Forty-four selected statements have saved Lean checker evidence. Taylor calculus, entropy and spectral arguments remain informal. Exact toy convergence and the displayed new accounting/algebra are separate from measured model performance.

## Traceability

[Current source coverage](SOURCE_COVERAGE_v8.json) records the deck hash, slide mapping, reviewed evidence hashes, scientific sources, four notebooks and unchanged chart/workbook parts. The [earlier source snapshot](SOURCE_COVERAGE.json) supplies the retained v2 dataset/figure identities. These are authoring-time snapshots; the publication manifest tracks current distribution hashes. Binary datasets and weights remain private and preserved.

## Complete slide index

| Slide | Topic | Primary source references |
| --- | --- | --- |
| 1 | GNNs in Neural Cellular Automata | README.md; RESULTS.md |
| 2 | Research map | RESULTS.md; docs/theory/main.tex |
| 3 | Problem statement and research question | RESEARCH_CONTEXT.md; METHODOLOGY_AUDIT.md |
| 4 | Research status through stage 8 | RESULTS.md |
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
| 22 | Actual 2 Ã— 2 patch-token encoding | transfer_v2/nca.py:76-81; SHA-256 4b7bbf83f70614d2feca5619a68f4313a18172900d0fb3397d14d96f0d0bc52f |
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
| 68 | Graph study and later diagnostic stages | RESULTS.md; RESEARCH_CONTEXT.md |
| 69 | V3 code: common random draws for graph and grid | transfer_v3/graph.py:65-73; SHA-256 a0787e8a15f5ff058107260c5f762615cdc832848c2bfd484c927f7860022ee2 |
| 70 | V3 code: a degree-preserving donor swap | transfer_v3/graph.py:49-56; SHA-256 a0787e8a15f5ff058107260c5f762615cdc832848c2bfd484c927f7860022ee2 |
| 71 | V3 notebook: actual five-arm execution cell | kaggle_nca_transfer_v3.ipynb, cell 30; notebook SHA-256 3b32f4030cec46df0970b11c58e16bac0af86e8c9b14872365f3ae1a242adfb7 |
| 72 | V3 design and completed audit | docs/TRANSFER_V3_PROTOCOL.md; docs/results/transfer_v3_run1/REVIEW.md |
| 73 | Research progression and unchanged comparisons | RESULTS.md; RESEARCH_CONTEXT.md |
| 74 | Definitions for the experimental comparisons | docs/TRANSFER_V4_PROTOCOL.md; docs/READINESS_V5_PROTOCOL.md |
| 75 | V3 primary comparison remains partly censored | docs/results/transfer_v3_run1/REVIEW.md |
| 76 | V3 paired graph effects | docs/results/transfer_v3_run1/paired_graph_transfer.png; docs/results/transfer_v3_run1/REVIEW.md |
| 77 | V3 recorded language curves | docs/results/transfer_v3_run1/gpu/learning_curves.png; docs/results/transfer_v3_run1/REVIEW.md |
| 78 | V3 synthetic learning diagnostics | docs/results/transfer_v3_run1/gpu/ppt_diagnostics.png; docs/results/transfer_v3_run1/REVIEW.md |
| 79 | V4 strengthens the supervised-prefix control | docs/TRANSFER_V4_PROTOCOL.md; transfer_v4/controls.py |
| 80 | Actual v4 prefix-control code | transfer_v4/controls.py:15-29 |
| 81 | V4 notebook execution cell | kaggle_nca_transfer_v4.ipynb:cell 34; build_mechanism_notebook.py |
| 82 | V4 crossings at the common scratch target | docs/results/transfer_v4_run1/primary_analysis.json |
| 83 | V4 frozen graph-advantage tests both fail | docs/results/transfer_v4_run1/primary_analysis.json; docs/TRANSFER_V4_PROTOCOL.md |
| 84 | V4 language gain and cumulative cost | docs/results/transfer_v4_run1/REVIEW.md; docs/results/transfer_v4_run1/aggregate.json |
| 85 | V4 paired graph comparisons | docs/results/transfer_v4_run1/paired_graph_transfer.png; docs/results/transfer_v4_run1/REVIEW.md |
| 86 | V4 recorded language curves | docs/results/transfer_v4_run1/gpu/learning_curves.png; docs/results/transfer_v4_run1/REVIEW.md |
| 87 | V4 original synthetic diagnostics | docs/results/transfer_v4_run1/gpu/ppt_diagnostics.png; docs/results/transfer_v4_run1/REVIEW.md |
| 88 | Causal frequency and dynamics interpretation | readiness_v5/frequency.py; docs/results/transfer_v4_run1/frequency_diagnostic.json |
| 89 | Actual causal frequency likelihood | readiness_v5/frequency.py:20-29 |
| 90 | Stage 5 readiness rejects the original checkpoints | docs/results/readiness_v5_run1/REVIEW.md; docs/results/readiness_v5_run1/summary.json; readiness_v5/gate.py |
| 91 | Stage 5 frequency and ordering evidence | docs/results/readiness_v5_run1/readiness_advantages.png; docs/results/readiness_v5_run1/REVIEW.md |
| 92 | Stage 6 tests basic transformer trainability | docs/TRAINABILITY_V6_PROTOCOL.md; docs/results/trainability_v6_run1/REVIEW.md |
| 93 | Stage 6 learns the complement rule | docs/results/trainability_v6_run1/REVIEW.md; docs/results/trainability_v6_run1/summary.json |
| 94 | Stage 6 fixed-budget learning curves | docs/results/trainability_v6_run1/learning_curves.png; docs/results/trainability_v6_run1/REVIEW.md |
| 95 | Stage 7 adds a matched spatial interaction | interaction_v7/data.py:10-15 |
| 96 | Exact period-two structure of the spatial rule | interaction_v7/data.py; tests/test_interaction_v7.py; docs/results/interaction_v7_run1/REVIEW.md |
| 97 | Stage 7 full-trajectory results | docs/results/interaction_v7_run1/REVIEW.md; docs/results/interaction_v7_run1/summary.json |
| 98 | Stage 7 matched learning curves | docs/results/interaction_v7_run1/learning_curves.png; docs/results/interaction_v7_run1/REVIEW.md |
| 99 | First transition and neighbor intervention | docs/INTERACTION_V7_FIRST_TRANSITION.md; docs/results/interaction_v7_run1/first_transition/FROZEN_PROTOCOL.md |
| 100 | Stage 7 first spatial predictions remain partial | docs/results/interaction_v7_run1/REVIEW.md; docs/results/interaction_v7_run1/first_transition/results.json |
| 101 | Stage 8 objective and exact loss-mass accounting | docs/LOSS_WEIGHT_V8_PROTOCOL.md; loss_weight_v8/objective.py |
| 102 | Actual weighted-loss implementation | loss_weight_v8/objective.py:18-25 |
| 103 | Stage 8 fails every coherent seed | docs/results/loss_weight_v8_run1/summary.json; docs/LOSS_WEIGHT_V8_PROTOCOL.md |
| 104 | Stage 8 coherent-model outcomes | docs/results/loss_weight_v8_run1/summary.json |
| 105 | Stage 8 original and intervened correctness | docs/results/loss_weight_v8_run1/first_transition_accuracy.png; docs/results/loss_weight_v8_run1/REVIEW.md |
| 106 | Stage 8 controls, replay and cost scope | docs/results/loss_weight_v8_run1/REVIEW.md; docs/results/loss_weight_v8_run1/verification.json; docs/results/loss_weight_v8_run1/PROVENANCE.json |
| 107 | Analytical survey of what the evidence supports | RESULTS.md; docs/results/loss_weight_v8_run1/REVIEW.md; proofs/lean/PROOF_MAP.md |
| 108 | Next declared diagnostic and research limits | docs/results/loss_weight_v8_run1/REVIEW.md; RESEARCH_CONTEXT.md |
| 109 | Comparison with the motivating study | sources/nca-pre-pretraining-reference/reference.json; docs/results/transfer_v4_run1/REVIEW.md; docs/presentation/SOURCE_COVERAGE.json |
| 110 | Applications at larger scale | docs/theory/sections/06_numerical.tex; docs/results/transfer_v2_run1/REVIEW.md; docs/TRANSFER_V3_PROTOCOL.md |
| 111 | Research decisions supported by the mathematics | docs/theory/sections/07_conclusions.tex; docs/TRANSFER_V3_PROTOCOL.md |
| 112 | Conclusion through stage 8 | RESULTS.md; docs/results/loss_weight_v8_run1/REVIEW.md; proofs/lean/PROOF_MAP.md |
| 113 | Documentation, proofs and publication | AGENTS.md; publication.json; proofs/lean/PROOF_MAP.md |
| 114 | Source and code navigation | publication.json; RESULTS.md |
| 115 | Mathematical documentation coverage | docs/theory/main.tex; proofs/lean/PROOF_MAP.md |
| 116 | Proof module and axiom inventory | proofs/lean/GraphNCA.lean; proofs/lean/verification/axiom-reports.txt |
| 117 | Evidence hierarchy and retained failures | LOGBOOK.md; RESULTS.md; proofs/lean/verification/README.md |
| 118 | Primary literature and local evidence | https://arxiv.org/html/2603.10055v1; https://distill.pub/2020/growing-ca/; https://arxiv.org/abs/1704.01212; https://arxiv.org/abs/1706.03762; https://arxiv.org/abs/2302.13971 |
| 119 | Appendix: posterior prediction and rule inference | docs/theory/sections/03_information.tex |
| 120 | Appendix: comparing graph and grid remainders | docs/theory/sections/05_transfer.tex: eq:graphvsgrid; proofs/lean/GraphNCA/Transfer.lean |
| 121 | Appendix: historical verified CPU learning curves | runs/cpu_verified/curves.png |
| 122 | Appendix: original pilot GPU learning curves | Kaggle Outputs/gnn_nca_results_1_review/artifacts/runs/pilot_20261016_214947_390021/curves.png |
| 123 | Appendix: earlier local CPU diagnostic | runs/cpu_check/curves.svg |
| 124 | Appendix: local smoke execution diagnostic | runs/smoke/curves.svg |
| 125 | Appendix: returned pilot smoke diagnostic | Kaggle Outputs/gnn_nca_results_1_review/artifacts/runs/smoke_20261016_214947_390021/curves.svg |
| 126 | Appendix: returned v2 language execution fixture | Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/runs/smoke/learning_curves.png |
| 127 | Appendix: returned v2 synthetic execution fixture | Kaggle Outputs/nca_transfer_v2_run1_review/artifacts/runs/smoke/ppt_diagnostics.png |
| 128 | Appendix: v3 local five-arm execution fixture | tmp/transfer_v3_check_ed51a18898ec4460bc1a79ac2ac40da4/run_complete/learning_curves.png |
