# Using the OOD evaluation layer

Read the [frozen protocol](OOD_V12_PROTOCOL.md) for task meanings, exclusions,
controls and interpretation. This diagnostic complements the NCA readiness and
categorical bridge. It supplies no language or algorithmic training command.

## Generate the local development task suite

From the research root, using the existing Python environment:

```powershell
python scripts/run_logged.py --label ood-local-dataset -- python scripts/generate_ood_v12_dataset.py --out data/ood_development
```

This independent generator requires no old model weights, Gmail, network or
official CLRS runtime dependencies. The output has training, ID-development and
assessment JSONL files and a source/input manifest. It is a **local adaptation**:
BFS/Dijkstra full distance vectors instead of official predecessor outputs.
Assessment has ID, size, family and joint shifts plus all four control views.
Raw generated datasets remain private under the publication policy.

Provide the predictor only `ood_v12.data.public_input(example)`, containing
`id` and `prompt`. Do not pass truth, references, graph edges metadata or any
assessment-derived hints to it. Input text already includes the necessary graph.
Write exactly one JSONL prediction per example:

```json
{"id":"the exact dataset ID","text":"[0,1,-1,2]","status":"ok"}
```

Values above illustrate the schema, not a solution to an actual example. Preserve
the exact input ID; the scorer rejects missing, duplicate and unexpected IDs.
`status: "context_overflow"` identifies an unsupported prompt rather than silently
truncating it. Record this as coverage failure; such comparisons are ineligible.

```powershell
python scripts/run_logged.py --label ood-score -- python -m ood_v12.run score --dataset data/ood_development/assessment.jsonl --predictions runs/my_predictions.jsonl --report runs/my_ood_scores.json
```

The report retains per-example correctness, cell-level validity/exact match/node
accuracy, coverage, native+intervention correctness and correctness on both
invariant views. A correct format or consistent wrong answer is not success.
Do not select checkpoints/hyperparameters from OOD assessment; only ID development
may drive selection. A later confirmation study needs separately frozen fresh
data/cohorts and targets. The generated development set is not that confirmation.

## Score official CLRS-Text exports separately

Use the official pinned generator or official exports with their own dependency
environment and output definitions. No upstream trace converter is copied into
the local task generator. Preserve `prompt`, `references` and `auxiliary` fields.
If an export lacks IDs, the adapter derives each ID as SHA-256 of
`json.dumps(record, sort_keys=True).encode()`; use that same ID in predictions.

```powershell
python scripts/run_logged.py --label clrs-official-score -- python -m ood_v12.run score-official --dataset data/official_clrs.jsonl --predictions runs/official_predictions.jsonl --report runs/official_scores.json
```

This adapter compares the generated final string against official references,
ignoring outer whitespace only. It preserves official semantics; it does not
turn predecessor vectors into distance vectors. Full official benchmark runs,
generation settings, trace choices and reported averages need their own protocol.

## Reproduce the retained historical calibration

The private original v4 model folders and pinned source snapshot must exist at
their retained locations. They are deliberately excluded from a public clone.
The GPT-2 tokenizer uses a private workspace cache. Its first download requires
network access; evaluation thereafter can run offline. Use a NEW run folder to
recompute the development calibration; this recreates the same deterministic
cases and is a replay, not an independent confirmation cohort.

```powershell
python scripts/run_logged.py --label ood-freeze -- python -m ood_v12.run freeze --out runs/ood_v12_replay
python scripts/run_logged.py --label ood-controls -- python -m ood_v12.run controls --out runs/ood_v12_replay
python scripts/run_logged.py --label ood-legacy -- python -m ood_v12.run legacy --out runs/ood_v12_replay
python scripts/run_logged.py --label ood-replay -- python -m ood_v12.run verify --out runs/ood_v12_replay
```

These commands perform no fitting. `greedy()` uses the retained tied-head LM
architecture and original tokenizer, checks complete context before decoding,
does not constrain logits and retains generated tokens/stop reasons. Replay
reopens the same checkpoints with the same code; it is not independent neural
loss code, repeated training or a new GPU benchmark.

For a learning-speed claim, first freeze eligible NCA initializations and fresh
paired cohorts. Match the full language stage AND any ID task-adaptation stage,
including common heads, actual data order, schedules and budgets. The helpers
`require_paired_transfer()` and `accuracy_crossing()` check declared record
consistency and censoring; they do not by themselves authenticate experimental
receipts or establish statistical significance. Report each OOD cell and total
cost alongside predeclared ID-development target crossing. Current historical
random-rule readiness failures remain failures.
