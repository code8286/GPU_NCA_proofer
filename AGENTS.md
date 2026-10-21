# Research and publication maintenance

- Preserve original datasets, checkpoints, returned ZIPs, extracted evidence and
  pinned third-party source. Never silently relabel the first pilot.
- The question is pre-pre-training transfer into an identical language stage.
  `experiment.py` implements the historical matched-total-budget pilot. Read
  `RESEARCH_CONTEXT.md` and `METHODOLOGY_AUDIT.md` before changing it.
- Record every meaningful code, configuration, documentation or result change in
  `LOGBOOK.md`; summarize public changes in `CHANGELOG.md`.
- Run checks through `scripts/run_logged.py` to retain commands, outputs, status,
  duration and source identity in `logs/checks.jsonl`. Retain failures and their
  resolution; do not remove or rewrite earlier entries.
- Distinguish imported historical evidence, new CPU checks and GPU measurements.
  Never report a check as completed just because it is configured in CI.
- In the research workspace, root files are canonical. Regenerate the notebook
  with `python build_notebook.py`, then run `python scripts/sync_prod.py` and
  `python scripts/sync_prod.py --check`. Do not edit `Prod` directly.
- For the independent-stage v2 runner, also regenerate its separate notebook
  with `python build_transfer_notebook.py`; preserve the historical v1 runner.
  Run the final sync/check directly after finishing logged checks: the logger
  appends its entry after a command exits, so logging a sync itself creates drift.
- A checkout with `PUBLICATION_MANIFEST.json` is a standalone publication:
  edit normally, regenerate the notebook and run
  `python scripts/check_release.py --refresh` after documenting/testing changes.
  Do not create a nested `Prod` there.
- Keep secrets, personal paths, corpora, model weights and caches out of public
  files. Explicitly select new distributable files in `publication.json`.
  Preserve upstream MIT notices and reference checksums.
- Every request to update Prod includes all current and newly added Lean proof
  sources, proof documentation and verification records under `proofs/lean/`,
  plus LaTeX sources and supporting mathematical documentation under
  `docs/theory/`, by default. Keep these directories recursively selected in
  `publication.json`; explicitly select any new proof/documentation locations.
  Exclude local build caches and private artifacts under the publication policy.
  Verify inclusion and root/Prod parity; do not imply a new Lean check or LaTeX
  compilation merely because their existing sources and evidence were copied.
- Do not push, publish or launch remote/paid training without user authorization.
