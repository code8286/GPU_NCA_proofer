# Historical result evidence

`pilot1/` contains the existing audited T4 pilot summaries, per-arm metrics,
checkpoint verification records and retrospective diagnosis. `cpu_check/`
contains the existing short CPU diagnostic summaries. `PROVENANCE.json` records
original and publication hashes; only the review links/availability wording was
adapted. These are historical imports, not newly run GPU measurements.

The complete original archive is not in this repository. Its SHA-256 is
`14f02cd87e924c18f6658333a0fde11b3e06b968504954b0320d33ff9a516ee6`.
Checkpoints, raw corpora and cached token arrays remain local. Ask maintainers
through a repository issue about full artifact availability; no public download
location has been established. Compact logs permit metric recalculation but
cannot reproduce held-out losses without the original data and weights.

Read `../../METHODOLOGY_AUDIT.md` relative to this directory for scientific
limitations. New publication check transcripts are in `../../logs/checks.jsonl`.
