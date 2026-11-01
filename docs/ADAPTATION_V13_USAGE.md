# Stage 13 ID-adaptation calibration

Read [the frozen protocol](ADAPTATION_V13_PROTOCOL.md) before running.
This is a CPU historical calibration, not a new PPT/language run. The five
private v4 language checkpoint folders and stage 12 private run must be present.
Weights, optimizer states, graph data and caches are deliberately not shipped.
A standalone publication alone cannot recreate the historical starting models.

From the research root, use the same interpreter/environment as stage 12:

```powershell
python scripts/run_logged.py --label NEW-freeze -- python -m adaptation_v13.run freeze --out runs/NEW_RUN
python scripts/run_logged.py --label NEW-fit -- python -m adaptation_v13.run fit --out runs/NEW_RUN
python scripts/run_logged.py --label NEW-replay -- python -m adaptation_v13.run verify --out runs/NEW_RUN
```

Choose a NEW directory. Freeze refuses existing folders, fit refuses a second
attempt, and source/config/protocol changes invalidate the receipt. These are
not resumable commands; preserve any failed fitting attempt before declaring
recovery. Do not rerun the scientific training just to view completed results.
The fixed original study is `runs/adaptation_v13_run1` and its public evidence is
[the stage 13 review](results/adaptation_v13_run1/REVIEW.md).

The logger captures output until the command ends. While it runs, each arm's
`models/ARM/progress.json` records the last completed fixed evaluation at updates
0,100,300,600. Completion requires root `results.json` and `results_receipt.json`;
independent invocation of replay must additionally produce `verification.json`
with `passed: true`. One arm reaching 600 is not full-study completion.

For the original study only, after replay, export and audit evidence:

```powershell
python scripts/run_logged.py --label adaptation-v13-export-verified-evidence -- python scripts/report_adaptation_v13.py
python scripts/run_logged.py --label adaptation-v13-evidence-decision-audit -- python scripts/check_adaptation_v13_review.py
```

The report exporter refuses an existing publication evidence directory. It is
bound to the original run; do not relabel a replication as the original.
The evidence audit expects the original recorded command chronology.

There is no OOD training or evaluation command in this package. If the frozen
ID criterion fails, OOD is skipped. A pass permits a separate fixed-endpoint
DEVELOPMENT evaluation, still without a confirmatory graph-acceleration claim.
Original random-rule readiness and the separate categorical bridge remain.

The inherited prior-receipt verifier rehashes assessment bytes for integrity;
only ID training/development examples are parsed. See the
[input-integrity wording clarification](ADAPTATION_V13_INTEGRITY_CLARIFICATION.md).

## Public score reproduction without private weights

A standalone publication can independently regenerate the declared data bytes
and rescore all 20 exported prediction files (2560 answers):

```powershell
python scripts/run_logged.py --label public-ID-score-reproduction -- python scripts/reproduce_adaptation_v13_scores.py
```

This uses only public generator/config/source/plan/prediction files, temporary
workspace files that are cleaned on completion, and NumPy/PyTorch imports. It
needs no private run folders, model weights, tokenizer cache, email or network.
It verifies the regenerated train/development SHA-256 identities and every
published task/size score. This reproduces answer scoring, not neural generation,
NLL or fitting. Those require the retained private weights and environment.
