# Stage18 usage

This user-authorized exploratory development experiment measures ID task-learning
curves AFTER the common historical language stage. It is not a new language/PPT
run, fresh-cohort confirmation or permission to run OOD. Read the
[prospective protocol](ACCEL_V18_PROTOCOL.md); earlier calibration failures remain.

Authoring sequence (requires private original checkpoints and offline GPT-2 cache):

```powershell
python scripts/run_logged.py --label accel-v18-freeze-before-first-update -- python -m accel_v18.run freeze
python scripts/run_logged.py --label accel-v18-five-matched-id-fits -- python -m accel_v18.run fit
python scripts/run_logged.py --label accel-v18-separate-checkpoint-replay -- python -m accel_v18.run verify
python scripts/run_logged.py --label accel-v18-export-verified-evidence -- python scripts/report_accel_v18.py
```

Freeze binds all arms, new ID sampling/order, baseline/targets, source states and
original cost evidence. Fit refuses to restart its first attempt. All arms finish
600 updates and seven full held-out evaluations; no early stopping. Unreached is
censored, not attainment at the budget limit, and undefined ratios remain null.
Step0 success has no infinite adaptation speedup. Original GPU cost components
and new CPU costs are separate; their summed account is not a controlled
hardware-throughput comparison. Shared paired generation is charged in full.

After export, from either root or a standalone publication:

```powershell
python scripts/run_logged.py --label accel-v18-public-reproduction -- python scripts/reproduce_accel_v18.py
```

This regenerates ID-only data/order, baseline, cost extraction and all4480
generated scores/crossings without weights, tokenizer cache, private data or
network. It reaggregates imported teacher counters; neural logits and optimizer
states require private replay. No OOD rows are generated (assessment count0).
The exact all-case results JSON has a bounded12MB publication allowance; private
weights/optimizer states/raw arrays remain excluded. The append-only log retains
its separate10MB bound. No other artifact limits are relaxed.

Regenerate notebooks and directly sync/check Prod after all logged checks in
the research root. In a standalone publication refresh its inventory after edits.
No remote/paid training, email, push or external publication is performed.
