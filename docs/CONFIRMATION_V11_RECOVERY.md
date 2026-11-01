# Stage 11 operational recovery and concurrent preparation

The first training invocation failed after 21 complete fits and 800 updates of
the next fit. Windows denied atomic replacement of `status.json` after its
800-update metrics were written. No assessment was run and no partial endpoint
checkpoint exists. Preserve the failed transcript, partial metrics/status and
all 21 completed checkpoint/summary identities.

The original numerical package, plan, data, initial weights, update kernels and
scientific criteria remain unchanged. The actual recovery was frozen in `execution_recovery.json` before
executing `scripts/recover_confirmation_v11.py train`. Audit all 21 completed
weights, final optimizer counts, budgets, sampled indices and common starts.
Move the entire unfinished folder into `interrupted_attempt_1` within the same
run, after verifying both resolved paths are inside that run. Preserve its
metrics, status snapshots and failed command record by checksum.

The executed helper adds one completed-folder guard to the original frozen
train function. It verifies and appends the 21 existing records, skipping their
keys; it archives the partial folder before restarting. The original and derived
function hashes and driver checksum are frozen in the recovery receipt. Bind
the same numerical globals, adding that guard and retries for PermissionError
during JSON replacement. This retry
does not repeat an optimizer update or change the loss. Every remaining fit
uses the original unchanged numerical body and begins from its original saved
initialization with fresh Adam and the original RNG stream. No checkpoint is
selected, extended or continued from a partial state.

Restart the interrupted 4096/standard/prefix/211 fit from update zero and run
both remaining 4096/weighted fits for seed 211. Each final endpoint still has
exactly 1000 updates/188000 input tokens/168000 targets. Verify the restarted
fit's recorded loss/development metrics through update 800 match the interrupted
prefix exactly (ignoring wall-clock timers). Then require all original 24
endpoints before the unchanged assessment and replay.

Account for the discarded 800 updates, 150400 input tokens and 134400 targets:
actual fitting totals are 24800 updates, 4662400 input tokens and 4166400 targets,
versus 24000/4512000/4032000 contributing to final checkpoints. The last partial
timer is only a lower bound on interrupted stage time; retain the failed whole-
command duration separately. No controlled throughput or speedup claim follows.
This is an operational recovery before outcome inspection, not another treatment
or a relaxed confirmation criterion. Language remains gated.

A second preparation in Attempt-II overlapped with an already active recovery
in Attempt-I in the shared workspace. Its freeze stopped on the retained-model
count guard before any fitting. Preserve that unused alternative driver/protocol
privately and the failed command in the append-only log. Restore the actual
executed driver byte-for-byte to its frozen checksum, rather than launch a second
recovery. No additional model fitting or outcome selection followed from that
alternative. The actual recovery receipt and verification determine provenance.
