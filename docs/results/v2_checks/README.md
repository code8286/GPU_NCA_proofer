# Version 2 local checks

Date: 17 October 2026. These checks validate software controls and input handling;
they are not a new GPU benchmark or an observed pre-pre-training benefit.

Environment: Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.5.2, Matplotlib 3.11.2,
tiktoken 0.14.0, PyArrow 26.0.0, Windows CPU. Additional direct versions are in
`requirements-v2-tested.txt`.

The recorded `v2-release-scientific-suite` check passed 33 tests, including 16
new tests. `v2-release-integration` exercised complete, gate-stopped and explicit
confirmation-fixture workflows, with selected checkpoints re-evaluated exactly.
These fixtures bypass the scientific gate only when explicitly labeled execution
checks; the blocked case enforced a deliberately impossible prerequisite.

The real input preparation check `v2-real-wikitext-tokenizer` downloaded only the
pinned official WikiText-2 train/validation splits, produced 2,403,644 and 248,461
GPT-2 BPE tokens, fingerprinted the tokenizer and left confirmation data untouched.
The default architecture/vocabulary also passed a short forward/backward check.

An initial offline check failed because this managed Windows session could not
write nested files in its OS temporary directory. Its transcript was retained;
the check now creates a unique ignored workspace directory and preserves its
artifacts. A documentation/source patch also initially failed to match an
embedded-string line; it was reapplied with the correct context before testing.

See the append-only `logs/checks.jsonl` for complete outputs and source hashes.
No smoke-fixture time or loss is promoted to scientific evidence. The next
Kaggle notebook must measure the real synthetic gate and language learning curves.
