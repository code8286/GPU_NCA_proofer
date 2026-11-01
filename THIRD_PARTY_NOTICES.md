# Third-party material

The unchanged snapshot in `sources/nca-pre-pretraining-reference/` comes from
[danihyunlee/nca-pre-pretraining](https://github.com/danihyunlee/nca-pre-pretraining),
revision `bdd1c71e04eb2e5a2c91481faf278e95846b4431` (12 March 2026).
Copyright (c) 2026 Dan Lee. It remains under the included
[MIT license](sources/nca-pre-pretraining-reference/bdd1c71e04eb2e5a2c91481faf278e95846b4431/LICENSE).
`reference.json` records URLs/hashes. It is not imported by the pilot; its extra
dependencies are not required for this project's checks.

The methodology audit discusses
[Training Language Models via Neural Cellular Automata, v1](https://arxiv.org/html/2603.10055v1).
The paper and competition rules PDF are not redistributed here.

The default corpus is fetched from the
[Tiny Shakespeare source](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
at runtime. A data manifest records its SHA-256 and byte splits. Review the source's
terms before redistributing derived data bundles. Raw corpus bytes, cached arrays
and model checkpoints are excluded from this source snapshot.

NumPy, PyTorch and Matplotlib are installed separately under their own licenses.
Version 2 additionally uses separately installed tiktoken and PyArrow. Its NCA
port is based on the retained MIT reference, with the explicit changes recorded
in `docs/TRANSFER_V2_PROTOCOL.md`; the standalone notebook includes that MIT
license. WikiText-2 is downloaded at runtime from the pinned Salesforce dataset
revision and retains its CC-BY-SA-3.0/GFDL terms. Dataset/checkpoint artifacts
remain excluded from the public source selection.
`docs/results/` contains project measurements and plots; its provenance manifest
distinguishes copied originals from adapted summaries.


The CLRS-Text methodological reference is retained at
[sources/clrs-text-reference](sources/clrs-text-reference/reference.json), commit
`d1c2ad7af8437c7536cf329d9cef8fdf93184d9d` (release v2.0.1), under the included
[Apache License 2.0](sources/clrs-text-reference/d1c2ad7af8437c7536cf329d9cef8fdf93184d9d/LICENSE).
Copyright DeepMind Technologies Limited; original file headers remain intact.
These seven files are reference material and are not imported by the local
`ood_v12` diagnostic. Its task/output adaptation is not a reproduction or
replacement of official CLRS-Text results. Public paper:
[CLRS-Text, arXiv:2406.04229v1](https://arxiv.org/html/2406.04229v1).
