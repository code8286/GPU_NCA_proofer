# Research manuscript

**Synthetic Pre-Pre-Training with Graph Neural Cellular Automata: Transfer,
Control Failures and Spatial Generalization**

Soumyajit Deb, Vellore Institute of Technology (VIT). Research snapshot:
20 October 2026, through completed stage 9.

The manuscript follows the supplied reference's A4, single-column academic
structure: structured abstract, keywords, nomenclature, related work, model,
mathematical derivations, methodology, validation, results, discussion,
applications, limitations, conclusion, numbered references and appendices.
The reference's highlighting, repetitive framing and unrelated subject matter
are not copied. Author name, affiliation and correspondence details were
provided expressly for this manuscript.

## Files and editing

- [main.tex](main.tex): complete editable LaTeX manuscript.
- [MANUSCRIPT.md](MANUSCRIPT.md): readable text, equations, tables and figures.
- [manuscript.json](manuscript.json): canonical content for both representations.
- [build_sources.py](build_sources.py): regenerate the TeX and Markdown sources.
- [figures/](figures/): eight scientific figures from retained data, audited
  records or the explicitly constructed analytic toy model.
- [numerical_checks.json](numerical_checks.json): exact/closed-form arithmetic
  checks, including the new position-wise loss decomposition.
- [SOURCE_MAP.json](SOURCE_MAP.json): authoring-time source hashes and figure lineage.
- [VALIDATION.md](VALIDATION.md): document, arithmetic and compilation scope.

Edit `manuscript.json`, then run `python docs/paper/build_sources.py` from the
repository root. Open this project in the LaTeX editor, or run `pdflatex main.tex`
twice from this directory with a standard TeX installation. The figures are
project-local, so no private dataset or model checkpoint is needed to compile.

The delivered PDF is locally stored at
`output/pdf/Graph_NCA_Transfer_Research_Paper.pdf` in the research workspace.
It has 23 pages, 28 numbered equations, 16 tables and eight figures. It and the
LaTeX version share the canonical manuscript; separate typesetting engines may
paginate differently. The LaTeX project compiled successfully with the desktop
editor on 20 October 2026. The delivered PDF's text, vector equations, table values
and rendered pages were checked independently.

The publication policy selects editable sources and figures, including this
entire folder, while rendered PDFs remain local generated artifacts. Do not place
compiled PDFs or build caches in this selected source directory. A standalone
Prod checkout includes the source project, not the local output directory.

## Scientific scope

Imported GPU studies remain separate from later CPU training and checkpoint-only
diagnostics. The paper retains failed graph comparisons, failed random-rule
readiness and the negative weighting primary. Stage 9 qualifies aggregate
prediction behavior without reversing stage 8's first-patch failure.

The selected [Lean proof core](../../proofs/lean/README.md) retains its prior
44-statement verification scope; the [mathematical note](../theory/README.md)
provides the longer derivations. The manuscript's new displayed decomposition
is a written algebraic inference, not a new Lean theorem or checkpoint evaluation.
No model training, official-test access, remote job or publication was performed
to prepare this paper.
