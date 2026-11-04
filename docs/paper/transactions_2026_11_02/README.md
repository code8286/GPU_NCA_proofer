# Updated Transactions template paper

Soumyajit Deb, Vellore Institute of Technology (VIT).
Prepared 2 November 2026; completed evidence through 1 November 2026.

The [Word article](Graph_NCA_Updated_Transactions_2026_11_02.docx) follows the
supplied IEEE Transactions template: Letter pages, full-width title, two-column
research body, single-paragraph abstract, index terms, numbered equations,
Roman-numbered tables, figure captions and numbered references. The full-width
appendix uses the template's allowance for page-wide tables and figures at the
end. It renders to 13 pages, with 15 equations, eight tables and seven figures.

The new edition covers all completed stages through 19 and the separately
declared scientific-transfer numerical milestone. It retains failed graph
primaries, objective-specific confirmation, censored targets, unresolved seen
fitting and distinctions between imported GPU evidence, CPU studies, retained
checkpoint checks and numerical reference checks. It preserves the
[October stage 9 edition](../README.md) and all underlying evidence.

## Editable sources

- [manuscript.json](manuscript.json) is the canonical content.
- [MANUSCRIPT.md](MANUSCRIPT.md) is the readable companion.
- [main.tex](main.tex) is the mathematical LaTeX companion; it compiled
  successfully with the desktop compiler on 2 November 2026. Its pagination is
  separate from the Word template edition.
- [build_sources.py](build_sources.py) regenerates LaTeX and Markdown.
- [build_docx.py](build_docx.py) rebuilds the Word article from your retained
  template, supplied with `--template`. The original template is not redistributed.
- [build_figures.py](build_figures.py) renders two updated plots from retained
  public results; the other five figures are exact copies of retained assets.
- [build_equation_assets.py](build_equation_assets.py) generates the equation
  compatibility artwork from the mathematical expressions.
- [SOURCE_MAP.json](SOURCE_MAP.json), [VALIDATION.md](VALIDATION.md) and
  [verification.json](verification.json) record lineage, scope and checks.

Run `python docs/paper/transactions_2026_11_02/build_sources.py` from the research
root after content edits. Word rebuilding additionally needs the unchanged
template, python-docx, lxml and Pillow. Figure/equation rendering needs Matplotlib.
The project-local figures and math assets allow the document sources to travel
with the public bundle without weights, corpora or private caches.

The DOCX retains native Office Math expressions in a compatibility branch and
600 dpi equation artwork for editors that do not support that branch. The local
page review verifies the artwork path; it does not claim a Microsoft Word native
equation-rendering test. LaTeX and JSON retain the mathematical editing source.
Word is configured to refresh fields when opened. The QA PDF and page images
remain local intermediates, not additional paper editions.

This article revision adds no scientific optimizer updates, neural evaluations,
new Lean execution, reserved-test access, remote training or external publication.
Existing [Lean proof sources](../../../proofs/lean/README.md) and
[longer theory documentation](../../theory/README.md) retain their stated scope.
