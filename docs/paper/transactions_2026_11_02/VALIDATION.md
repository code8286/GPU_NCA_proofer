# Validation of the updated paper

2 November 2026. The scientific evidence cutoff is 1 November 2026.

The final Word article renders to 13 Letter pages. Every page was rendered and
visually inspected, including all 15 equations, eight tables and seven figures.
The final content check retains every manuscript prose paragraph in OOXML,
checks 210 rendered prose windows, numbering, author/affiliation, references,
page bounds and nonempty page images. No scientific model or solver was rerun.

The exact author-supplied template is preserved unchanged. Twenty preserve-only
package parts, including styles, numbering, fonts and theme, remain byte-identical.
The original title/body section geometry is preserved. A third full-width section
implements the template's allowance for end-of-article wide figures and tables.
Header placeholder and sample authors, funding, biographies and visible examples
are replaced. The author affiliation and correspondence are the details supplied
by the user. Fields are configured to refresh on opening in Word; no native Word
field-refresh or native equation-rendering test is claimed.

The bundled runtime lacks a Windows office renderer and Word COM is unavailable.
The first packaged render fails for missing soffice. A temporary official
LibreOffice package was downloaded and extracted within ignored task space,
without installation or changing desktop applications. Administrative extraction
was unavailable; archive extraction plus MSI file/directory mapping supplied the
task-local executable. Rendering uses the packaged `render_docx.py` with that
explicit task-local path, never a user's installed desktop LibreOffice.

Initial visual failures were repaired: a title-only page from an incorrectly
placed section-start rule, duplicated running text, and unsupported Office Math
preview. Minimal equation probes and package-font checks confirmed the preview
limitation. The DOCX retains native mathematical expressions in a compatibility
choice, with 600 dpi rendered artwork as fallback. All fallback equations have
nonwhite content strictly inside their image bounds. An early artwork layout
clipped deep subscripts; exact math-image bounding boxes repaired it. The final
local preview verifies the fallback path, while LaTeX/JSON provide the complete
mathematical editing source. A portable Word rebuild reproduces every reviewed
package part exactly; ZIP timestamps are not used as scientific identity.

The final `main.tex` compiled successfully using the desktop LaTeX compiler.
It is a companion mathematical representation of the same content, not the
rendering engine for the delivered Word article. No new compilation of the older
theory project, Lean execution or remote CI run follows from this success.

Arithmetic/content checks reproduce the exact quadratic first hits 39 and 25,
normalized weighting masses and position-wise loss decomposition. Thirty-two
finite validity/correctness cases check exact-match <= corresponding node score.
The stage 11 table and objective decisions match retained full-precision JSON;
all stage 18 crossings remain censored and contrast ratios null; every stage 19
seen/development exact score remains zero. These are accounting/arithmetic checks
of retained evidence, not new neural inference or an extension of the Lean scope.

All authoring/check failures remain in the append-only research log. The initial
manuscript builder had a quoted-prime syntax error, repaired without scientific
changes. A diagnostic console required UTF-8. No earlier result or target,
original reference, October paper, checkpoint or returned archive is overwritten.
The publication project includes the Word artifact, LaTeX/Markdown/JSON sources,
figures, equation assets and verification/source records. QA PDFs/PNGs, the
temporary renderer and downloaded package remain local ignored intermediates.
