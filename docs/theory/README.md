# Mathematical note

Open [main.tex](main.tex) as a LaTeX project. It develops graph-NCA
pre-pre-training through proofs, assumptions, counterexamples and worked
numerical substitutions. It contains no simulation or training implementation.

The note covers:

- Typed graph updates, exact grid/convolution equivalence, degree-preserving
  rewiring, a solved diffusion spectrum and a logit-margin perturbation bound.
- Discrete rule entropy, contextual rule inference, conditional prediction loss,
  and residual dependence in a frequency-preserving shuffle.
- Exact patch encoding and source-token lags. In the binary pilot the full-window
  parent-lag bound is 33, mean lag is 18 for every typed-permutation topology,
  and grid lag variance is 19.125. Crop-boundary truncation is treated separately.
- Backbone transfer with embedding reset, a sufficient descent condition,
  a topology-dependent linear prediction example, and Hessian-mode convergence
  with a ranking-reversal counterexample.
- First-hit/break-even calculations, exact transformer parameter counting,
  token budgets and the interpretation of loss/perplexity ratios.

General derivations apply to the proposed graph extension; binary numerical
examples use the preserved historical pilot. Categorical/stochastic extensions
are distinguished explicitly. The note does not assert that graph rewiring has
already been added to every experimental runner or that real language transfer
has been proved.

`sections/` contains seven included sections. References are embedded in
`main.tex`, so no bibliography service is needed. With the standard AMS,
geometry, booktabs and hyperref packages installed, run `pdflatex main.tex`
twice from this directory. The Codex LaTeX editor can open the project directly.
Compilation status and failures are recorded in the research logbook.

[numerical_checks.json](numerical_checks.json) records independent evaluations
of the displayed arithmetic. It is not simulation output or a fitted model.
The editable equations and derivations are the primary artifact.

The separate [Lean proof folder](../../proofs/lean/README.md) formalizes a selected
subset, with an explicit [coverage map](../../proofs/lean/PROOF_MAP.md), proof
sources, toolchain pins and retained checker results. Its assumptions and open
formalization obligations remain explicit.

Primary background sources: [Lee et al., paper v1](https://arxiv.org/html/2603.10055v1),
[Gilmer et al.](https://arxiv.org/abs/1704.01212), and
[Mordvintsev et al.](https://distill.pub/2020/growing-ca/).
The note distinguishes its own derivations and constructed examples from those
sources' empirical claims.

Historical experiment SHA-256:
`76b535acbf08050295648d5692d46690366c1f4802d5668c99edeca003cf1981`.
No historical experiment, dataset or checkpoint was changed for this document.
