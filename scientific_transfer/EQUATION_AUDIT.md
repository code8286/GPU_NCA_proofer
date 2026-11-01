# Manuscript audit and deliberate bridge boundary

Source: user-supplied CEJA-D-26-02086, a 29-page manuscript draft titled
"Exploring Radiation Acts Hybrid Nanomaterial over the stagnation point region
on a rotating sphere with Engineering applications". It is not treated as a
published or experimentally validated reference. The PDF remains unmodified
outside this repository; no embedded document instruction is executed.

Original PDF SHA256:
`e3b598624e24b8340a14c2b74cd407ad97daf506fda3f3799d22ca74f9f9c732`.

## Open items before faithful replication

- PDF pp. 11-12, equations 9-12: three dependent functions give a seven-state
  first-order system, with four wall and three far-field boundary conditions.
- PDF p. 13, equations 18-20: visually comparing against equations 9-11
  RESOLVES the earlier suspected Pr mismatch: the entire numerator in Eq11
  is over Pr, and Eq20 agrees algebraically. The earlier concern came from
  text extraction and is withdrawn, not confirmed as a manuscript defect.
- PDF pp. 8 and 11: reconcile free-stream velocity and the x-dependent similarity
  transformation, including time-dependent rotation/acceleration definitions.
- PDF pp. 9-10, equations 2-7: audit dimensional balance, magnetic coupling,
  density/mixture-property formulas and which particle is Zn versus ZnO.
- PDF pp. 12-13, equations 13-16: establish exact heat-flux and skin-friction
  normalizations, reference velocity and radiative wall flux from the derivation.
- PDF pp. 13-14: distinguish any implemented shooting scheme from BVP4C.
  BVP4C itself is collocation; the draft repeatedly uses both descriptions.
- Obtain author reference values/code or independently derive limiting cases.
  Smooth profiles and solver convergence cannot validate the manuscript physics.

Full manuscript replication is disabled, rather than silently choosing a rewrite
or patching the printed formulas. These are audit questions, not a completed
peer review or a proof that every suspected discrepancy is an author error.

## Visual audit update, 1 November 2026

Rendered and inspected complete PDF pages9-13 using PDFium. Beyond the resolved
Pr question, these issues are visibly printed, not extraction artifacts:

- Eq7 density uses (1-1)rho_f. Literal zero particle fractions give rho_h=0.
  Replacing it with (1-Psi1)rho_f would recover a base-fluid limit under a
  sequential convention, but that author-intended correction is unconfirmed.
- Eq2 uses dr/dy in the centrifugal term although r=r(x); this would eliminate
  the printed centrifugal contribution. Its external acceleration terms lack
  density while the left side and viscous term retain it.
- Eq3 ends in -sigma_h B0 V, rather than azimuthal damping proportional to
  B0 squared times W. No silent substitution is made.
- Eq11 and20 both contain eta*theta/4 (not theta prime); their algebraic
  agreement does not establish the term follows from the stated dimensional PDE.

Searches for the exact title and manuscript number did not locate author code
or an exact published reference. Related rotating-sphere papers are not substitutes
for this manuscript. No contact message was sent to authors. The exact manuscript
model remains blocked for faithful replication; the separately declared
[steady_hybrid_v2 derivation](DERIVATION_V2.md) does not resolve author intent.

## Explicit executable alternative: steady_bridge_v1

With eta the dimensionless wall-normal coordinate and primes denoting eta
derivatives, the independently DECLARED development equations are:

```text
f''' + 2 f f'' + 1 - (f')^2 + rotation g^2
     - magnetic^2 (f' - 1) + buoyancy theta = 0
g'' + 2 f g' - 2 f' g - magnetic^2 g = 0
theta'' + 2 Pr f theta' = 0

f(0)=f'(0)=0; g(0)=theta(0)=1
f'(L)=1; g(L)=theta(L)=0
```

This is a steady, constant-property bridge, with no claimed equivalence to
equations 9-11 or 18-20 of the draft. Rotation is a coefficient of g squared;
magnetic is squared in both momentum equations; no dimensional units are
assigned to these controls. Buoyancy is nonnegative in the initial range.
No graph topology is used by the reference solver; its mesh is one-dimensional.

References for NUMERICAL METHODS only:

- [SciPy solve_bvp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html): fourth-order collocation with residual control.
- [MATLAB bvp4c](https://www.mathworks.com/help/matlab/ref/bvp4c.html): collocation, not a shooting method.

Neither numerical-method reference validates this fluid model. SciPy is used
as an open local development solver, not claimed as a bit-exact MATLAB replica.
