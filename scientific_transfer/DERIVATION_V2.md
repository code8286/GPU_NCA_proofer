# Versioned physical declaration: steady_hybrid_v2

1 November 2026. New development model, not a corrected edition or replication
of CEJA-D-26-02086. V1 source and all earlier data/results remain intact.
This declaration precedes numerical milestone checks; no neural fit is enabled.

## Assumptions and dimensional equations

Use local cylindrical coordinates (r,z) near the pole of a sphere of radius R.
Approximate its tangential geometry by r, with steady imposed strain a>0,
constant angular velocity Omega, constant Tw and Tinf, and uniform normal B.
The effective single-phase mixture is incompressible, Newtonian and dilute.
Properties are constant; only the Rosseland radiative conductivity varies with
temperature. Local radial buoyancy uses g_r approximately g r/R, with a linear
Boussinesq density law. This is not global rotating-sphere CFD.

With radial u, normal w and azimuthal v, the declared boundary-layer equations
are:

```text
(r u)_r + r w_z = 0
rho_h (u u_r + w u_z - v^2/r) = rho_h a^2 r + mu_h u_zz
    - sigma_h B^2 (u-a r) + (rho beta)_h (g r/R) (T-Tinf)
rho_h (u v_r + w v_z + u v/r) = mu_h v_zz - sigma_h B^2 v
(rho cp)_h (u T_r + w T_z) = d/dz[(k_h + k_rad(T)) T_z]
k_rad(T) = 16 sigma_SB T^3/(3 k_star)
```

The radial magnetic term presumes external forcing adjusted to maintain the
prescribed magnetized free stream. Neglect Joule/viscous heating, induced magnetic
field, particle migration, heat sources, slip and unsteadiness. The Rosseland
closure presumes an optically thick medium; no evidence here establishes that
an actual Zn/TiO2 sample satisfies it. Constant-property and mixture closures
are modeling assumptions, not experimentally validated here.

## Similarity reduction

Set eta=z sqrt(a/nu_f), u=a r f', w=-2 sqrt(a nu_f) f,
v=Omega r g, T=Tinf+DeltaT theta, with nu_f=mu_f/rho_f.
These fields satisfy continuity exactly. Radial acceleration becomes
a^2 r[(f')^2-2 f f''-(Omega/a)^2 g^2]. Azimuthal acceleration becomes
2 a Omega r(f'g-fg'). Thermal advection is -2a DeltaT f theta'.
Divide momentum by rho_f a^2 r (azimuthal by rho_f a Omega r) and heat
by a k_f DeltaT/nu_f. At Omega=0 use the regular passive g equation;
dimensional azimuthal velocity is identically zero even though g(0)=1.

Let n_mu=mu_h/mu_f, n_rho=rho_h/rho_f,
n_sigma=sigma_h/sigma_f, n_beta=(rho beta)_h/(rho beta)_f,
n_cp=(rho cp)_h/(rho cp)_f, n_k=k_h/k_f. Then:

```text
n_mu f''' + n_rho(2 f f'' + 1 - f'^2 + rotation g^2)
    - n_sigma magnetic^2 (f'-1) + n_beta buoyancy theta = 0
n_mu g'' + n_rho(2 f g' - 2 f' g) - n_sigma magnetic^2 g = 0
d/deta[K(theta) theta'] + 2 Pr n_cp f theta' = 0
K(theta)=n_k + radiation [1+(temperature_ratio-1)theta]^3

rotation=(Omega/a)^2
magnetic^2=sigma_f B^2/(rho_f a)
buoyancy=g beta_f DeltaT/(a^2 R)
Pr=mu_f cp_f/k_f
radiation=16 sigma_SB Tinf^3/(3 k_star k_f)
temperature_ratio=Tw/Tinf
```

For the thermal RHS use K theta'' + (dK/dtheta)(theta')^2; the implemented
nonlinear radiation term follows this conservative derivative exactly. Tw>=Tinf
and positive absolute temperatures are required. temperature_ratio=1 is also
the isothermal/linearized-temperature limiting equation; it must not be read as
a nonzero dimensional DeltaT measurement. Boundaries are f=f'=0, g=theta=1 at
eta=0, and f'=1,g=theta=0 at the finite-domain approximation L.

## Material convention and limits

Inputs in materials.py transcribe the five property rows of manuscript Table1
(PDF p11), labeling Zn as elemental zinc, not ZnO. They are IMPORTED inputs,
not new measurements or independently certified material constants. Electrical
properties in particular require verification before engineering use.

phi_zinc and phi_titania are FINAL mixture volume fractions. Density, volumetric
heat capacity and rho*beta use direct volume sums with water fraction 1-phi_z-phi_t.
Viscosity ratio is (1-phi_z-phi_t)^(-2.5), a declared dilute-suspension closure.
Conductivities use spherical Maxwell mixing, Zn first with intermediate fraction
phi_z/(1-phi_t), then TiO2 with fraction phi_t. Order dependence is explicit;
do not confuse these fractions with the manuscript's sequential convention.
Require total loading<=0.10; this is a development bound, not a validated range.

Zero loading returns all six ratios exactly one. Setting radiation=0 then
reduces the differential equations to steady_bridge_v1. With additional zero
magnetic, rotation and buoyancy, the f equation is the classical Homann limit.

## Numerical reference and outputs

Freeze milestone2_config.json/source/doc hashes before checking four declared
cases: pure water, Zn only, TiO2 only and a hybrid radiative case. Check tolerance,
initial mesh and domain extension separately at the existing absolute1e-4
profile/wall agreement bound. A separate Chebyshev differentiation/root method
solves the Homann velocity equation, with a linear spectral thermal solve.
It does not call solve_bvp or initialize from its solution. Compare 49/65-node
wall derivatives to the collocation reference at absolute1e-5. A rounded
literature consistency anchor is f''(0)=1.311938, reproduced in Table5 of
[the 2023 Homann study](https://www.mdpi.com/2079-4991/13/6/984/pdf?version=1678276422).
Only that scalar and its limiting-case context are used from that paper.

Report n_mu*f''(0), -n_mu*g'(0) and -K(1)*theta'(0) as explicitly named
dimensionless wall stress/heat-flux coefficients. These are NOT the manuscript's
Cf or Nu normalizations. Agreement between independent discretizations checks
numerics of the declared limit, not every physical closure or full-model branch.

## Milestone boundaries

The exact manuscript's author code/reference values have not been obtained;
visually verified printing problems remain in EQUATION_AUDIT.md. Milestone1 is
therefore only partially resolved. Milestone2 is this explicit alternative model.
Milestone3 neural interface/checkpoint integration/fitting is still pending a
separate freeze; no scientific starts are imported or adapted here. Milestone4
ID/OOD gates remain closed. Milestone5 records only newly measured CPU reference
costs here; historical PPT/language costs are neither zeroed nor mixed into a
speedup claim. No reserved outcome or new learning result is introduced.
