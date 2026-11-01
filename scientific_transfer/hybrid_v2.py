"""steady_hybrid_v2: explicitly derived local steady model, not CEJA replication."""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicHermiteSpline

from .materials import ratios
from .physics import boundary, initial_guess

MODEL_ID = "steady_hybrid_v2"


@dataclass(frozen=True)
class HybridParameters:
    magnetic: float = 0.0  # sqrt(sigma_f B^2/(rho_f a))
    rotation: float = 0.0  # (Omega/a)^2, NOT Omega/a
    buoyancy: float = 0.0  # g beta_f DeltaT/(a^2 R), local pole approximation
    prandtl: float = 1.0
    phi_zinc: float = 0.0  # final-mixture volume fractions
    phi_titania: float = 0.0
    radiation: float = 0.0  # 16 sigma_SB T_inf^3/(3 k_star k_f)
    temperature_ratio: float = 1.0  # Tw/Tinf, >=1

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise ValueError("Finite nonnegative parameters required")
        if self.prandtl <= 0 or self.temperature_ratio < 1:
            raise ValueError("Pr>0, Tw/Tinf>=1 required")
        ratios(self.phi_zinc, self.phi_titania)


def coefficients(p):
    return ratios(p.phi_zinc, p.phi_titania)


def rhs(eta, y, p):
    c = coefficients(p)
    f, u, du, g, dg, theta, dt = y
    absolute_temperature = 1+(p.temperature_ratio-1)*theta
    conductivity = c["k"]+p.radiation*absolute_temperature**3
    if (absolute_temperature <= 0).any() or (conductivity <= 0).any():
        raise ValueError("Nonpositive absolute temperature or effective conductivity")
    conductivity_prime_theta = 3*p.radiation*(p.temperature_ratio-1)*absolute_temperature**2
    return np.array([u, du,
        (c["rho"]*(-2*f*du-1+u*u-p.rotation*g*g)
         +c["sigma"]*p.magnetic**2*(u-1)-c["rho_beta"]*p.buoyancy*theta)/c["mu"],
        dg, (c["rho"]*(-2*f*dg+2*u*g)+c["sigma"]*p.magnetic**2*g)/c["mu"],
        dt, (-2*p.prandtl*c["rho_cp"]*f*dt-conductivity_prime_theta*dt*dt)/conductivity])


def wall(y, p):
    c = coefficients(p)
    return {"mu_ratio_fpp0": float(c["mu"]*y[2, 0]),
            "minus_mu_ratio_gp0": float(-c["mu"]*y[4, 0]),
            "minus_total_conductivity_theta_p0": float(
                -(c["k"]+p.radiation*p.temperature_ratio**3)*y[6, 0])}


def solve(p, length=10.0, points=161, tol=1e-6, max_nodes=5000):
    if not math.isfinite(length) or length <= 0 or points < 5 or tol <= 0 or max_nodes < points:
        raise ValueError("Invalid solver settings")
    x = np.linspace(0, length, points)
    tick = time.perf_counter()
    sol = solve_bvp(lambda x, y: rhs(x, y, p), boundary, x, initial_guess(x),
                    tol=tol, bc_tol=tol, max_nodes=max_nodes)
    seconds = time.perf_counter()-tick
    y = sol.sol(x)
    probe = np.linspace(0, length, 4*points-3)
    defect = float(np.max(np.abs(sol.sol(probe, 1)-rhs(probe, sol.sol(probe), p))))
    bc = float(np.max(np.abs(boundary(y[:, 0], y[:, -1]))))
    if not np.isfinite(y).all() or not math.isfinite(defect):
        raise FloatingPointError("Nonfinite reference output")
    return {"model_id": MODEL_ID, "parameters": asdict(p), "coefficients": coefficients(p),
            "eta": x.tolist(), "state": y.tolist(), "wall": wall(y, p),
            "status": int(sol.status), "message": sol.message, "iterations": int(sol.niter),
            "mesh_nodes": len(sol.x), "seconds": seconds, "boundary_max": bc,
            "ode_defect_max": defect, "accepted": bool(sol.success and bc <= 10*tol and defect <= 100*tol)}


def verify_case(p):
    base = solve(p)
    # Separate tolerance, initial-mesh and domain checks; no conflation of causes.
    variants = {"tolerance": dict(tol=1e-7), "mesh": dict(points=321),
                "domain": dict(length=14.0, points=225)}
    comparisons = {}
    for name, options in variants.items():
        other = solve(p, **options)
        x, y = np.array(other["eta"]), np.array(other["state"])
        interp = CubicHermiteSpline(x, y, rhs(x, y, p), axis=1)(base["eta"])
        profile_error = float(np.max(np.abs(interp[[1, 3, 5]]-np.array(base["state"])[[1, 3, 5]])))
        wall_error = max(abs(other["wall"][k]-base["wall"][k]) for k in base["wall"])
        comparisons[name] = {"profile_max": profile_error, "wall_max": wall_error,
                             "passed": bool(other["accepted"] and max(profile_error, wall_error) <= 1e-4),
                             "solve": other}
    return {"reference": base, "checks": comparisons,
            "accepted": bool(base["accepted"] and all(c["passed"] for c in comparisons.values()))}
