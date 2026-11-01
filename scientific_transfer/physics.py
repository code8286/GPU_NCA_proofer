"""Declared steady bridge, NOT a transcription/replication of CEJA-D-26-02086."""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.integrate import solve_bvp


@dataclass(frozen=True)
class Parameters:
    magnetic: float = 0.0
    rotation: float = 0.0
    buoyancy: float = 0.0
    prandtl: float = 1.0

    def __post_init__(self):
        values = np.array(list(asdict(self).values()))
        if not np.isfinite(values).all() or min(values) < 0 or self.prandtl <= 0:
            raise ValueError("Finite nonnegative parameters and positive Pr required")


def rhs(eta, y, p):
    """y = [f, f', f'', g, g', theta, theta']; eta is distance, not time.

    f''' + 2 f f'' + 1 - f'^2 + rotation*g^2
         - magnetic^2*(f'-1) + buoyancy*theta = 0
    g'' + 2 f g' - 2 f' g - magnetic^2*g = 0
    theta'' + 2 Pr f theta' = 0
    """
    f, u, du, g, dg, theta, dt = y
    return np.array([u, du,
                     -2*f*du - 1 + u*u - p.rotation*g*g
                     + p.magnetic**2*(u-1) - p.buoyancy*theta,
                     dg, -2*f*dg + 2*u*g + p.magnetic**2*g,
                     dt, -2*p.prandtl*f*dt])


def boundary(a, b):
    return np.array([a[0], a[1], a[3]-1, a[5]-1,
                     b[1]-1, b[3], b[5]])


def initial_guess(x):
    e = np.exp(-x)
    return np.array([x-1+e, 1-e, e, e, -e, e, -e])


def wall_quantities(y):
    # Unscaled derivatives only: no claim of manuscript Cf or Nu normalization.
    return {"fpp0": float(y[2, 0]), "minus_gp0": float(-y[4, 0]),
            "minus_theta_p0": float(-y[6, 0])}


def solve(p, length=10.0, points=161, tol=1e-6, max_nodes=5000, guess=None):
    if length <= 0 or points < 5 or tol <= 0 or max_nodes < points:
        raise ValueError("Invalid solver settings")
    x = np.linspace(0, length, points)
    y0 = initial_guess(x) if guess is None else np.asarray(guess, dtype=float)
    if y0.shape != (7, points) or not np.isfinite(y0).all():
        raise ValueError("Guess must be finite with shape (7, points)")
    tick = time.perf_counter()
    sol = solve_bvp(lambda t, y: rhs(t, y, p), boundary, x, y0,
                    tol=tol, bc_tol=tol, max_nodes=max_nodes)
    elapsed = time.perf_counter()-tick
    y = sol.sol(x)
    probe = np.linspace(0, length, 4*points-3)
    defect = sol.sol(probe, 1)-rhs(probe, sol.sol(probe), p)
    bc = float(np.max(np.abs(boundary(y[:, 0], y[:, -1]))))
    residual = float(np.max(np.abs(defect)))
    finite = bool(np.isfinite(y).all() and np.isfinite(residual))
    if not finite:
        raise FloatingPointError("Nonfinite solver output; reference case must remain failed")
    accepted = bool(sol.success and finite and bc <= 10*tol and residual <= 100*tol)
    return {"parameters": asdict(p), "eta": x.tolist(), "state": y.tolist(),
            "status": int(sol.status), "message": sol.message,
            "accepted": accepted, "iterations": int(sol.niter),
            "mesh_nodes": int(len(sol.x)), "seconds": elapsed,
            "boundary_max": bc, "ode_defect_max": residual,
            "wall": wall_quantities(y), "length": length, "tol": tol}
