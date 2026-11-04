"""Independent full-system discretization: Chebyshev derivative + nonlinear root.

Shares the DECLARED physical RHS, not solve_bvp or its starting solution.
This is independent numerics, not an independent derivation of the physics.
"""
import time
import numpy as np
from scipy.optimize import root
from scipy.interpolate import BarycentricInterpolator
from .independent_reference import differentiation
from .hybrid_v2 import rhs, wall, coefficients
from .physics import initial_guess, boundary


def solve(p, points=49, length=10.0):
    x, d = differentiation(points, length)
    d2, d3 = d@d, d@d@d
    c = coefficients(p)
    eye = np.eye(points)
    def state(flat):
        f, g, t = flat.reshape(3, points)
        return np.array([f, d@f, d2@f, g, d@g, t, d@t])
    def residual(flat):
        y = state(flat)
        f, g, t = flat.reshape(3, points)
        absolute_t = 1+(p.temperature_ratio-1)*t
        k = c["k"]+p.radiation*absolute_t**3
        dk = 3*p.radiation*(p.temperature_ratio-1)*absolute_t**2
        res = np.array([c["mu"]*(d3@f)+c["rho"]*(2*f*y[2]+1-y[1]**2+p.rotation*g*g)
                        -c["sigma"]*p.magnetic**2*(y[1]-1)+c["rho_beta"]*p.buoyancy*t,
                        c["mu"]*(d2@g)+c["rho"]*(2*f*y[4]-2*y[1]*g)-c["sigma"]*p.magnetic**2*g,
                        k*(d2@t)+dk*y[6]**2+2*p.prandtl*c["rho_cp"]*f*y[6]])
        res[0, 0], res[0, 1], res[0, -1] = f[0], y[1, 0], y[1, -1]-1
        res[1, 0], res[1, -1] = g[0]-1, g[-1]
        res[2, 0], res[2, -1] = t[0]-1, t[-1]
        return res.ravel()
    def jacobian(flat):
        f, g, t = flat.reshape(3, points)
        u, v, dg, dt = d@f, d2@f, d@g, d@t
        eps = p.temperature_ratio-1
        absolute_t = 1+eps*t
        k = c["k"]+p.radiation*absolute_t**3
        dk, ddk = 3*p.radiation*eps*absolute_t**2, 6*p.radiation*eps**2*absolute_t
        ff = c["mu"]*d3+c["rho"]*(2*f[:, None]*d2+2*np.diag(v)-2*u[:, None]*d)-c["sigma"]*p.magnetic**2*d
        fg, ft = np.diag(2*c["rho"]*p.rotation*g), c["rho_beta"]*p.buoyancy*eye
        gf = 2*c["rho"]*(np.diag(dg)-g[:, None]*d)
        gg = c["mu"]*d2+2*c["rho"]*(f[:, None]*d-np.diag(u))-c["sigma"]*p.magnetic**2*eye
        tf = np.diag(2*p.prandtl*c["rho_cp"]*dt)
        tt = k[:, None]*d2+np.diag(dk*(d2@t)+ddk*dt**2)+(2*dk*dt+2*p.prandtl*c["rho_cp"]*f)[:, None]*d
        zero = np.zeros_like(eye)
        j = np.block([[ff, fg, ft], [gf, gg, zero], [tf, zero, tt]])
        for row in (0, 1, points-1, points, 2*points-1, 2*points, 3*points-1):
            j[row] = 0
        j[0, 0] = 1
        j[1, :points], j[points-1, :points] = d[0], d[-1]
        j[points, points], j[2*points-1, 2*points-1] = 1, 1
        j[2*points, 2*points], j[3*points-1, 3*points-1] = 1, 1
        return j
    tick = time.perf_counter()
    result = root(residual, initial_guess(x)[[0, 3, 5]].ravel(), jac=jacobian, tol=1e-9)
    y = state(result.x)
    target = np.linspace(0, length, 161)
    interp = BarycentricInterpolator(x, result.x.reshape(3, points), axis=1, rng=0)
    values, first, second, third = interp(target), interp.derivative(target), interp.derivative(target, 2), interp.derivative(target, 3)
    exported = np.array([values[0], first[0], second[0], values[1], first[1], values[2], first[2]])
    derivative = np.array([first[0], second[0], third[0], first[1], second[1], first[2], second[2]])
    dense_defect = float(np.max(np.abs(derivative-rhs(target, exported, p))))
    bc = float(np.max(np.abs(boundary(y[:, 0], y[:, -1]))))
    residual_max = float(np.max(np.abs(residual(result.x))))
    return {"method": "three-field Chebyshev/root + analytic Jacobian, seven states recovered by differentiation; ordinary initial guess",
            "root_success": bool(result.success), "root_message": str(result.message),
            "seconds": time.perf_counter()-tick, "collocation_residual_max": residual_max,
            "dense_defect_max": dense_defect, "boundary_max": bc,
            "eta": target.tolist(), "state": exported.tolist(), "wall": wall(y, p),
            "accepted": bool(result.success and np.isfinite(exported).all()
                             and residual_max <= 1e-7 and bc <= 1e-7 and dense_defect <= 1e-4)}
