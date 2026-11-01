"""Independent Chebyshev/root Homann + linear thermal solver (no solve_bvp)."""
import numpy as np
from scipy.optimize import root


def differentiation(points, length):
    z = -np.cos(np.pi*np.arange(points)/(points-1))
    x = length*(z+1)/2
    weights = (-1.0)**np.arange(points)
    weights[[0, -1]] *= 0.5
    delta = x[:, None]-x[None, :]
    np.fill_diagonal(delta, 1)
    d = weights[None, :]/weights[:, None]/delta
    np.fill_diagonal(d, 0)
    np.fill_diagonal(d, -d.sum(axis=1))
    return x, d


def homann(points=65, length=10.0, prandtl=1.0):
    if points < 9 or length <= 0 or prandtl <= 0:
        raise ValueError("Invalid spectral benchmark settings")
    x, d = differentiation(points, length)
    d2, d3 = d@d, d@d@d
    def residual(f):
        u, v = d@f, d2@f
        res = d3@f+2*f*v+1-u*u
        res[0], res[1], res[-1] = f[0], u[0], u[-1]-1
        return res
    def jacobian(f):
        result = d3+2*f[:, None]*d2+2*np.diag(d2@f)-2*(d@f)[:, None]*d
        result[0] = 0
        result[0, 0] = 1
        result[1], result[-1] = d[0], d[-1]
        return result
    result = root(residual, x-1+np.exp(-x), jac=jacobian, tol=1e-9)
    f = result.x
    matrix = d2+2*prandtl*f[:, None]*d
    matrix[0], matrix[-1] = 0, 0
    matrix[0, 0], matrix[-1, -1] = 1, 1
    target = np.zeros(points)
    target[0] = 1
    theta = np.linalg.solve(matrix, target)
    error = float(np.max(np.abs(residual(f))))
    return {"method": "Chebyshev differentiation + analytic-Jacobian nonlinear root + thermal linear solve",
            "points": points, "length": length, "prandtl": prandtl,
            "root_success": bool(result.success), "root_message": str(result.message),
            "residual_max": error, "accepted": bool(result.success and error < 1e-5),
            "fpp0": float((d2@f)[0]), "minus_theta_p0": float(-(d@theta)[0])}
