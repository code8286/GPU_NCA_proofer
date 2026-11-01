"""Score full seven-state guesses; invalid predictions never disappear from coverage."""
import numpy as np
from .physics import boundary, rhs, wall_quantities


def score(eta, predicted, reference, parameters):
    x = np.asarray(eta, dtype=float)
    ref = np.asarray(reference, dtype=float)
    try:
        y = np.asarray(predicted, dtype=float)
    except (TypeError, ValueError):
        y = np.array([])
    if x.ndim != 1 or len(x) < 5 or not np.isfinite(x).all() or (np.diff(x) <= 0).any():
        raise ValueError("Strictly increasing finite mesh required")
    if ref.shape != (7, len(x)) or not np.isfinite(ref).all():
        raise ValueError("Invalid reference")
    if y.shape != ref.shape or not np.isfinite(y).all():
        return {"valid": False, "profile_rmse": None, "boundary_max": None,
                "finite_difference_ode_defect_max": None, "wall_absolute_errors": None}
    # Finite-difference defect is a diagnostic, not the collocation acceptance test.
    derivative = np.gradient(y, x, axis=1, edge_order=2)
    predicted_wall, true_wall = wall_quantities(y), wall_quantities(ref)
    return {"valid": True,
            "profile_rmse": float(np.sqrt(np.mean((y[[1, 3, 5]]-ref[[1, 3, 5]])**2))),
            "boundary_max": float(np.max(np.abs(boundary(y[:, 0], y[:, -1])))),
            "finite_difference_ode_defect_max": float(np.max(np.abs(derivative-rhs(x, y, parameters)))),
            "wall_absolute_errors": {k: abs(predicted_wall[k]-true_wall[k]) for k in true_wall}}
