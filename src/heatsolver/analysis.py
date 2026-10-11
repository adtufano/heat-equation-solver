


"""Error measurement and convergence analysis."""

import numpy as np


def exact_solution(x, t: float, alpha: float):
    """Exact solution for u(x,0) = sin(pi x)."""
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)


def max_error(x, u_numeric, t: float, alpha: float) -> float:
    """Maximum absolute difference between numerical and exact solution."""
    u_exact = exact_solution(x, t, alpha)
    return float(np.max(np.abs(u_numeric - u_exact)))


def convergence_order(solver, alpha: float, T: float, nx_values, dt_factor: float):
    """Estimate convergence order using dt approximately proportional to dx**2."""
    nx_values = np.asarray(nx_values, dtype=int)

    if nx_values.size < 2:
        raise ValueError("At least two grid resolutions are required.")
    if np.any(nx_values <= 0):
        raise ValueError("All nx values must be positive.")
    if dt_factor <= 0 or T <= 0:
        raise ValueError("dt_factor and T must be positive.")

    dx_values = 1.0 / nx_values
    errors = []

    for nx, dx in zip(nx_values, dx_values):
        dt_target = dt_factor * dx**2
        nt = max(1, int(np.ceil(T / dt_target)))

        x, u_numeric = solver(alpha, int(nx), nt, T)
        error = max_error(x, u_numeric, T, alpha)
        errors.append(error)

    errors = np.asarray(errors)

    if np.any(errors <= 0):
        raise ValueError("Errors must be positive to estimate convergence order.")

    slope, _ = np.polyfit(np.log(dx_values), np.log(errors), 1)

    return errors, float(slope)