"""Finite-difference solvers for u_t = alpha * u_xx on [0, 1].

Boundary conditions: u(0, t) = u(1, t) = 0.
Initial condition:   u(x, 0) = sin(pi * x).

All solvers share the same signature and return (x, u) where x is the
spatial grid (nx + 1 points) and u the numerical solution at time T.


"""


"""Finite-difference solvers for u_t = alpha * u_xx on [0, 1].

Boundary conditions: u(0, t) = u(1, t) = 0.
Initial condition:   u(x, 0) = sin(pi * x).
"""

import numpy as np
from scipy.linalg import solve_banded


def solve_explicit(alpha: float, nx: int, nt: int, T: float):
    """Forward Euler in time, central differences in space (FTCS)."""
    dx = 1.0 / nx
    dt = T / nt
    r = alpha * dt / dx**2

    x = np.linspace(0.0, 1.0, nx + 1)
    u = np.sin(np.pi * x)
    u[0] = u[-1] = 0.0

    for _ in range(nt):
        u_new = u.copy()
        u_new[1:-1] = (
            u[1:-1]
            + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        )
        u = u_new

    return x, u


def solve_implicit(alpha: float, nx: int, nt: int, T: float):
    """Backward Euler in time, central differences in space."""
    dx = 1.0 / nx
    dt = T / nt
    r = alpha * dt / dx**2

    x = np.linspace(0.0, 1.0, nx + 1)
    u = np.sin(np.pi * x)
    u[0] = u[-1] = 0.0

    n = nx - 1
    ab = np.zeros((3, n))
    ab[0, 1:] = -r
    ab[1, :] = 1.0 + 2.0 * r
    ab[2, :-1] = -r

    for _ in range(nt):
        u[1:-1] = solve_banded((1, 1), ab, u[1:-1])

    return x, u


def solve_crank_nicolson(alpha: float, nx: int, nt: int, T: float):
    """Crank-Nicolson: second-order time and space discretization."""
    dx = 1.0 / nx
    dt = T / nt
    r = alpha * dt / dx**2

    x = np.linspace(0.0, 1.0, nx + 1)
    u = np.sin(np.pi * x)
    u[0] = u[-1] = 0.0

    n = nx - 1
    ab = np.zeros((3, n))
    ab[0, 1:] = -r / 2.0
    ab[1, :] = 1.0 + r
    ab[2, :-1] = -r / 2.0

    for _ in range(nt):
        rhs = (
            (1.0 - r) * u[1:-1]
            + (r / 2.0) * (u[2:] + u[:-2])
        )
        u[1:-1] = solve_banded((1, 1), ab, rhs)

    return x, u