"""heatsolver: finite-difference solvers for the 1D heat equation."""

from .solvers import solve_explicit, solve_implicit, solve_crank_nicolson
from .analysis import exact_solution, max_error, convergence_order

__all__ = [
    "solve_explicit",
    "solve_implicit",
    "solve_crank_nicolson",
    "exact_solution",
    "max_error",
    "convergence_order",
]
