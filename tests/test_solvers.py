"""Tests are written first on purpose: they fail until the solvers exist."""

import numpy as np
import pytest

from heatsolver import (
    solve_explicit, solve_implicit, solve_crank_nicolson,
    exact_solution, max_error,
)

ALPHA, T = 1.0, 0.1


def test_exact_solution_at_t0_is_initial_condition():
    x = np.linspace(0, 1, 11)
    assert np.allclose(exact_solution(x, 0.0, ALPHA), np.sin(np.pi * x))


def test_explicit_stable_matches_exact():
    nx = 20
    nt = 400  # dt = 0.00025, dx**2 = 0.0025 -> r = 0.1 <= 0.5
    x, u = solve_explicit(ALPHA, nx, nt, T)
    assert max_error(x, u, T, ALPHA) < 1e-3


def test_explicit_unstable_above_half():
    nx = 20
    nt = 50  # dt = 0.002, dx**2 = 0.0025 -> r = 0.8 > 0.5
    x, u = solve_explicit(ALPHA, nx, nt, T)
    assert max_error(x, u, T, ALPHA) > 1.0


@pytest.mark.parametrize("solver", [solve_implicit, solve_crank_nicolson])
def test_implicit_methods_stable_with_large_dt(solver):
    x, u = solver(ALPHA, 20, 10, T)  # r = 4: explicit would blow up
    assert np.all(np.isfinite(u))
    assert max_error(x, u, T, ALPHA) < 5e-2


def test_crank_nicolson_more_accurate_than_implicit():
    x, u_i = solve_implicit(ALPHA, 40, 40, T)
    _, u_cn = solve_crank_nicolson(ALPHA, 40, 40, T)
    assert max_error(x, u_cn, T, ALPHA) < max_error(x, u_i, T, ALPHA)


@pytest.mark.parametrize(
    "solver", [solve_explicit, solve_implicit, solve_crank_nicolson]
)
def test_boundary_conditions_hold(solver):
    x, u = solver(ALPHA, 20, 400, T)
    assert u[0] == pytest.approx(0.0) and u[-1] == pytest.approx(0.0)
