


"""Generate the figures and convergence results used in the README."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from heatsolver import (
    solve_explicit,
    solve_implicit,
    solve_crank_nicolson,
)
from heatsolver.analysis import exact_solution, convergence_order


def main():
    alpha = 1.0
    T = 0.1
    nx = 100
    dt_factor = 0.1

    output_dir = Path(__file__).resolve().parents[1] / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Numerical and exact solutions at several times
    # ---------------------------------------------------------
    x = np.linspace(0.0, 1.0, nx + 1)
    times = [0.025, 0.05, 0.1]

    fig, axes = plt.subplots(1, len(times), figsize=(15, 4))

    for ax, t in zip(axes, times):
        nt = max(1, int(np.ceil(t / (dt_factor * (1.0 / nx) ** 2))))

        x_num, u_num = solve_crank_nicolson(alpha, nx, nt, t)
        u_exact = exact_solution(x, t, alpha)

        ax.plot(x_num, u_num, label="Crank-Nicolson")
        ax.plot(x, u_exact, "--", label="Soluzione esatta")
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Equazione del calore: soluzione numerica ed esatta")
    fig.tight_layout()
    fig.savefig(output_dir / "solution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # 2. Convergence errors and observed orders
    # ---------------------------------------------------------
    nx_values = [10, 20, 40, 80]
    dx_values = 1.0 / np.asarray(nx_values, dtype=float)

    methods = [
        ("Esplicito", solve_explicit),
        ("Implicito", solve_implicit),
        ("Crank-Nicolson", solve_crank_nicolson),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    print("\nOrdini di convergenza osservati")
    print("-" * 65)
    print(f"{'Metodo':<20} {'Ordine':>10} {'Errori per nx = 10, 20, 40, 80'}")
    print("-" * 65)

    for name, solver in methods:
        errors, order = convergence_order(
            solver, alpha, T, nx_values, dt_factor
        )

        ax.loglog(dx_values, errors, marker="o", label=name)
        error_text = ", ".join(f"{error:.3e}" for error in errors)
        print(f"{name:<20} {order:>10.3f}   {error_text}")

    ax.set_xlabel("Passo spaziale dx")
    ax.set_ylabel("Errore massimo al tempo T")
    ax.set_title("Convergenza dei metodi numerici")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "convergence.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    
    # ---------------------------------------------------------
    # 3. Stability experiment for the explicit Euler method
    # ---------------------------------------------------------
    nx_stability = 50
    dx_stability = 1.0 / nx_stability

    print("\nEsperimento di stabilità del metodo esplicito")
    print("-" * 75)
    print(f"{'r richiesto':>12} {'r effettivo':>12} {'nt':>8} {'max|u(T)|':>18}")
    print("-" * 75)

    for r_target in (0.4, 0.6):
        dt_target = r_target * dx_stability**2 / alpha
        nt_stability = int(np.ceil(T / dt_target))

        _, u_stability = solve_explicit(
            alpha, nx_stability, nt_stability, T
        )

        r_effective = alpha * (T / nt_stability) / dx_stability**2
        max_amplitude = np.max(np.abs(u_stability))

        print(
            f"{r_target:12.1f} {r_effective:12.4f} "
            f"{nt_stability:8d} {max_amplitude:18.6e}"
        )

    print(f"\nFigure salvate nella cartella: {output_dir}")
    print(" - solution.png")
    print(" - convergence.png")


if __name__ == "__main__":
    main()