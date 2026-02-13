"""Plotting routines for the diffusion solver results."""

import os

import matplotlib.pyplot as plt
import numpy as np

from analytical import analytical_solution
from solver import solve_diffusion

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_concentration_profiles(N=65, scheme="forward", filename="concentration_profile.png"):
    """Plot numerical vs analytical concentration profiles (Q.D.a / Q.E.c)."""
    _ensure_results_dir()

    r_num, C_num = solve_diffusion(N, scheme=scheme)
    r_fine = np.linspace(0.0, 0.5, 500)
    C_ana = analytical_solution(r_fine)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_fine, C_ana, "k-", linewidth=2, label="Analytical")
    ax.plot(r_num, C_num, "ro--", markersize=5, label=f"Numerical ({scheme}, N={N})")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("C [mol/m³]")
    ax.set_title("Salt Concentration Profile in Concrete Pillar")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_convergence(results, scheme_label, filename):
    """Plot log-log convergence of error norms (Q.D.b / Q.E.b)."""
    _ensure_results_dir()

    dr = results["dr"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for norm_name, marker in [("L1", "o"), ("L2", "s"), ("Linf", "^")]:
        ax.loglog(dr, results[norm_name], f"-{marker}", label=norm_name)

    # Reference slopes
    dr_ref = np.array([dr.min(), dr.max()])
    scale1 = results["L2"][-1] / dr[-1]
    scale2 = results["L2"][-1] / dr[-1] ** 2
    ax.loglog(dr_ref, scale1 * dr_ref, "k--", alpha=0.4, label="O(Δr)")
    ax.loglog(dr_ref, scale2 * dr_ref**2, "k:", alpha=0.4, label="O(Δr²)")

    ax.set_xlabel("Δr [m]")
    ax.set_ylabel("Error norm")
    ax.set_title(f"Convergence Analysis — {scheme_label}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_comparison(N=65, filename="comparison_both_schemes.png"):
    """Plot both schemes vs analytical on the same figure (Q.E.c)."""
    _ensure_results_dir()

    r_fwd, C_fwd = solve_diffusion(N, scheme="forward")
    r_ctr, C_ctr = solve_diffusion(N, scheme="central")
    r_fine = np.linspace(0.0, 0.5, 500)
    C_ana = analytical_solution(r_fine)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_fine, C_ana, "k-", linewidth=2, label="Analytical")
    ax.plot(r_fwd, C_fwd, "ro--", markersize=5, label=f"Scheme 1 (forward, N={N})")
    ax.plot(r_ctr, C_ctr, "bs--", markersize=5, label=f"Scheme 2 (central, N={N})")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("C [mol/m³]")
    ax.set_title("Comparison of Both FD Schemes vs Analytical Solution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")
