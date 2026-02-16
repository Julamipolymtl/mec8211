"""
Plotting routines for the diffusion solver results and convergence analysis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from analytical import analytical_solution
from solver import solve_diffusion
from convergence import *

# Save plots in ../results/ relative to this script (in src/)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def plot_concentration_profiles(N=5, scheme="forward", filename="concentration_profile.png"):
    """
    Plot numerical vs analytical concentration profiles for a given grid size and scheme.
    
    Parameters
    ----------
    N : int
        Number of grid points (including boundaries) for the numerical solution.
    scheme : str
        Finite difference scheme to use ("forward" or "central").
    filename : str
        Name of the output PNG file to save the plot.
    """

    r_num, C_num = solve_diffusion(N, scheme=scheme)
    r_analytical = np.linspace(0.0, 0.5, 500)
    C_ana = analytical_solution(r_analytical)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_analytical, C_ana, color="black", linewidth=2, label="Analytical")
    ax.plot(r_num, C_num, color="red", ls="--", marker="o", markersize=5, label=f"Numerical ({scheme}, N={N})")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("C $[mol/m^3]$")
    ax.set_title("Salt Concentration Profile in Concrete Pillar")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    
    filepath = os.path.join(RESULTS_DIR, filename)
    fig.savefig(filepath, dpi=300)
    
def plot_convergence(scheme="forward", filename="convergence_plot.png"):
    """
    Plot log-log convergence of error norms for a given finite difference scheme.
    
    Parameters
    ----------
    scheme : str
        Finite difference scheme to use ("forward" or "central").
    filename : str
        Name of the output PNG file to save the plot.
    """
    results = convergence_study(scheme=scheme)
    
    dr = results["dr"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for norm_name, marker in [("L1", "o"), ("L2", "s"), ("Linf", "^")]:
        ax.loglog(dr, results[norm_name], f"-{marker}", label=norm_name)

    # Reference slopes for O(r) and O(r2)   
    dr_ref = np.array([dr.min(), dr.max()])
    scale1 = results["L2"][0] / dr[0]
    scale2 = results["L2"][0] / dr[0]**2
    ax.loglog(dr_ref, scale1 * dr_ref, "k--", label=r"$O(\Delta r)$")
    ax.loglog(dr_ref, scale2 * dr_ref**2, "k:", label=r"$O(\Delta r^2)$")

    ax.set_xlabel(r"$\Delta r$ [m]")
    ax.set_ylabel("Error norm")
    ax.set_title(f"Convergence Analysis — {scheme} Scheme")
    ax.legend()
    ax.grid()
    
    filename = filename.replace(".png", f"_{scheme}.png")
    filepath = os.path.join(RESULTS_DIR, filename)
    fig.savefig(filepath, dpi=300)