"""
Plotting routines for the diffusion solver results and convergence analysis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from solver import DiffusionParams, solve_diffusion


def _results_dir(params):
    """Return the results directory for this run, creating it if needed."""
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    path = os.path.join(base, params.run_name)
    os.makedirs(path, exist_ok=True)
    return path


def plot_concentration_profiles(params, filename="concentration_profile.png"):
    """Plot the physical concentration profile (no MMS source) at final time.

    Parameters
    ----------
    params : DiffusionParams
        Solver parameters. mms is forced to False for the physical run.
    filename : str
        Output PNG filename.
    """
    from dataclasses import replace
    params = replace(params, mms=False)
    r_num, t_num, C_num = solve_diffusion(params)
    t_final = t_num[-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_num, C_num[-1, :], color="red", ls="--", marker="o",
            markersize=5, label=f"Numerical (N={params.N_r})")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("C $[mol/m^3]$")
    ax.set_title(f"Salt Concentration Profile in Concrete Pillar at t = {t_final:.2e} s")
    ax.legend()
    ax.grid()
    fig.tight_layout()

    fig.savefig(os.path.join(_results_dir(params), filename), dpi=300)


def plot_convergence(params, results, ctime=False, filename="convergence_plot.png"):
    """Log-log convergence plot of L1, L2, and Linf error norms.

    Parameters
    ----------
    params : DiffusionParams
        Used for run_name to determine output directory.
    results : dict
        Output of convergence_study_spatial or convergence_study_temporal.
    ctime : bool
        If True, label the x-axis as dt instead of dr.
    filename : str
        Output PNG filename.
    """
    dr = results["dr"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for norm_name, marker in [("L1", "o"), ("L2", "s"), ("Linf", "^")]:
        mean_order = float(np.mean(results[f"order_{norm_name}"]))
        label = f"{norm_name}  (p = {mean_order:.2f})"
        ax.loglog(dr, results[norm_name], f"-{marker}", label=label)

    ax.set_xlabel(r"$\Delta t$ [s]" if ctime else r"$\Delta r$ [m]")
    ax.set_ylabel("Error norm")
    ax.set_title("Convergence Analysis")
    ax.legend()
    ax.grid()

    fig.savefig(os.path.join(_results_dir(params), filename), dpi=300)


def plot_mms(params, filename="mms_heatmap.png"):
    """Heatmap of the manufactured solution C_mms(r, t).

    Parameters
    ----------
    params : DiffusionParams
        Used to derive C_fn and for t_max and run_name.
    filename : str
        Output PNG filename.
    """
    C_fn, _ = params.mms_functions()
    r = np.linspace(0, params.R, 200)
    t = np.linspace(0, params.t_max, 200)

    C_grid = C_fn(r[np.newaxis, :], t[:, np.newaxis])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(r, t, C_grid, shading="auto", cmap="viridis")
    fig.colorbar(im).set_label("C $[mol/m^3]$")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("t [s]")
    ax.set_title("MMS Solution")
    fig.tight_layout()

    fig.savefig(os.path.join(_results_dir(params), filename), dpi=300)


def plot_concentration_heatmap(params, filename="concentration_heatmap.png"):
    """Heatmap of the numerical concentration C(r, t) for the physical problem.

    Parameters
    ----------
    params : DiffusionParams
        Solver parameters. mms is forced to False for the physical run.
    filename : str
        Output PNG filename.
    """
    from dataclasses import replace
    params = replace(params, mms=False)
    r_num, t_num, C_num = solve_diffusion(params)

    # Drop t=0 (log-undefined) for the log-scale axis
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(r_num, t_num[1:], C_num[1:, :], shading="auto", cmap="viridis")
    fig.colorbar(im).set_label("C $[mol/m^3]$")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("t [s]")
    ax.set_yscale("log")
    ax.set_title("Salt Concentration C(r, t) in Concrete Pillar")
    fig.tight_layout()

    fig.savefig(os.path.join(_results_dir(params), filename), dpi=300)


def plot_sourceterm(params, filename="mms_source.png"):
    """Heatmap of the MMS source term S_mms(r, t).

    Parameters
    ----------
    params : DiffusionParams
        Used to derive S_fn and for t_max and run_name.
    filename : str
        Output PNG filename.
    """
    _, S_fn = params.mms_functions()
    r = np.linspace(0, params.R, 200)
    t = np.linspace(0, params.t_max, 200)
    R_grid, T_grid = np.meshgrid(r, t)

    S_grid = S_fn(R_grid, T_grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(r, t, S_grid, shading="auto", cmap="viridis")
    fig.colorbar(im).set_label("S $[mol/m^3/s]$")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("t [s]")
    ax.set_title("MMS Source Term")
    fig.tight_layout()

    fig.savefig(os.path.join(_results_dir(params), filename), dpi=300)
