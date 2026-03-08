"""
Convergence analysis: error norms and observed convergence orders.
"""

import numpy as np

from analytical import manufactured_solution
from solver import solve_diffusion

def compute_error_norms(C_numerical, C_manufactured):
    """Compute L1, L2, and L-infinity error norms.

    Parameters
    ----------
    C_numerical : ndarray
        Numerical solution.
    C_analytical : ndarray
        Analytical solution evaluated at the same grid points.

    Returns
    -------
    L1, L2, Linf : float
        Error norms.
    """
    error = np.abs(C_numerical - C_manufactured)
    N_t = len(error[:, 0])
    N_r = len(error[0, :])
    L1 = np.sum(error) / (N_t * N_r)
    L2 = np.sqrt(np.sum(error**2) / (N_t * N_r)) 
    Linf = np.max(error)
    return L1, L2, Linf

def convergence_study(initial_grid_size=5, Nt=200, t=1.0, num_refinements=8):
    """Run a grid convergence study for the given finite difference scheme.

    Parameters
    ----------
    scheme : str
        "forward" or "central".
    initial_grid_size : int
        Initial number of grid points (including boundaries).
    num_refinements : int
        Number of grid refinements to perform.

    Returns
    -------
    results : dict
        Keys: "N", "dr", "L1", "L2", "Linf", "order_L1", "order_L2", "order_Linf".
        Each value is a list (or array) over refinement levels.
    """
    drs = np.empty(num_refinements)
    L1s = np.empty(num_refinements)
    L2s = np.empty(num_refinements)
    Linfs = np.empty(num_refinements)
    
    # Divide the domain by 2 at each refinement.
    grid_sizes = [(initial_grid_size-1) * (2**i) + 1 for i in range(num_refinements)]
    
    for i, N in enumerate(grid_sizes):
        r, C_num = solve_diffusion(N, T_max=t, t_steps=Nt)
        time = np.linspace(0, t, Nt)
        C_ana = manufactured_solution(r, time)
        L1, L2, Linf = compute_error_norms(C_num, C_ana)

        drs[i] = r[1] - r[0]
        L1s[i] = L1
        L2s[i] = L2
        Linfs[i] = Linf

    order_L1 = np.log(L1s[:-1] / L1s[1:]) / np.log(drs[:-1] / drs[1:])
    order_L2 = np.log(L2s[:-1] / L2s[1:]) / np.log(drs[:-1] / drs[1:])
    order_Linf = np.log(Linfs[:-1] / Linfs[1:]) / np.log(drs[:-1] / drs[1:])

    return {
        "N": grid_sizes,
        "dr": drs,
        "L1": L1s,
        "L2": L2s,
        "Linf": Linfs,
        "order_L1": order_L1,
        "order_L2": order_L2,
        "order_Linf": order_Linf,
    }

def convergence_study_time(initial_grid_size=5, Nr=200, t=1.0, R=0.5, num_refinements=8):
    """Run a grid convergence study for the given finite difference scheme.

    Parameters
    ----------
    scheme : str
        "forward" or "central".
    initial_grid_size : int
        Initial number of grid points (including boundaries).
    num_refinements : int
        Number of grid refinements to perform.

    Returns
    -------
    results : dict
        Keys: "N", "dr", "L1", "L2", "Linf", "order_L1", "order_L2", "order_Linf".
        Each value is a list (or array) over refinement levels.
    """
    drs = np.empty(num_refinements)
    L1s = np.empty(num_refinements)
    L2s = np.empty(num_refinements)
    Linfs = np.empty(num_refinements)
    
    # Divide the domain by 2 at each refinement.
    grid_sizes = [(initial_grid_size-1) * (2**i) + 1 for i in range(num_refinements)]
    
    for i, N in enumerate(grid_sizes):
        r, C_num = solve_diffusion(Nr, T_max=t, t_steps=N)
        time = np.linspace(0, t, N)
        C_ana = manufactured_solution(r, time)
        L1, L2, Linf = compute_error_norms(C_num, C_ana)

        drs[i] = r[1] - r[0]
        L1s[i] = L1
        L2s[i] = L2
        Linfs[i] = Linf

    order_L1 = np.log(L1s[:-1] / L1s[1:]) / np.log(drs[:-1] / drs[1:])
    order_L2 = np.log(L2s[:-1] / L2s[1:]) / np.log(drs[:-1] / drs[1:])
    order_Linf = np.log(Linfs[:-1] / Linfs[1:]) / np.log(drs[:-1] / drs[1:])

    return {
        "N": grid_sizes,
        "dr": drs,
        "L1": L1s,
        "L2": L2s,
        "Linf": Linfs,
        "order_L1": order_L1,
        "order_L2": order_L2,
        "order_Linf": order_Linf,
    }