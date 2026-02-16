"""
Convergence analysis: error norms and observed convergence orders.
"""

import numpy as np

from analytical import analytical_solution
from solver import solve_diffusion

def compute_error_norms(C_numerical, C_analytical):
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
    error = np.abs(C_numerical - C_analytical)
    N = len(error)
    L1 = np.sum(error) / N
    L2 = np.sqrt(np.sum(error**2) / N)
    Linf = np.max(error)
    return L1, L2, Linf

def convergence_study(scheme="forward", initial_grid_size=5, num_refinements=8):
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
        r, C_num = solve_diffusion(N, scheme=scheme)
        C_ana = analytical_solution(r)
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