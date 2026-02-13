"""Convergence analysis: error norms and observed convergence orders."""

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
    e = np.abs(C_numerical - C_analytical)
    N = len(e)
    L1 = np.sum(e) / N
    L2 = np.sqrt(np.sum(e**2) / N)
    Linf = np.max(e)
    return L1, L2, Linf


def convergence_study(scheme="forward", grid_sizes=None):
    """Run a grid convergence study for the given finite difference scheme.

    Parameters
    ----------
    scheme : str
        "forward" or "central".
    grid_sizes : list of int, optional
        Number of grid points for each refinement level.
        Defaults to [5, 9, 17, 33, 65, 129, 257, 513].

    Returns
    -------
    results : dict
        Keys: "N", "dr", "L1", "L2", "Linf", "order_L1", "order_L2", "order_Linf".
        Each value is a list (or array) over refinement levels.
    """
    if grid_sizes is None:
        grid_sizes = [5, 9, 17, 33, 65, 129, 257, 513]

    drs = []
    L1s, L2s, Linfs = [], [], []

    for N in grid_sizes:
        r, C_num = solve_diffusion(N, scheme=scheme)
        C_ana = analytical_solution(r)
        L1, L2, Linf = compute_error_norms(C_num, C_ana)

        drs.append(r[1] - r[0])
        L1s.append(L1)
        L2s.append(L2)
        Linfs.append(Linf)

    drs = np.array(drs)
    L1s = np.array(L1s)
    L2s = np.array(L2s)
    Linfs = np.array(Linfs)

    def observed_orders(errs):
        """Compute observed convergence order between successive refinements."""
        orders = []
        for k in range(1, len(errs)):
            if errs[k] > 0 and errs[k - 1] > 0:
                p = np.log(errs[k - 1] / errs[k]) / np.log(drs[k - 1] / drs[k])
                orders.append(p)
            else:
                orders.append(np.nan)
        return np.array(orders)

    return {
        "N": grid_sizes,
        "dr": drs,
        "L1": L1s,
        "L2": L2s,
        "Linf": Linfs,
        "order_L1": observed_orders(L1s),
        "order_L2": observed_orders(L2s),
        "order_Linf": observed_orders(Linfs),
    }
