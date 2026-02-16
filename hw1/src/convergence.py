"""
Convergence analysis: error norms and observed convergence orders.
"""

import numpy as np

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