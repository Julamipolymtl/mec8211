"""
Convergence analysis: error norms and observed convergence orders.
"""

import numpy as np

from solver import DiffusionParams, solve_diffusion


def compute_error_norms(C_numerical, C_manufactured):
    """Compute L1, L2, and L-infinity error norms over the full space-time domain.

    Parameters
    ----------
    C_numerical : ndarray, shape (N_t, N_r)
        Numerical solution.
    C_manufactured : ndarray, shape (N_t, N_r)
        Manufactured solution evaluated at the same grid points.

    Returns
    -------
    L1, L2, Linf : float
        Error norms.
    """
    error = np.abs(C_numerical - C_manufactured)
    N_t, N_r = error.shape
    L1   = np.sum(error) / (N_t * N_r)
    L2   = np.sqrt(np.sum(error**2) / (N_t * N_r))
    Linf = np.max(error)
    return L1, L2, Linf


def convergence_study_spatial(base_params):
    """Spatial convergence study: refine the radial grid, keep N_t fixed.

    The coarsest grid starts at ``base_params.N_r`` and is doubled
    ``base_params.num_refinements`` times. The time grid is held fixed at
    ``base_params.N_t_conv`` throughout.

    Parameters
    ----------
    base_params : DiffusionParams
        Provides all parameters. Physical params (D_eff, R, k) and convergence
        settings (N_t_conv, num_refinements, t_max) are used directly.
        Ce and mms are taken from base_params unchanged.

    Returns
    -------
    results : dict
        Keys: "N", "dr", "L1", "L2", "Linf", "order_L1", "order_L2", "order_Linf".
    """
    if not base_params.mms:
        raise ValueError("convergence_study_spatial requires mms=True in base_params.")
    N_t            = base_params.N_t_conv
    t_max          = base_params.t_max
    num_refinements = base_params.num_refinements

    drs   = np.empty(num_refinements)
    L1s   = np.empty(num_refinements)
    L2s   = np.empty(num_refinements)
    Linfs = np.empty(num_refinements)

    # Start from base_params.N_r, halve dr at each level: N -> 2*(N-1)+1
    grid_sizes = [(base_params.N_r - 1) * (2**i) + 1 for i in range(num_refinements)]
    C_fn, _ = base_params.mms_functions()

    for i, N in enumerate(grid_sizes):
        params = DiffusionParams(D_eff=base_params.D_eff, R=base_params.R,
                                 k=base_params.k, Ce=base_params.Ce,
                                 N_r=N, N_t=N_t, t_max=t_max,
                                 mms=base_params.mms,
                                 mms_solution=base_params.mms_solution)
        r = np.linspace(0, params.R, N)
        C_0 = C_fn(r, 0.0)
        r, time, C_num = solve_diffusion(params, C_0=C_0)
        C_ana = C_fn(r[np.newaxis, :], time[:, np.newaxis])
        L1, L2, Linf = compute_error_norms(C_num, C_ana)

        drs[i]   = r[1] - r[0]
        L1s[i]   = L1
        L2s[i]   = L2
        Linfs[i] = Linf

    order_L1   = np.log(L1s[:-1]   / L1s[1:])   / np.log(drs[:-1] / drs[1:])
    order_L2   = np.log(L2s[:-1]   / L2s[1:])   / np.log(drs[:-1] / drs[1:])
    order_Linf = np.log(Linfs[:-1] / Linfs[1:]) / np.log(drs[:-1] / drs[1:])

    return {
        "N":          grid_sizes,
        "dr":         drs,
        "L1":         L1s,
        "L2":         L2s,
        "Linf":       Linfs,
        "order_L1":   order_L1,
        "order_L2":   order_L2,
        "order_Linf": order_Linf,
    }


def convergence_study_temporal(base_params):
    """Temporal convergence study: refine the time grid, keep N_r fixed.

    The coarsest grid starts at ``base_params.N_t`` and is doubled
    ``base_params.num_refinements`` times. The radial grid is held fixed at
    ``base_params.N_r_conv`` throughout.

    Parameters
    ----------
    base_params : DiffusionParams
        Provides all parameters. Physical params (D_eff, R, k) and convergence
        settings (N_r_conv, num_refinements, t_max) are used directly.
        Ce and mms are taken from base_params unchanged.

    Returns
    -------
    results : dict
        Keys: "N", "dr", "L1", "L2", "Linf", "order_L1", "order_L2", "order_Linf".
        "dr" holds dt values for this study.
    """
    if not base_params.mms:
        raise ValueError("convergence_study_temporal requires mms=True in base_params.")
    N_r            = base_params.N_r_conv
    t_max          = base_params.t_max
    num_refinements = base_params.num_refinements

    dts   = np.empty(num_refinements)
    L1s   = np.empty(num_refinements)
    L2s   = np.empty(num_refinements)
    Linfs = np.empty(num_refinements)

    # Start from base_params.N_t, double the step count at each level
    grid_sizes = [base_params.N_t * (2**i) for i in range(num_refinements)]
    C_fn, _ = base_params.mms_functions()

    for i, N in enumerate(grid_sizes):
        params = DiffusionParams(D_eff=base_params.D_eff, R=base_params.R,
                                 k=base_params.k, Ce=base_params.Ce,
                                 N_r=N_r, N_t=N, t_max=t_max,
                                 mms=base_params.mms,
                                 mms_solution=base_params.mms_solution)
        r = np.linspace(0, params.R, N_r)
        C_0 = C_fn(r, 0.0)
        r, time, C_num = solve_diffusion(params, C_0=C_0)
        C_ana = C_fn(r[np.newaxis, :], time[:, np.newaxis])
        L1, L2, Linf = compute_error_norms(C_num, C_ana)

        dts[i]   = time[1] - time[0]
        L1s[i]   = L1
        L2s[i]   = L2
        Linfs[i] = Linf

    order_L1   = np.log(L1s[:-1]   / L1s[1:])   / np.log(dts[:-1] / dts[1:])
    order_L2   = np.log(L2s[:-1]   / L2s[1:])   / np.log(dts[:-1] / dts[1:])
    order_Linf = np.log(Linfs[:-1] / Linfs[1:]) / np.log(dts[:-1] / dts[1:])

    return {
        "N":          grid_sizes,
        "dr":         dts,
        "L1":         L1s,
        "L2":         L2s,
        "Linf":       Linfs,
        "order_L1":   order_L1,
        "order_L2":   order_L2,
        "order_Linf": order_Linf,
    }
