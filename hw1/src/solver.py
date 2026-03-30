"""
Finite difference solver for steady-state radial diffusion (1D) in a cylinder.
"""

import numpy as np


def _thomas(lower, main, upper, upper2_row0, rhs):
    """Thomas algorithm for the quasi-tridiagonal system.

    The matrix is tridiagonal except for one extra entry A[0, 2] = upper2_row0
    (from the 2nd-order Neumann BC stencil). np.linalg.solve does not support
    longdouble, so this Thomas solver is used instead.

    Parameters
    ----------
    lower : 1-D array, length n-1
        Sub-diagonal: lower[i] = A[i+1, i].
    main : 1-D array, length n
        Main diagonal: main[i] = A[i, i].
    upper : 1-D array, length n-1
        Super-diagonal: upper[i] = A[i, i+1].
    upper2_row0 : scalar
        Extra entry A[0, 2].
    rhs : 1-D array, length n
        Right-hand side vector.

    Returns
    -------
    x : 1-D array, length n
        Solution vector (same dtype as inputs).
    """
    n = len(rhs)
    d = main.copy()
    u = upper.copy()
    f = rhs.copy()

    # Forward sweep: eliminate sub-diagonal
    for i in range(1, n):
        m = lower[i - 1] / d[i - 1]
        d[i] -= m * u[i - 1]
        f[i] -= m * f[i - 1]
        if i == 1:
            # A[0, 2] is non-zero, so it modifies u[1] = A[1, 2]
            u[1] -= m * upper2_row0

    # Back substitution
    x = np.empty(n, dtype=d.dtype)
    x[n - 1] = f[n - 1] / d[n - 1]
    for i in range(n - 2, 0, -1):
        x[i] = (f[i] - u[i] * x[i + 1]) / d[i]
    # Row 0 has entries at columns 0, 1, and 2 (all still at original values)
    x[0] = (f[0] - u[0] * x[1] - upper2_row0 * x[2]) / d[0]

    return x


def solve_diffusion(N, scheme="forward", S=2e-8, D_eff=1e-10, R=0.5, Ce=20.0,
                    dtype=np.longdouble):
    """Solve the steady-state radial diffusion equation using finite differences.

    D_eff * (d2C/dr2 + (1/r)*dC/dr) = S

    with BCs:
        dC/dr = 0  at r=0  (symmetry)
        C = Ce     at r=R  (Dirichlet)

    Parameters
    ----------
    N : int
        Number of grid points (including boundaries).
    scheme : str
        "forward" for O(Δr) forward difference on dC/dr (Scheme 1),
        "central" for O(Δr2) central difference on dC/dr (Scheme 2).
    S : float
        Source term [mol/m3/s].
    D_eff : float
        Effective diffusion coefficient [m2/s].
    R : float
        Pillar radius [m].
    Ce : float
        External concentration [mol/m3].
    dtype : numpy dtype, optional
        Floating-point type for all arrays. Default is np.longdouble (~18–19
        significant digits on x86-64) so the round-off floor is lower and the
        2nd-order scheme converges further before being swamped by cancellation.
        Use np.float64 to reproduce the standard double-precision behaviour.

    Returns
    -------
    r : ndarray
        Radial grid positions, shape (N,).
    C : ndarray
        Numerical concentration solution, shape (N,).
    """
    S     = dtype(S)
    D_eff = dtype(D_eff)
    R     = dtype(R)
    Ce    = dtype(Ce)

    dr = R / dtype(N - 1)
    r  = np.linspace(dtype(0), R, N, dtype=dtype)

    lower = np.zeros(N - 1, dtype=dtype)
    main  = np.zeros(N,     dtype=dtype)
    upper = np.zeros(N - 1, dtype=dtype)
    b     = np.zeros(N,     dtype=dtype)

    # --- Interior rows ---
    _dr2 = dtype(1) / dr ** 2
    for i in range(1, N - 1):
        r_i = r[i]

        # d2C/dr2 ≈ (C_{i+1} - 2*C_i + C_{i-1}) / dr2
        coeff_im1 = D_eff * _dr2
        coeff_i   = dtype(-2) * D_eff * _dr2
        coeff_ip1 = D_eff * _dr2

        match scheme:
            case "forward":
                # dC/dr ≈ (C_{i+1} - C_i) / dr
                coeff_i   -= D_eff / (r_i * dr)
                coeff_ip1 += D_eff / (r_i * dr)
            case "central":
                # dC/dr ≈ (C_{i+1} - C_{i-1}) / (2*dr)
                coeff_im1 -= D_eff / (r_i * dtype(2) * dr)
                coeff_ip1 += D_eff / (r_i * dtype(2) * dr)

        lower[i - 1] = coeff_im1   # A[i, i-1]
        main[i]      = coeff_i     # A[i, i]
        upper[i]     = coeff_ip1   # A[i, i+1]
        b[i]         = S

    # --- Homogeneous Neumann BC @ r = 0: dC/dr = 0 ---
    # 2nd-order forward difference to keep O(Δr²) accuracy at the boundary:
    main[0]      = dtype(-3)
    upper[0]     = dtype(4)
    upper2_row0  = dtype(-1)   # A[0, 2] — the extra entry handled by _thomas
    b[0]         = dtype(0)

    # --- Dirichlet BC @ r = R: C = Ce ---
    main[N - 1] = dtype(1)
    b[N - 1]    = Ce

    C = _thomas(lower, main, upper, upper2_row0, b)
    return r, C