"""Finite difference solver for steady-state radial diffusion in a cylinder."""

import numpy as np


def solve_diffusion(N, scheme="forward", S=2e-8, D_eff=1e-10, R=0.5, Ce=20.0):
    """Solve the steady-state radial diffusion equation using finite differences.

    D_eff * (d²C/dr² + (1/r)*dC/dr) = S

    with BCs:
        dC/dr = 0  at r=0  (symmetry)
        C = Ce     at r=R  (Dirichlet)

    Parameters
    ----------
    N : int
        Number of grid points (including boundaries).
    scheme : str
        "forward" for O(Δr) forward difference on dC/dr (Scheme 1),
        "central" for O(Δr²) central difference on dC/dr (Scheme 2).
    S : float
        Source term [mol/m³/s].
    D_eff : float
        Effective diffusion coefficient [m²/s].
    R : float
        Pillar radius [m].
    Ce : float
        External concentration [mol/m³].

    Returns
    -------
    r : ndarray
        Radial grid positions, shape (N,).
    C : ndarray
        Numerical concentration solution, shape (N,).
    """
    dr = R / (N - 1)
    r = np.linspace(0.0, R, N)

    A = np.zeros((N, N))
    b = np.zeros(N)

    # --- Node 0: Neumann BC dC/dr = 0 at r=0 ---
    if scheme == "forward":
        # Forward difference: (C_1 - C_0)/dr = 0  =>  C_0 = C_1
        A[0, 0] = -1.0
        A[0, 1] = 1.0
        b[0] = 0.0
    elif scheme == "central":
        # 2nd-order one-sided: (-3C_0 + 4C_1 - C_2)/(2*dr) = 0
        A[0, 0] = -3.0
        A[0, 1] = 4.0
        A[0, 2] = -1.0
        b[0] = 0.0
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}. Use 'forward' or 'central'.")

    # --- Interior nodes: i = 1 .. N-2 ---
    for i in range(1, N - 1):
        r_i = r[i]

        # d²C/dr² ≈ (C_{i+1} - 2*C_i + C_{i-1}) / dr²
        coeff_im1 = D_eff / dr**2          # coefficient of C_{i-1}
        coeff_i = -2.0 * D_eff / dr**2     # coefficient of C_i
        coeff_ip1 = D_eff / dr**2           # coefficient of C_{i+1}

        if scheme == "forward":
            # dC/dr ≈ (C_{i+1} - C_i) / dr
            coeff_i += D_eff / (r_i * dr) * (-1.0)
            coeff_ip1 += D_eff / (r_i * dr) * 1.0
        else:
            # dC/dr ≈ (C_{i+1} - C_{i-1}) / (2*dr)
            coeff_im1 += D_eff / (r_i * 2.0 * dr) * (-1.0)
            coeff_ip1 += D_eff / (r_i * 2.0 * dr) * 1.0

        A[i, i - 1] = coeff_im1
        A[i, i] = coeff_i
        A[i, i + 1] = coeff_ip1
        b[i] = S

    # --- Node N-1: Dirichlet BC C = Ce at r=R ---
    A[N - 1, N - 1] = 1.0
    b[N - 1] = Ce

    C = np.linalg.solve(A, b)
    return r, C
