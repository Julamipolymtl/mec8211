"""
Finite difference solver for steady-state radial diffusion (1D) in a cylinder.
"""

import numpy as np

def solve_diffusion(N, scheme="forward", S=2e-8, D_eff=1e-10, R=0.5, Ce=20.0):
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

    Returns
    -------
    r : ndarray
        Radial grid positions, shape (N,).
    C : ndarray
        Numerical concentration solution, shape (N,).
    """
    
    dr = R / (N - 1)
    r = np.linspace(0., R, N)

    A = np.zeros((N, N))
    b = np.zeros(N)

    # --- Interior domain ---
    _dr2 = 1 / dr**2
    for i in range(1, N - 1):
        r_i = r[i]

        # d2C/dr2 ≈ (C_{i+1} - 2*C_i + C_{i-1}) / dr2
        coeff_im1 = D_eff * _dr2          # coefficient of C_{i-1}
        coeff_i = -2.0 * D_eff * _dr2     # coefficient of C_i
        coeff_ip1 = D_eff * _dr2           # coefficient of C_{i+1}

        match scheme:
            case "forward":
                # dC/dr ≈ (C_{i+1} - C_i) / dr
                coeff_i -= D_eff / (r_i * dr)
                coeff_ip1 += D_eff / (r_i * dr)
            case "central":
                # dC/dr ≈ (C_{i+1} - C_{i-1}) / (2*dr)
                coeff_im1 -= D_eff / (r_i * 2.0 * dr)
                coeff_ip1 += D_eff / (r_i * 2.0 * dr)

        A[i, i - 1] = coeff_im1
        A[i, i] = coeff_i
        A[i, i + 1] = coeff_ip1
        b[i] = S
        
    # --- Homogenous Neumann BC @ R = 0: dC/dr = 0 ---
    # 2nd-order foward difference to maintain O(Δr2) accuracy at the boundary for both schemes:
    A[0, 0] = -3.0
    A[0, 1] = 4.0
    A[0, 2] = -1.0
    b[0] = 0.0

    # --- Dirichlet BC @ r = R: C = Ce ---
    A[-1, -1] = 1.0
    b[-1] = Ce

    C = np.linalg.solve(A, b)
    return r, C