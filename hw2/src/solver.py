"""
Finite difference solver for steady-state radial diffusion (1D) in a cylinder.
"""

import numpy as np

def solve_diffusion(N, T_max=1.0, t_steps=200, S=2e-8, D_eff=1e-10, R=0.5, Ce=0.0):
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

    dt = T_max / t_steps
    C = np.zeros(N)

    A = np.zeros((N, N))
    b = np.zeros(N)

    k = 4 * 10**-9

    # --- Interior domain ---
    _dr2 = 1 / dr**2
    for i in range(1, N - 1):
            r_i = r[i]

            # d2C/dr2 ≈ (C_{i+1} - 2*C_i + C_{i-1}) / dr2
            coeff_im1 = - (D_eff * _dr2 * dt 
                           + (D_eff * dt) / (2 * r_i * dr))     # coefficient of C_{i-1}
            
            coeff_i = 2.0 * (D_eff * _dr2 * dt) + k * dt + 1    # coefficient of C_i

            coeff_ip1 = - (D_eff * _dr2 * dt 
                          - (D_eff * dt) / (2 * r_i * dr))      # coefficient of C_{i+1}

            A[i, i - 1] = coeff_im1
            A[i, i] = coeff_i
            A[i, i + 1] = coeff_ip1
            
        
    # --- Homogenous Neumann BC @ R = 0: dC/dr = 0 ---
    A[0, 0] = -3.0
    A[0, 1] = 4.0
    A[0, 2] = -1.0
    b[0] = 0

    # --- Dirichlet BC @ r = R: C = Ce ---
    A[-1, -1] = 1.0
    b[-1] = Ce

    def S_MMS(r, t):
     S = np.exp(-t) * ((1 - ((r / R)**2)) * (k - 1) 
                       + (4 * D_eff) / (R**2))
     return S
    
    for n in range(t_steps):
        t = n * dt
        S = S_MMS(r,t)
        b[1:-1] = C[1:-1] + dt * S[1:-1]
        C = np.linalg.solve(A, b)
    return r, C