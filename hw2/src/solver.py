"""
Finite difference solver for steady-state radial diffusion (1D) in a cylinder.
"""

import numpy as np

def S_MMS(r, t, R=0.5, D_eff=1e-10, k=4e-9):
     """Computes the source term for MMS verification.

    Parameters
    ----------
    r : float
        Radial position [m].
    t : float
        Time [s].
    R : float
        Pillar radius [m].
    D_eff : float
        Effective diffusion coefficient [m2/s].
    k : float
        Reaction constant [1/s]

    Returns
    -------
    S : float
        Source term for MMS.
    """
     S = np.exp(-t) * ((1 - ((r / R)**2)) * (k - 1) 
                       + (4 * D_eff) / (R**2)) 
     return S

def solve_diffusion(N, T_max=1.0, t_steps=200, D_eff=1e-10, R=0.5, Ce=0.0, k=4e-9):
    """Solve the steady-state radial diffusion equation using finite differences.

    dC/dt = D_eff * (d2C/dr2 + (1/r)*dC/dr) - kC

    with BCs:
        dC/dr = 0  at r=0  (symmetry)
        C = Ce     at r=R  (Dirichlet)

    Parameters
    ----------
    N : int
        Number of grid points (including boundaries).
    T_max : float
        Model time duration [s].
    t_steps : int
        Number of time steps
    D_eff : float
        Effective diffusion coefficient [m2/s].
    R : float
        Pillar radius [m].
    Ce : float
        External concentration [mol/m3].
    k : float
        Reaction constant [1/s]

    Returns
    -------
    r : ndarray
        Radial grid positions, shape (N,).
    time : ndarray
        Time grid positions, shape (t_steps,).
    C_all : ndarray
        Numerical concentration solution, shape (t_steps, N).
    """
    
    dr = R / (N - 1)
    r = np.linspace(0., R, N)
    time = np.linspace(0, T_max, t_steps)

    dt = T_max / t_steps
    # MMS initial condition at time t = 0
    C = 1 - (r/R)**2
    C_all = np.zeros((t_steps, N))

    A = np.zeros((N, N))
    b = np.zeros(N)


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

    
    for n in range(t_steps):
        t = n * dt
        S = S_MMS(r,t)
        b[1:-1] = C[1:-1] + dt * S[1:-1]
        C = np.linalg.solve(A, b)
        C_all[n, :] = C
    return r, time, C_all