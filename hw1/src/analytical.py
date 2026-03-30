"""
Analytical solution for salt diffusion in a cylindrical concrete pillar with uniform source term and dirichlet boundary conditions at r=R.
"""

import numpy as np


def analytical_solution(r, S=2e-8, D_eff=1e-10, R=0.5, Ce=20.0):
    """Compute the analytical solution for the steady-state radial diffusion problem.

    C(r) = (1/4)*(S/D_eff)*R2*(r²/R2 - 1) + Ce

    Parameters
    ----------
    r : array_like
        Radial positions [m].  The dtype of r is preserved in the output.
    S : float
        Source term [mol/m3/s].
    D_eff : float
        Effective diffusion coefficient [m2/s].
    R : float
        Pillar radius [m].
    Ce : float
        External concentration (Dirichlet BC at r=R) [mol/m3].

    Returns
    -------
    C : ndarray
        Concentration at each radial position [mol/m3].
    """
    dtype = np.asarray(r).dtype
    S, D_eff, R, Ce = dtype.type(S), dtype.type(D_eff), dtype.type(R), dtype.type(Ce)
    return dtype.type(0.25) * (S / D_eff) * (r * r - R * R) + Ce