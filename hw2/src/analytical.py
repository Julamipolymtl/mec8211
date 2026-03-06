"""
Analytical solution for salt diffusion in a cylindrical concrete pillar with uniform source term and dirichlet boundary conditions at r=R.
"""

import numpy as np


def manufactured_solution(r, t=1.0, R=0.5):
    """Compute the analytical solution for the steady-state radial diffusion problem.

    C(r) = (1/4)*(S/D_eff)*R2*(r²/R2 - 1) + Ce

    Parameters
    ----------
    r : array_like
        Radial positions [m].
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
    return np.exp(-t)*(1 - ((r/R)**2))