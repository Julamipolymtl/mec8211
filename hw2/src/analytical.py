"""
Analytical solution for salt diffusion in a cylindrical concrete pillar with uniform source term and dirichlet boundary conditions at r=R.
"""

import numpy as np


def manufactured_solution(r, t=1.0, R=0.5):
    """Compute the manufactured solution for the radial diffusion problem.

    C(r, t) = exp(-t)*(1 - ((r/R)**2))

    Parameters
    ----------
    r : float
        Radial position [m].
    t : float
        Time position [s].
    R : float
        Pillar radius [m].

    Returns
    -------
    C : float
        Concentration at position r, time t [mol/m3].
    """
    r = r[np.newaxis, :]   # (1, Nr)
    t = t[:, np.newaxis]   # (Nt, 1)

    return np.exp(-t)*(1 - ((r/R)**2))