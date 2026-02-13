"""Analytical solution for salt diffusion in a cylindrical concrete pillar."""

import numpy as np


def analytical_solution(r, S=2e-8, D_eff=1e-10, R=0.5, Ce=20.0):
    """Compute the analytical solution for the steady-state radial diffusion problem.

    C(r) = (1/4)*(S/D_eff)*R²*(r²/R² - 1) + Ce

    Parameters
    ----------
    r : array_like
        Radial positions [m].
    S : float
        Source term [mol/m³/s].
    D_eff : float
        Effective diffusion coefficient [m²/s].
    R : float
        Pillar radius [m].
    Ce : float
        External concentration (Dirichlet BC at r=R) [mol/m³].

    Returns
    -------
    C : ndarray
        Concentration at each radial position [mol/m³].
    """
    r = np.asarray(r, dtype=float)
    return 0.25 * (S / D_eff) * (r * r - R * R) + Ce
