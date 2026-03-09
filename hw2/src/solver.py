"""
Finite difference solver for transient radial diffusion (1D) in a cylinder.
"""

import json
from dataclasses import dataclass, asdict, fields

import numpy as np

from mms import DEFAULT_MMS, derive_mms


@dataclass
class DiffusionParams:
    """Parameters for the transient radial diffusion solver.

    Physical parameters
    -------------------
    D_eff : float
        Effective diffusion coefficient [m2/s].
    R : float
        Pillar radius [m].
    Ce : float
        External (boundary) concentration [mol/m3].
    k : float
        First-order reaction constant [1/s].

    Numerical parameters
    --------------------
    N_r : int
        Number of radial grid points (including boundaries).
        Also the starting (coarsest) grid size for spatial convergence sweeps.
    N_t : int
        Number of time steps (including t=0).
        Also the starting (coarsest) grid size for temporal convergence sweeps.
    t_max : float
        Simulation end time [s]. Set to 1.0 for MMS, 4e9 for physical.
    mms : bool
        If True, adds the derived MMS source term to the RHS at each time step.
        Set to False (default) for the physical problem.
    mms_solution : str
        SymPy-parseable expression for the manufactured solution C(r, t).
        May reference ``r``, ``t``, ``R``, ``D_eff``, ``k``, ``Ce``.
        Must satisfy dC/dr=0 at r=0 and C(R,t)=0.
        Example: ``"exp(-t) * (1 - (r/R)**4)"``.

    Convergence study parameters
    ----------------------------
    run_convergence : bool
        If False, skip convergence studies (useful for parameter sweeps).
    N_t_conv : int
        Fixed number of time steps used during spatial convergence study.
        Should be large enough that temporal error does not pollute spatial error.
    N_r_conv : int
        Fixed number of radial points used during temporal convergence study.
        Should be large enough that spatial error does not pollute temporal error.
    num_refinements : int
        Number of grid doublings in each convergence study.

    Run identification
    ------------------
    run_name : str
        Label for this simulation. Figures are saved to ``results/{run_name}/``
        and raw data to ``data/{run_name}/``. Use descriptive names for
        parameter sweeps (e.g. ``"D0.1-k0.005"`` or ``"big-pillar"``).
    """
    # Physical
    D_eff:           float = 1e-10
    R:               float = 0.5
    Ce:              float = 0.0
    k:               float = 4e-9
    # Numerical
    N_r:             int   = 11
    N_t:             int   = 200
    t_max:           float = 1.0
    mms:             bool  = False
    mms_solution:    str   = DEFAULT_MMS
    # Convergence
    run_convergence: bool  = True
    N_r_conv:        int   = 500
    N_t_conv:        int   = 5000
    num_refinements: int   = 4
    # Run identification
    run_name:        str   = "default"

    def mms_functions(self):
        """Return (C_fn, S_fn) derived from mms_solution via mms.derive_mms."""
        return derive_mms(self.mms_solution, self.D_eff, self.R, self.k, self.Ce)

    def to_json(self, path: str) -> None:
        """Save parameters to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DiffusionParams":
        """Load parameters from a JSON file, ignoring unrecognised keys."""
        known = {f.name for f in fields(cls)}
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in known})


def solve_diffusion(params: DiffusionParams, C_0=0.0):
    """Solve the transient radial diffusion equation using implicit Euler (BDF1).

    Spatial discretisation: 2nd-order central differences for both d^2C/dr^2
    and (1/r)*dC/dr.  Time integration: backward Euler (BDF1, 1st-order).

    dC/dt = D_eff * (d^2C/dr^2 + (1/r)*dC/dr) - k*C  [+ S_mms if params.mms]

    Boundary conditions:
        dC/dr = 0  at r=0  (symmetry, 2nd-order one-sided Neumann)
        C = Ce     at r=R  (Dirichlet)

    Parameters
    ----------
    params : DiffusionParams
        All physical and numerical parameters.
    C_0 : float or ndarray of shape (N_r,)
        Initial concentration [mol/m3]. A scalar broadcasts to all nodes;
        pass an array for a spatially-varying IC (e.g. C_fn(r, 0) where
        C_fn comes from params.mms_functions()).

    Returns
    -------
    r : ndarray
        Radial grid positions, shape (N_r,).
    t : ndarray
        Time grid positions, shape (N_t,).
    C : ndarray
        Numerical concentration solution, shape (N_t, N_r).
    """
    D_eff, R, Ce, k = params.D_eff, params.R, params.Ce, params.k
    N_r, N_t, t_max = params.N_r, params.N_t, params.t_max

    dr = R / (N_r - 1)
    r  = np.linspace(0., R, N_r)
    dt = t_max / (N_t - 1)
    t  = np.linspace(0, t_max, N_t)

    A = np.zeros((N_r, N_r))
    b = np.zeros(N_r)

    # --- Interior domain (assembled once, A is time-independent) ---
    _dr2 = 1.0 / dr**2
    for i in range(1, N_r - 1):
        r_i = r[i]

        # BDF1: C_i^{n+1} - dt*(D_eff*Lap - k)*C_i^{n+1} = C_i^n + dt*S^{n+1}
        coeff_im1 = -dt * D_eff * _dr2 + dt * D_eff / (2.0 * r_i * dr)
        coeff_i   =  dt * D_eff * (2.0 * _dr2) + dt * k
        coeff_ip1 = -dt * D_eff * _dr2 - dt * D_eff / (2.0 * r_i * dr)

        A[i, i - 1] = coeff_im1
        A[i, i]     = coeff_i + 1.0   # +1 from identity (time derivative)
        A[i, i + 1] = coeff_ip1

    # --- Neumann BC @ r=0: dC/dr = 0 (2nd-order one-sided) ---
    A[0, 0] = -3.0
    A[0, 1] =  4.0
    A[0, 2] = -1.0
    b[0]    =  0.0

    # --- Dirichlet BC @ r=R: C = Ce ---
    A[-1, -1] = 1.0
    b[-1]     = Ce

    # --- MMS functions (derived once before the time loop) ---
    C_fn, S_fn = params.mms_functions() if params.mms else (None, None)

    # --- Time integration ---
    C = np.zeros((N_t, N_r))
    C_t = np.zeros(N_r)
    C_t[:] = C_0
    C[0, :] = C_t.copy()

    for i, t_i in enumerate(t[1:], start=1):
        b[1:-1] = C_t[1:-1]
        if S_fn is not None:
            b[1:-1] += dt * S_fn(r, t_i)[1:-1]
            b[-1] = C_fn(R, t_i)   # time-varying Dirichlet BC from MMS
        C_t      = np.linalg.solve(A, b)
        C[i, :] = C_t.copy()

    return r, t, C
