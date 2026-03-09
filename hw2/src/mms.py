"""
Manufactured Method of Solution (MMS) utilities.

Provides symbolic derivation of the manufactured concentration field C_mms
and its corresponding source term S_mms from a user-supplied expression.

The source term is derived automatically from the PDE residual:

    S = dC/dt - D_eff * (d^2C/dr^2 + (1/r) * dC/dr) + k * C
"""

import sympy as sp

DEFAULT_MMS = "exp(-t) * (1 - (r/R)**4)"


def derive_mms(mms_expr, D_eff, R, k, Ce=0.0):
    """Parse a manufactured solution expression and derive C_fn and S_fn.

    Parameters
    ----------
    mms_expr : str
        SymPy-parseable expression for C(r, t). May reference the symbols
        ``r``, ``t`` and the physical parameters ``R``, ``D_eff``, ``k``,
        ``Ce`` by name (substituted with their numerical values).
    D_eff : float
        Effective diffusion coefficient [m2/s].
    R : float
        Pillar radius [m].
    k : float
        First-order reaction constant [1/s].
    Ce : float
        External (boundary) concentration [mol/m3].

    Returns
    -------
    C_fn : callable(r, t) -> float or ndarray
        Manufactured concentration field.
    S_fn : callable(r, t) -> float or ndarray
        Corresponding source term to be added to the PDE RHS.
    """
    r_sym, t_sym = sp.symbols("r t")

    local_ns = {
        "r": r_sym, "t": t_sym,
        "R": R, "D_eff": D_eff, "k": k, "Ce": Ce,
        "exp": sp.exp, "sin": sp.sin, "cos": sp.cos, "pi": sp.pi,
    }
    C_sym = sp.sympify(mms_expr, locals=local_ns)

    # PDE residual gives the source term needed to make C_sym an exact solution
    S_sym = (
        sp.diff(C_sym, t_sym)
        - D_eff * (sp.diff(C_sym, r_sym, 2) + sp.diff(C_sym, r_sym) / r_sym)
        + k * C_sym
    )
    S_sym = sp.simplify(S_sym)

    C_fn = sp.lambdify((r_sym, t_sym), C_sym, modules="numpy")
    S_fn = sp.lambdify((r_sym, t_sym), S_sym, modules="numpy")
    return C_fn, S_fn
