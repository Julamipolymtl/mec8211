"""
Shared beam problem definitions for the test suite and convergence study.

Each BeamCase bundles:
  - setup(n)  : returns (K, f, prescribed_dofs, prescribed_values) for n elements
  - v_exact   : callable v_exact(x) giving the reference displacement

Non-dimensional parameters: E = I = L = w = 1.0.
"""

import numpy as np
from beam import assemble_K, assemble_distributed_load, assemble_general_load

E = 1.0
I = 1.0
L = 1.0
w = 1.0


class BeamCase:
    """
    A beam verification case bundling the FEM setup and reference solution.

    Attributes
    ----------
    name        : short identifier (used as pytest parametrize id)
    description : one-line description printed in convergence reports
    v_exact     : callable v_exact(x) -> float
    setup       : callable setup(n) -> (K, f, prescribed_dofs, prescribed_values)
    """

    def __init__(self, name, description, setup_fn, v_exact_fn):
        self.name = name
        self.description = description
        self.v_exact = v_exact_fn
        self._setup_fn = setup_fn

    def setup(self, n):
        """Return (K, f, prescribed_dofs, prescribed_values) for n elements."""
        return self._setup_fn(n)


# ---------------------------------------------------------------------------
# Case A: cantilever + UDL
# ---------------------------------------------------------------------------

def _make_cantilever():
    def v_exact(x):
        return w / (24*E*I) * x**2 * (x**2 - 4*L*x + 6*L**2)

    def setup(n):
        K = assemble_K(n, E, I, L)
        f = assemble_distributed_load(n, L, w)
        return K, f, [0, 1], [0.0, 0.0]

    return BeamCase(
        name="cantilever_udl",
        description="Cantilever + UDL  (clamped at x=0, free at x=L)",
        setup_fn=setup,
        v_exact_fn=v_exact,
    )


# ---------------------------------------------------------------------------
# Case B: simply supported beam + UDL
# ---------------------------------------------------------------------------

def _make_ss_udl():
    def v_exact(x):
        return w / (24*E*I) * x * (L**3 - 2*L*x**2 + x**3)

    def setup(n):
        K = assemble_K(n, E, I, L)
        f = assemble_distributed_load(n, L, w)
        return K, f, [0, 2*n], [0.0, 0.0]

    return BeamCase(
        name="ss_udl",
        description="Simply supported beam + UDL  (pinned at x=0 and x=L)",
        setup_fn=setup,
        v_exact_fn=v_exact,
    )


# ---------------------------------------------------------------------------
# Case C: MMS sine wave
# ---------------------------------------------------------------------------

def _make_mms_sine():
    EI = E * I

    def w_mms(x):
        return EI * (np.pi / L)**4 * np.sin(np.pi * x / L)

    def v_exact(x):
        return np.sin(np.pi * x / L)

    def setup(n):
        K = assemble_K(n, E, I, L)
        f = assemble_general_load(n, L, w_mms)
        return K, f, [0, 2*n], [0.0, 0.0]

    return BeamCase(
        name="mms_sine",
        description="MMS sine wave  v_mms(x) = sin(pi*x/L), pinned at x=0 and x=L",
        setup_fn=setup,
        v_exact_fn=v_exact,
    )


# ---------------------------------------------------------------------------
# Public list of all cases
# ---------------------------------------------------------------------------

CANTILEVER_UDL = _make_cantilever()
SS_UDL        = _make_ss_udl()
MMS_SINE      = _make_mms_sine()

ALL_CASES = [CANTILEVER_UDL, SS_UDL, MMS_SINE]
