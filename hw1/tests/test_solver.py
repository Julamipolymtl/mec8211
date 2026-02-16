"""Tests for the analytical solution and finite difference solver."""

import numpy as np
import pytest

from analytical import analytical_solution

S = 2e-8
D_EFF = 1e-10
R = 0.5
CE = 20.0


# --- Analytical solution tests ---

class TestAnalytical:
    def test_dirichlet_bc(self):
        """C(R) must equal Ce."""
        C = analytical_solution(np.array([R]), S=S, D_eff=D_EFF, R=R, Ce=CE)
        assert C[0] == pytest.approx(CE)

    def test_center_value(self):
        """C(0) = Ce - S*R^2 / (4*D_eff)."""
        expected = CE - S * R**2 / (4 * D_EFF)
        C = analytical_solution(np.array([0.0]), S=S, D_eff=D_EFF, R=R, Ce=CE)
        assert C[0] == pytest.approx(expected)

    def test_neumann_bc(self):
        """dC/dr ~ 0 at r = 0 (symmetry)."""
        dr = 1e-8
        C = analytical_solution(np.array([0.0, dr]), S=S, D_eff=D_EFF, R=R, Ce=CE)
        dCdr = (C[1] - C[0]) / dr
        assert dCdr == pytest.approx(0.0, abs=1e-3)
