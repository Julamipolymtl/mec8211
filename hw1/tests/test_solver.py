"""Tests for the analytical solution and finite difference solver."""

import numpy as np
import pytest

from analytical import analytical_solution
from solver import solve_diffusion

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


# --- Solver tests ---

class TestSolverBoundaryConditions:
    @pytest.mark.parametrize("scheme", ["forward", "central"])
    def test_dirichlet_bc(self, scheme):
        """C(R) must equal Ce for both schemes."""
        r, C = solve_diffusion(20, scheme=scheme)
        assert C[-1] == pytest.approx(CE)

    def test_neumann_bc_forward(self):
        """dC/dr ~ 0 at r = 0 using 1st-order forward difference."""
        r, C = solve_diffusion(50, scheme="forward")
        dCdr = (C[1] - C[0]) / (r[1] - r[0])
        assert dCdr == pytest.approx(0.0, abs=1e-2)

    def test_neumann_bc_central(self):
        """dC/dr ~ 0 at r = 0 using 2nd-order forward difference."""
        r, C = solve_diffusion(50, scheme="central")
        dr = r[1] - r[0]
        dCdr = (-3 * C[0] + 4 * C[1] - C[2]) / (2 * dr)
        assert dCdr == pytest.approx(0.0, abs=1e-2)


class TestSolverMonotonicity:
    @pytest.mark.parametrize("scheme", ["forward", "central"])
    def test_monotonically_increasing(self, scheme):
        """With a positive source term, concentration should increase from center to edge."""
        _, C = solve_diffusion(50, scheme=scheme)
        assert np.all(np.diff(C) >= 0)
