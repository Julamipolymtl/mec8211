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
        assert dCdr == pytest.approx(0.0, abs=1e-6)


# --- Solver tests ---

class TestSolverBoundaryConditions:
    @pytest.mark.parametrize("scheme", ["forward", "central"])
    def test_dirichlet_bc(self, scheme):
        """C(R) must equal Ce for both schemes."""
        r, C = solve_diffusion(20, scheme=scheme)
        assert C[-1] == pytest.approx(CE)

    @pytest.mark.parametrize("scheme", ["forward", "central"])
    def test_neumann_bc(self, scheme):
        """dC/dr ~ 0 at r = 0 using 2nd-order forward difference."""
        r, C = solve_diffusion(50, scheme=scheme)
        dr = r[1] - r[0]
        dCdr = (-3 * C[0] + 4 * C[1] - C[2]) / (2 * dr)
        assert dCdr == pytest.approx(0.0, abs=1e-12)


class TestZeroSource:
    """With S=0 the exact solution is C(r) = Ce everywhere."""

    @pytest.mark.parametrize("scheme", ["forward", "central"])
    def test_uniform_solution(self, scheme):
        """Solver must return C = Ce at every node when there is no source."""
        _, C = solve_diffusion(50, scheme=scheme, S=0.0)
        np.testing.assert_allclose(C, CE, atol=1e-12)


class TestSolverMonotonicity:
    @pytest.mark.parametrize("scheme", ["forward", "central"])
    def test_monotonically_increasing(self, scheme):
        """With a positive source term, concentration should increase from center to edge."""
        _, C = solve_diffusion(50, scheme=scheme)
        assert np.all(np.diff(C) >= 0)


class TestConservation:
    """Flux balance: total source in the domain must equal diffusive flux out at r=R."""

    @pytest.mark.parametrize("scheme", ["forward", "central"])
    def test_flux_balance(self, scheme):
        """∫₀ᴿ S·2πr·dr = S·π·R² must equal -D_eff·(dC/dr)|_{r=R}·2πR."""
        N = 1000
        r, C = solve_diffusion(N, scheme=scheme, S=S, D_eff=D_EFF, R=R, Ce=CE)
        dr = r[1] - r[0]

        # Total source integrated over cylinder cross-section (per unit length)
        total_source = S * np.pi * R**2

        # Diffusive flux into the domain at r=R (S is a sink, so flux is inward)
        # Balance: D_eff * (dC/dr)|_{r=R} * 2πR = S * πR²
        # Use 2nd order backward difference for dC/dr at r=R:
        dCdr_R = (3 * C[-1] - 4 * C[-2] + C[-3]) / (2 * dr)
        flux_in = D_EFF * dCdr_R * 2 * np.pi * R
        assert flux_in == pytest.approx(total_source, rel=1e-3)