"""Unit tests for the diffusion solver."""

import numpy as np
import pytest

from src.analytical import analytical_solution
from src.convergence import compute_error_norms, convergence_study
from src.solver import solve_diffusion


class TestAnalytical:
    def test_boundary_value(self):
        """C(R) should equal Ce."""
        R, Ce = 0.5, 20.0
        assert analytical_solution(R) == pytest.approx(Ce)

    def test_center_value(self):
        """C(0) should be less than Ce (source is positive, concentration builds up less near center)."""
        C0 = analytical_solution(0.0)
        # C(0) = (1/4)*(S/D_eff)*R²*(0 - 1) + Ce = -(1/4)*(2e-8/1e-10)*0.25 + 20
        # = -0.25*200*0.25 + 20 = -12.5 + 20 = 7.5
        assert C0 == pytest.approx(7.5)

    def test_symmetry_gradient(self):
        """dC/dr at r=0 should be zero by symmetry (check with finite diff)."""
        dr = 1e-8
        C_left = analytical_solution(0.0)
        C_right = analytical_solution(dr)
        dCdr = (C_right - C_left) / dr
        assert abs(dCdr) < 1e-4


class TestSolver:
    @pytest.mark.parametrize("scheme", ["forward", "central"])
    def test_dirichlet_bc(self, scheme):
        """Last node should satisfy C = Ce."""
        r, C = solve_diffusion(33, scheme=scheme)
        assert C[-1] == pytest.approx(20.0)

    @pytest.mark.parametrize("scheme", ["forward", "central"])
    def test_neumann_bc(self, scheme):
        """dC/dr ≈ 0 at r=0 (check first two nodes are close)."""
        r, C = solve_diffusion(65, scheme=scheme)
        # For both schemes, C[0] should be very close to the analytical C(0)
        dCdr_approx = (C[1] - C[0]) / (r[1] - r[0])
        assert abs(dCdr_approx) < 0.5  # loose check; exact is 0

    @pytest.mark.parametrize("scheme", ["forward", "central"])
    def test_solution_accuracy(self, scheme):
        """Numerical solution should be close to analytical on a fine grid."""
        N = 257
        r, C_num = solve_diffusion(N, scheme=scheme)
        C_ana = analytical_solution(r)
        _, L2, _ = compute_error_norms(C_num, C_ana)
        assert L2 < 1e-2  # should be much smaller on 257-point grid

    def test_invalid_scheme(self):
        with pytest.raises(ValueError):
            solve_diffusion(10, scheme="invalid")


class TestConvergence:
    def test_forward_scheme_order(self):
        """Scheme 1 (forward) should converge at ~O(Δr), i.e. order ≈ 1."""
        results = convergence_study(scheme="forward", grid_sizes=[33, 65, 129, 257])
        # Check the last observed order for L2
        avg_order = np.mean(results["order_L2"])
        assert 0.8 < avg_order < 1.5, f"Expected ~1, got {avg_order:.2f}"

    def test_central_scheme_order(self):
        """Scheme 2 (central) should converge at ~O(Δr²), i.e. order ≈ 2."""
        results = convergence_study(scheme="central", grid_sizes=[33, 65, 129, 257])
        avg_order = np.mean(results["order_L2"])
        assert 1.5 < avg_order < 2.5, f"Expected ~2, got {avg_order:.2f}"

    def test_errors_decrease(self):
        """Errors should decrease as grid is refined."""
        results = convergence_study(scheme="central", grid_sizes=[17, 33, 65])
        assert results["L2"][0] > results["L2"][1] > results["L2"][2]
