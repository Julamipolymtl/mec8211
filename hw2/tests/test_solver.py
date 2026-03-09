"""Tests for the diffusion solver (DiffusionParams + solve_diffusion) and MMS."""

import numpy as np
import pytest

from solver import DiffusionParams, solve_diffusion


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

D_EFF = 1e-10
R     = 0.5
CE    = 20.0
K     = 4e-9


def make_params(**kwargs):
    """Return a DiffusionParams with fast test defaults, overriding with kwargs."""
    defaults = dict(
        D_eff=D_EFF, R=R, Ce=CE, k=K,
        N_r=21, N_t=50, t_max=1e8,
        mms=False, run_convergence=False,
        run_name="test",
    )
    defaults.update(kwargs)
    return DiffusionParams(**defaults)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    def test_dirichlet_at_R(self):
        """C(R, t) must equal Ce for all t > 0 (the IC at t=0 is the raw C_0)."""
        params = make_params()
        r, t, C = solve_diffusion(params)
        np.testing.assert_allclose(C[1:, -1], CE, atol=1e-12)

    def test_neumann_at_r0(self):
        """2nd-order one-sided stencil gives dC/dr = 0 at r=0 at every time step."""
        params = make_params(N_r=51)
        r, t, C = solve_diffusion(params)
        dr = r[1] - r[0]
        dCdr = (-3*C[:, 0] + 4*C[:, 1] - C[:, 2]) / (2*dr)
        np.testing.assert_allclose(dCdr, 0.0, atol=1e-8)

    def test_initial_condition_scalar(self):
        """C[0, :] must match a scalar C_0 at interior nodes."""
        params = make_params()
        r, t, C = solve_diffusion(params, C_0=5.0)
        np.testing.assert_allclose(C[0, :-1], 5.0, atol=1e-12)

    def test_initial_condition_array(self):
        """C[0, :] must match an array C_0 at interior nodes."""
        params = make_params(N_r=21)
        r_grid = np.linspace(0, R, 21)
        C_0 = 1.0 + r_grid**2
        r, t, C = solve_diffusion(params, C_0=C_0)
        np.testing.assert_allclose(C[0, :-1], C_0[:-1], atol=1e-12)


# ---------------------------------------------------------------------------
# Trivial / degenerate cases
# ---------------------------------------------------------------------------

class TestTrivialSolutions:
    def test_uniform_ic_at_Ce_no_reaction(self):
        """Uniform IC = Ce with k=0 is an exact equilibrium: solution stays at Ce."""
        params = make_params(k=0.0)
        r, t, C = solve_diffusion(params, C_0=CE)
        np.testing.assert_allclose(C, CE, atol=1e-10)

    def test_long_time_approaches_Ce(self):
        """With no source and k=0, solution from C_0=0 reaches Ce at long times."""
        # Diffusion time scale: R^2 / D_eff ~ 2.5e9 s; run 1.6x that.
        params = make_params(k=0.0, N_r=21, N_t=300, t_max=4e9)
        r, t, C = solve_diffusion(params, C_0=0.0)
        np.testing.assert_allclose(C[-1, :], CE, atol=0.1)


# ---------------------------------------------------------------------------
# Physical behaviour
# ---------------------------------------------------------------------------

class TestPhysicalBehaviour:
    def test_concentration_increases_toward_boundary(self):
        """At late times, concentration is non-decreasing from r=0 to R."""
        params = make_params(N_r=51, N_t=500, t_max=4e9)
        r, t, C = solve_diffusion(params, C_0=0.0)
        assert np.all(np.diff(C[-1, :]) >= -1e-10)

    def test_concentration_bounded(self):
        """Solution must stay in [0, Ce] when starting from zero IC."""
        params = make_params(N_r=51, N_t=200, t_max=4e9)
        r, t, C = solve_diffusion(params, C_0=0.0)
        assert np.all(C >= -1e-12)
        assert np.all(C <= CE + 1e-12)


# ---------------------------------------------------------------------------
# MMS verification
# ---------------------------------------------------------------------------

class TestMMS:
    """Verify that the MMS source term drives the solver to the manufactured solution."""

    @pytest.fixture
    def mms_params(self):
        return DiffusionParams(
            D_eff=1.0, R=0.5, Ce=0.0, k=4.0,
            N_r=51, N_t=500, t_max=1.0,
            mms=True,
            mms_solution="exp(-t) * (1 - (r/R)**4)",
            run_convergence=False,
            run_name="test_mms",
        )

    def test_ic_matches_manufactured(self, mms_params):
        """C[0, :] must match C_fn(r, 0) when the MMS initial condition is used."""
        C_fn, _ = mms_params.mms_functions()
        r_grid = np.linspace(0, mms_params.R, mms_params.N_r)
        C_0 = C_fn(r_grid, 0.0)
        r, t, C = solve_diffusion(mms_params, C_0=C_0)
        np.testing.assert_allclose(C[0, :], C_0, atol=1e-12)

    def test_mms_error_is_small(self, mms_params):
        """Linf error against the manufactured solution must be small for N_r=51."""
        C_fn, _ = mms_params.mms_functions()
        r_grid = np.linspace(0, mms_params.R, mms_params.N_r)
        C_0 = C_fn(r_grid, 0.0)
        r, t, C = solve_diffusion(mms_params, C_0=C_0)
        C_ana = C_fn(r[np.newaxis, :], t[:, np.newaxis])
        Linf = np.max(np.abs(C - C_ana))
        assert Linf < 1e-3

    def test_mms_finer_grid_reduces_error(self, mms_params):
        """Spatial grid refinement must monotonically reduce the Linf error."""
        from dataclasses import replace

        C_fn, _ = mms_params.mms_functions()
        errors = []
        for N in (21, 41, 81):
            p = replace(mms_params, N_r=N)
            r_grid = np.linspace(0, p.R, N)
            C_0 = C_fn(r_grid, 0.0)
            r, t, C = solve_diffusion(p, C_0=C_0)
            C_ana = C_fn(r[np.newaxis, :], t[:, np.newaxis])
            errors.append(np.max(np.abs(C - C_ana)))

        assert errors[0] > errors[1] > errors[2]

    def test_mms_disabled_no_source(self):
        """With mms=False the source is absent and the error grows large."""
        params = DiffusionParams(
            D_eff=1.0, R=0.5, Ce=0.0, k=4.0,
            N_r=51, N_t=500, t_max=1.0,
            mms=False,
            mms_solution="exp(-t) * (1 - (r/R)**4)",
            run_convergence=False,
            run_name="test_no_mms",
        )
        C_fn, _ = params.mms_functions()
        r_grid = np.linspace(0, params.R, params.N_r)
        C_0 = C_fn(r_grid, 0.0)
        r, t, C = solve_diffusion(params, C_0=C_0)
        C_ana = C_fn(r[np.newaxis, :], t[:, np.newaxis])
        Linf = np.max(np.abs(C - C_ana))
        assert Linf > 1e-2
