"""
Verification test suite for the Euler-Bernoulli beam FEM solver.

Hierarchy follows MEC8211 - Verification de code (Trepanier & Vidal, Hiver 2026):

  1. Symmetry tests
  2. Conservation tests
  3. Galilean invariance - scaling
  4. Analytical accuracy: cantilever + UDL
  5. Analytical accuracy: 3-pt bending exactness (cubic solution)
  6. Analytical accuracy: SS beam + UDL
  7. MMS accuracy: sine wave manufactured solution

Tests 4, 6, 7 check that the FEM solution is within a tolerance of the
analytical/manufactured solution on a fixed mesh. The full convergence order
study and GCI computation are handled in scripts/convergence_study.py.

Run with:  pytest tests/test_beam.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from beam import (
    assemble_K,
    assemble_distributed_load,
    apply_point_load,
    apply_prescribed_displacement,
    solve,
    solve_mr,
)
from cases import ALL_CASES, L as CASES_L

# ---------------------------------------------------------------------------
# Shared non-dimensional parameters
# ---------------------------------------------------------------------------
E = 1.0
I = 1.0
L = 1.0
w = 1.0
P = 1.0

# Fixed mesh used for accuracy tests (fine enough to be well below tolerance)
N_ACCURACY = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_nodal_error(u, x_nodes, v_exact_func):
    """
    L2 norm of nodal displacement error:
        E = sqrt( mean_i |v_h(x_i) - v_exact(x_i)|^2 )
    Displacement DOFs only (even-indexed).
    """
    v_h = u[0::2]
    v_ex = np.array([v_exact_func(xi) for xi in x_nodes])
    return np.sqrt(np.mean((v_h - v_ex) ** 2))


def _l2_interior_error(u, n, L_beam, v_exact_func, n_pts=4):
    """
    L2 error sampled at interior points of each element (not at nodes).

    Hermite elements are superconvergent at nodes for smooth solutions, so
    nodal errors sit at machine precision even on coarse meshes. Sampling
    strictly inside each element reveals the genuine O(h^4) discretization
    error without the superconvergence effect.

    Uses n_pts uniformly spaced points per element, excluding the endpoints.
    """
    Le = L_beam / n
    errors_sq = []
    for e in range(n):
        x_e = e * Le
        for k in range(1, n_pts + 1):
            xi = k / (n_pts + 1)
            x = x_e + xi * Le
            N1 =  1 - 3*xi**2 + 2*xi**3
            N2 =  Le * xi * (1 - xi)**2
            N3 =  3*xi**2 - 2*xi**3
            N4 =  Le * xi**2 * (xi - 1)
            v_h = N1*u[2*e] + N2*u[2*e+1] + N3*u[2*e+2] + N4*u[2*e+3]
            errors_sq.append((v_h - v_exact_func(x))**2)
    return np.sqrt(np.mean(errors_sq))


def _3pt_v(x):
    """SS beam + midspan point load P. Piecewise cubic, symmetric."""
    if x <= L / 2:
        return P / (48*E*I) * x * (3*L**2 - 4*x**2)
    else:
        return P / (48*E*I) * (L - x) * (3*L**2 - 4*(L - x)**2)


# ===========================================================================
# 1. SYMMETRY TESTS
# ===========================================================================

def test_stiffness_symmetry():
    """Global stiffness matrix must satisfy K = K^T at machine precision."""
    K = assemble_K(10, E, I, L)
    np.testing.assert_allclose(K, K.T, atol=1e-14,
                               err_msg="Stiffness matrix is not symmetric")


def test_displacement_symmetry_ss_udl():
    """
    SS beam with UDL: deflection v(x) must equal v(L-x) at all nodes.
    Error must be at machine precision (symmetry test, slide 9).
    """
    n = 10
    K = assemble_K(n, E, I, L)
    f = assemble_distributed_load(n, L, w)
    u, _ = solve(K, f, [0, 2*n], [0.0, 0.0])
    v = u[0::2]
    np.testing.assert_allclose(v, v[::-1], atol=1e-13,
                               err_msg="Displacement not symmetric about midspan")


# ===========================================================================
# 2. CONSERVATION TESTS
# ===========================================================================

def test_reaction_equilibrium_cantilever_udl():
    """
    Cantilever + UDL: clamp reactions must satisfy global force and moment
    equilibrium (sum F = 0, sum M = 0).
    """
    n = 10
    K = assemble_K(n, E, I, L)
    f = assemble_distributed_load(n, L, w)
    _, R = solve(K, f, [0, 1], [0.0, 0.0])

    np.testing.assert_allclose(R[0], -w*L, rtol=1e-12,
                               err_msg="Vertical clamp reaction incorrect")
    np.testing.assert_allclose(R[1], -w*L**2 / 2, rtol=1e-12,
                               err_msg="Moment clamp reaction incorrect")


def test_reaction_equilibrium_ss_udl():
    """
    SS beam + UDL: sum of support reactions must equal total applied load.
    Each support carries half by symmetry.
    """
    n = 10
    K = assemble_K(n, E, I, L)
    f = assemble_distributed_load(n, L, w)
    _, R = solve(K, f, [0, 2*n], [0.0, 0.0])

    np.testing.assert_allclose(R[0] + R[2*n], -w*L, rtol=1e-12,
                               err_msg="Total vertical reaction incorrect")
    np.testing.assert_allclose(R[0], -w*L / 2, rtol=1e-12,
                               err_msg="Left support reaction incorrect")
    np.testing.assert_allclose(R[1], 0.0, atol=1e-12,
                               err_msg="Unexpected moment reaction at left pin")


# ===========================================================================
# 3. GALILEAN INVARIANCE - SCALING
# ===========================================================================

def test_scaling_invariance_cantilever_udl():
    """
    Scale L -> alpha*L and E -> alpha^3*E (preserving EI/L^3): nodal
    displacements must scale by alpha and rotations must be invariant.
    Detects unit and dimensional bugs (scaling test, slide 15).
    """
    alpha = 2.0
    n = 6

    K_ref = assemble_K(n, E, I, L)
    f_ref = assemble_distributed_load(n, L, w)
    u_ref, _ = solve(K_ref, f_ref, [0, 1], [0.0, 0.0])

    E_s = E * alpha**3
    L_s = L * alpha
    K_s = assemble_K(n, E_s, I, L_s)
    f_s = assemble_distributed_load(n, L_s, w)
    u_s, _ = solve(K_s, f_s, [0, 1], [0.0, 0.0])

    np.testing.assert_allclose(u_s[0::2], alpha * u_ref[0::2], rtol=1e-12,
                               err_msg="Displacements do not scale by alpha")
    np.testing.assert_allclose(u_s[1::2], u_ref[1::2], rtol=1e-12,
                               err_msg="Rotations not invariant under scaling")


# ===========================================================================
# 4. ANALYTICAL ACCURACY: PARAMETRIZED OVER ALL CASES
# ===========================================================================

@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c.name)
def test_accuracy(case):
    """
    Fixed mesh (n=N_ACCURACY): interior L2 displacement error must be below
    tolerance for each case defined in src/cases.py. Full convergence order
    study and GCI are in scripts/convergence_study.py.
    """
    K, f, dofs, vals = case.setup(N_ACCURACY)
    u, _ = solve(K, f, dofs, vals)
    err = _l2_interior_error(u, N_ACCURACY, CASES_L, case.v_exact)
    assert err < 1e-5, (
        f"{case.name}: interior L2 error {err:.3e} exceeds tolerance 1e-5"
    )


# ===========================================================================
# 5. ANALYTICAL ACCURACY: 3-PT BENDING EXACTNESS
# ===========================================================================

def test_3pt_bending_midspan_load_exact():
    """
    SS beam + midspan point load: exact solution is piecewise cubic.
    For n even the load lands exactly on a node, so Hermite elements reproduce
    all nodal displacements to machine precision (error < 1e-10).
    Demonstrates FEM exactness for polynomial solutions of degree <= 3.
    """
    for n in [2, 4, 8, 16]:
        K = assemble_K(n, E, I, L)
        f = np.zeros(2 * (n + 1))
        apply_point_load(f, x=L/2, L=L, n=n, force=P)
        u, _ = solve(K, f, [0, 2*n], [0.0, 0.0])
        x_nodes = np.linspace(0, L, n + 1)
        err = _l2_nodal_error(u, x_nodes, _3pt_v)
        assert err < 1e-10, (
            f"n={n}: 3-pt bending exactness failed, L2 error = {err:.3e}"
        )


# ===========================================================================
# 6. PRESCRIBED DISPLACEMENT / ROTATION (PENALTY METHOD)
# ===========================================================================

def test_prescribed_displacement_clamped_clamped():
    """
    Clamped-clamped beam + UDL: clamp at x=0 via direct DOF prescription,
    clamp at x=L (disp=0, rotation=0) via apply_prescribed_displacement.
    Exact midspan deflection is wL^4/(384EI).

    Tests that the penalty method correctly enforces both a displacement and a
    rotation constraint, and that the resulting field matches the known solution.
    """
    n = 16
    K = assemble_K(n, E, I, L)
    f = assemble_distributed_load(n, L, w)
    apply_prescribed_displacement(K, f, x=L, L=L, n=n, disp=0.0, rotation=0.0)
    u, _ = solve(K, f, [0, 1], [0.0, 0.0])

    v_mid = u[n]    # displacement DOF at midspan node (n elements -> node n/2 = n at midspan)
    v_exact_mid = w * L**4 / (384 * E * I)
    assert abs(v_mid - v_exact_mid) < 1e-4, (
        f"Clamped-clamped midspan deflection: got {v_mid:.6e}, expected {v_exact_mid:.6e}"
    )


def test_prescribed_displacement_off_node_enforced():
    """
    SS beam with n=3 elements (nodes at x=0, L/3, 2L/3, L): prescribe v=0
    at the off-node position x=L/2 using apply_prescribed_displacement.

    Verifies that the interpolated displacement at the constrained point is
    driven to near zero, confirming the penalty method is active at off-node
    positions.
    """
    n = 3
    K = assemble_K(n, E, I, L)
    f = assemble_distributed_load(n, L, w)
    apply_prescribed_displacement(K, f, x=L/2, L=L, n=n, disp=0.0)
    u, _ = solve(K, f, [0, 2*n], [0.0, 0.0])

    # Reconstruct v_h(L/2) from the Hermite shape functions
    Le  = L / n
    e   = min(int((L/2) / Le), n - 1)
    xi  = (L/2 - e * Le) / Le
    N1  =  1 - 3*xi**2 + 2*xi**3
    N2  =  Le * xi * (1 - xi)**2
    N3  =  3*xi**2 - 2*xi**3
    N4  =  Le * xi**2 * (xi - 1)
    v_h = N1*u[2*e] + N2*u[2*e+1] + N3*u[2*e+2] + N4*u[2*e+3]

    assert abs(v_h) < 1e-8, (
        f"Off-node penalty constraint not enforced: v(L/2) = {v_h:.3e}"
    )


# ===========================================================================
# 7. MOONEY-RIVLIN NONLINEAR SOLVER
# ===========================================================================

# Physical parameters shared across MR tests
_C10 = 2.6643e5    # Pa
_C01 = 6.6007e5    # Pa
_E0  = 6.0 * (_C10 + _C01)   # linearised modulus [Pa]
_D   = 5e-3        # rod diameter [m]
_I   = np.pi * _D**4 / 64


def test_solve_mr_small_strain_matches_linear():
    """
    At vanishingly small prescribed displacement the MR model must recover the
    linear result: fibre strains -> 0  =>  sigma_MR -> E0 * eps.
    Reaction force must agree with the linear solve to within 0.1 %.
    """
    n   = 10
    Lb  = 0.10      # beam length [m]
    mid = n // 2
    delta = 1e-9    # ~0 strain: fibre strain << 1e-6

    f = np.zeros(2 * (n + 1))
    K = assemble_K(n, _E0, _I, Lb)
    _, R_lin = solve(K, f, [0, 2*n, 2*mid], [0.0, 0.0, delta])
    _, R_mr  = solve_mr(n, _D, Lb, _C10, _C01, f,
                        [0, 2*n, 2*mid], [0.0, 0.0, delta])

    np.testing.assert_allclose(
        R_mr[2*mid], R_lin[2*mid], rtol=1e-3,
        err_msg="MR reaction does not match linear at near-zero strain",
    )


def test_solve_mr_symmetry():
    """
    SS beam with symmetric MR material and symmetric midspan displacement:
    the nodal displacement field must be symmetric about midspan.
    """
    n   = 10
    Lb  = 0.10
    mid = n // 2
    delta = 2e-3    # moderate deflection

    f = np.zeros(2 * (n + 1))
    u, _ = solve_mr(n, _D, Lb, _C10, _C01, f,
                    [0, 2*n, 2*mid], [0.0, 0.0, delta])
    v = u[0::2]
    np.testing.assert_allclose(
        v, v[::-1], atol=1e-10,
        err_msg="MR displacement field not symmetric about midspan",
    )


def test_solve_mr_reaction_equilibrium():
    """
    SS beam under UDL: sum of support reactions must equal total applied load,
    verifying global equilibrium of the MR nonlinear solve.
    """
    n    = 10
    Lb   = 0.10
    w_load = -50.0   # N/m  (downward)

    f = assemble_distributed_load(n, Lb, w_load)
    _, R = solve_mr(n, _D, Lb, _C10, _C01, f, [0, 2*n], [0.0, 0.0])

    np.testing.assert_allclose(
        R[0] + R[2*n], -w_load * Lb, rtol=1e-6,
        err_msg="MR support reactions do not sum to total applied load",
    )
