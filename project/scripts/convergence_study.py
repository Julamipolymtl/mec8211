"""
Convergence study and GCI computation for the Euler-Bernoulli beam FEM solver.

For each test case the script:
  1. Solves on a sequence of uniformly refined meshes (refinement ratio r = 2).
  2. Computes the observed convergence order p_hat using the pairwise formula
     (slide 19, MEC8211 - Verification de code).
  3. Computes the GCI (Grid Convergence Index) for each consecutive mesh triplet
     using the interior L2 error as the system response quantity (SRQ).
  4. Reports the asymptotic convergence check: (E3-E2)/((E2-E1)*r^p) ~ 1.

Note on nodal superconvergence
------------------------------
Hermite cubic elements give essentially machine-precision nodal displacements
even on coarse meshes for smooth solutions (the Galerkin weak form is satisfied
exactly at nodes for polynomial right-hand sides). The genuine O(h^4)
discretization error is therefore measured at interior element points, not at
nodes. The GCI is accordingly applied to the interior L2 error, which
represents the accuracy of the full displacement field between nodes.

Test cases are defined in src/cases.py and shared with the test suite.

Run with:  python scripts/convergence_study.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from beam import solve
from cases import ALL_CASES, L

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

FS     = 1.25    # GCI factor of safety (Roache 1994, recommended value)
R      = 2.0     # mesh refinement ratio between consecutive meshes
N_LIST = [4, 8, 16, 32, 64]


# ---------------------------------------------------------------------------
# Interior L2 error
# ---------------------------------------------------------------------------

def l2_interior_error(u, n, v_exact_func, n_pts=4):
    """
    L2 displacement error sampled at interior points of each element (not at
    nodes). Uses n_pts uniformly spaced points per element, excluding endpoints.
    """
    Le = L / n
    errors_sq = []
    for e in range(n):
        x_e = e * Le
        for k in range(1, n_pts + 1):
            xi  = k / (n_pts + 1)
            x   = x_e + xi * Le
            N1  =  1 - 3*xi**2 + 2*xi**3
            N2  =  Le * xi * (1 - xi)**2
            N3  =  3*xi**2 - 2*xi**3
            N4  =  Le * xi**2 * (xi - 1)
            v_h = N1*u[2*e] + N2*u[2*e+1] + N3*u[2*e+2] + N4*u[2*e+3]
            errors_sq.append((v_h - v_exact_func(x))**2)
    return float(np.sqrt(np.mean(errors_sq)))


# ---------------------------------------------------------------------------
# Convergence order and GCI
# ---------------------------------------------------------------------------

def pairwise_orders(values):
    """
    Observed convergence order for each consecutive pair (coarse -> fine).
    values must be ordered from coarsest to finest mesh.
        p_hat = log(v_coarse / v_fine) / log(R)
    Returns an array of length len(values)-1.
    """
    orders = []
    for i in range(len(values) - 1):
        if values[i] > 0 and values[i+1] > 0:
            orders.append(np.log(values[i] / values[i+1]) / np.log(R))
        else:
            orders.append(np.nan)
    return np.array(orders)


def gci(e_fine, e_medium, p):
    """
    GCI for the fine-mesh L2 error (Roache 1994):
        GCI = Fs * |e_fine - e_medium| / (e_fine * (r^p - 1))
    A value close to Fs*100% (125% with Fs=1.25) indicates perfect O(h^p)
    convergence in the asymptotic regime.
    """
    return FS * abs(e_fine - e_medium) / (e_fine * (R**p - 1))


def asymptotic_ratio(e_coarse, e_medium, e_fine, p):
    """
    Asymptotic convergence check (Richardson extrapolation consistency):
        ratio = (e_coarse - e_medium) / ((e_medium - e_fine) * r^p)
    Should equal 1.0 when the solution is in the asymptotic regime.
    """
    return (e_coarse - e_medium) / ((e_medium - e_fine) * R**p)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def section(title):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def print_error_table(n_list, errors, orders):
    h_list = [L / n for n in n_list]
    print(f"\n  {'n':>6}  {'h':>10}  {'L2 error (interior)':>20}  {'p_hat':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*20}  {'-'*8}")
    for i, (n, h, e) in enumerate(zip(n_list, h_list, errors)):
        p_str = f"{orders[i-1]:8.3f}" if i > 0 else "       -"
        print(f"  {n:>6}  {h:>10.5f}  {e:>20.6e}  {p_str}")


def print_gci_table(n_list, errors, orders):
    print(f"\n  {'triplet':>14}  {'p_hat':>8}  {'GCI_fine (%)':>14}  {'asymp. ratio':>14}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*14}  {'-'*14}")
    for i in range(2, len(n_list)):
        e_c = errors[i-2]
        e_m = errors[i-1]
        e_f = errors[i]
        p   = orders[i-1]
        if np.isnan(p) or R**p <= 1:
            continue
        g  = gci(e_f, e_m, p) * 100
        ar = asymptotic_ratio(e_c, e_m, e_f, p)
        triplet = f"n={n_list[i-2]}/{n_list[i-1]}/{n_list[i]}"
        print(f"  {triplet:>14}  {p:>8.3f}  {g:>13.3f}%  {ar:>14.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Convergence study - Euler-Bernoulli beam FEM solver")
    print(f"Meshes : {N_LIST}  (refinement ratio r={int(R)})")
    print(f"GCI Fs : {FS}")

    for case in ALL_CASES:
        section(case.description)

        errors = []
        for n in N_LIST:
            K, f, dofs, vals = case.setup(n)
            u, _ = solve(K, f, dofs, vals)
            errors.append(l2_interior_error(u, n, case.v_exact))

        orders = pairwise_orders(errors)
        print_error_table(N_LIST, errors, orders)
        print_gci_table(N_LIST, errors, orders)

    print()
    print("Done.")
