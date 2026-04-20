"""
Euler-Bernoulli beam FEM solver.

DOF ordering per node: [v, theta]
  v     : transverse displacement [m]
  theta : rotation [rad]

Node i occupies global DOFs [2i, 2i+1].
For n elements there are n+1 nodes and 2*(n+1) total DOFs.
"""

import numpy as np


def element_stiffness(E: float, I: float, Le: float) -> np.ndarray:
    """
    4x4 Euler-Bernoulli element stiffness matrix.

    Parameters
    ----------
    E  : Young's modulus [Pa]
    I  : second moment of area [m^4]
    Le : element length [m]

    Returns
    -------
    Ke : (4, 4) ndarray
    """
    c = E * I / Le**3
    return c * np.array([
        [ 12,      6*Le,   -12,      6*Le  ],
        [  6*Le,   4*Le**2, -6*Le,   2*Le**2],
        [-12,     -6*Le,    12,     -6*Le  ],
        [  6*Le,   2*Le**2, -6*Le,   4*Le**2],
    ])


def element_distributed_load(w: float, Le: float) -> np.ndarray:
    """
    Consistent nodal load vector for a uniform transverse load w [N/m].

    Parameters
    ----------
    w  : distributed load intensity [N/m]  (positive = same direction as +v)
    Le : element length [m]

    Returns
    -------
    fe : (4,) ndarray  [F1, M1, F2, M2]
    """
    return w * np.array([
        Le / 2,
        Le**2 / 12,
        Le / 2,
        -Le**2 / 12,
    ])


def assemble_K(n: int, E: float, I: float, L: float) -> np.ndarray:
    """
    Assemble the global stiffness matrix for a uniform beam.

    Parameters
    ----------
    n : number of elements
    E : Young's modulus [Pa]
    I : second moment of area [m^4]
    L : total beam length [m]

    Returns
    -------
    K : (2*(n+1), 2*(n+1)) ndarray
    """
    Le = L / n
    n_dof = 2 * (n + 1)
    K = np.zeros((n_dof, n_dof))
    for e in range(n):
        Ke = element_stiffness(E, I, Le)
        dofs = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K[di, dj] += Ke[i, j]
    return K


def assemble_distributed_load(n: int, L: float, w: float) -> np.ndarray:
    """
    Assemble the global load vector for a uniform distributed load.

    Parameters
    ----------
    n : number of elements
    L : total beam length [m]
    w : distributed load [N/m]

    Returns
    -------
    f : (2*(n+1),) ndarray
    """
    Le = L / n
    n_dof = 2 * (n + 1)
    f = np.zeros(n_dof)
    for e in range(n):
        fe = element_distributed_load(w, Le)
        dofs = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
        for i, di in enumerate(dofs):
            f[di] += fe[i]
    return f


def solve(
    K: np.ndarray,
    f_ext: np.ndarray,
    prescribed_dofs: list,
    prescribed_values: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply boundary conditions and solve K u = f.

    Prescribed DOFs can be zero (fixed) or non-zero (imposed displacement).
    Reaction forces at prescribed DOFs are recovered after solving.

    Parameters
    ----------
    K                : (n_dof, n_dof) global stiffness matrix
    f_ext            : (n_dof,) external force/moment vector
    prescribed_dofs  : list of DOF indices with prescribed values
    prescribed_values: corresponding prescribed displacement/rotation values

    Returns
    -------
    u : (n_dof,) full displacement vector
    R : (n_dof,) reaction vector  R = K @ u - f_ext
          At free DOFs      : ~= 0 (equilibrium residual)
          At prescribed DOFs: reaction force (load cell reading)
    """
    n_dof = K.shape[0]
    all_dofs = np.arange(n_dof)
    p_dofs = np.array(prescribed_dofs, dtype=int)
    p_vals = np.array(prescribed_values, dtype=float)
    f_dofs = np.setdiff1d(all_dofs, p_dofs)

    # Partition K and f_ext according to free/prescribed DOFs
    K_ff = K[np.ix_(f_dofs, f_dofs)]
    K_fp = K[np.ix_(f_dofs, p_dofs)]

    rhs_f = f_ext[f_dofs] - K_fp @ p_vals
    u_f = np.linalg.solve(K_ff, rhs_f)

    u = np.zeros(n_dof)
    u[f_dofs] = u_f
    u[p_dofs] = p_vals

    # Reaction vector: R = K @ u - f_ext
    #   At free DOFs    : R ~= 0  (equilibrium residual, check for correctness)
    #   At prescribed DOFs: R = reaction force (what a load cell would read)
    R = K @ u - f_ext

    return u, R


def assemble_general_load(n: int, L: float, w_func, n_gauss: int = 5) -> np.ndarray:
    """
    Assemble the global load vector for an arbitrary distributed load w(x).

    Uses Gauss-Legendre quadrature on each element to integrate the load
    against the Hermite shape functions. Reduces exactly to
    assemble_distributed_load for a constant w_func.

    Parameters
    ----------
    n       : number of elements
    L       : total beam length [m]
    w_func  : callable w(x) returning load intensity [N/m] at position x [m]
    n_gauss : number of Gauss-Legendre quadrature points per element (default 5)

    Returns
    -------
    f : (2*(n+1),) ndarray
    """
    Le = L / n
    n_dof = 2 * (n + 1)
    f = np.zeros(n_dof)
    pts, wts = np.polynomial.legendre.leggauss(n_gauss)   # on [-1, 1]
    for e in range(n):
        x_e = e * Le
        fe = np.zeros(4)
        for pt, wt in zip(pts, wts):
            xi = (pt + 1) / 2           # map Gauss point to [0, 1]
            x = x_e + xi * Le           # physical coordinate
            N = np.array([
                1 - 3*xi**2 + 2*xi**3,
                Le * xi * (1 - xi)**2,
                3*xi**2 - 2*xi**3,
                Le * xi**2 * (xi - 1),
            ])
            fe += wt * w_func(x) * N * (Le / 2)
        dofs = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
        for i, di in enumerate(dofs):
            f[di] += fe[i]
    return f


# --- Not used in the current project (kept for reference) ---

def apply_prescribed_displacement(
    K: np.ndarray,
    f: np.ndarray,
    x: float,
    L: float,
    n: int,
    disp: float = None,
    rotation: float = None,
    penalty: float = 1e14,
) -> None:
    """
    Weakly enforce a prescribed displacement and/or rotation at position x (in-place).

    Uses the penalty method: a large spring stiffness is added at x to penalize
    deviations from the target value. Modifies K and f in-place; call solve()
    afterwards without listing x in prescribed_dofs.

    For constraints that fall exactly on a node, the direct DOF prescription
    in solve() is exact and preferred. Use this function for off-node positions
    or when a soft constraint at an arbitrary location is needed.

    Parameters
    ----------
    K        : (2*(n+1), 2*(n+1)) global stiffness matrix  (modified in-place)
    f        : (2*(n+1),) global load vector                (modified in-place)
    x        : position along the beam [m],  0 <= x <= L
    L        : total beam length [m]
    n        : number of elements
    disp     : prescribed transverse displacement [m],  or None to skip
    rotation : prescribed rotation [rad],               or None to skip
    penalty  : penalty stiffness (default 1e14)
    """
    if not (0.0 <= x <= L):
        raise ValueError(f"x={x} is outside the beam [0, {L}]")

    Le = L / n
    e  = min(int(x / Le), n - 1)
    xi = (x - e * Le) / Le

    # Hermite shape functions for displacement
    N1 =  1 - 3*xi**2 + 2*xi**3
    N2 =  Le * xi * (1 - xi)**2
    N3 =  3*xi**2 - 2*xi**3
    N4 =  Le * xi**2 * (xi - 1)
    N  = np.array([N1, N2, N3, N4])

    # Shape function derivatives for rotation
    dN1 = (-6*xi + 6*xi**2) / Le
    dN2 =  1 - 4*xi + 3*xi**2
    dN3 = ( 6*xi - 6*xi**2) / Le
    dN4 = -2*xi + 3*xi**2
    dN  = np.array([dN1, dN2, dN3, dN4])

    dofs = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]

    if disp is not None:
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K[di, dj] += penalty * N[i] * N[j]
            f[di] += penalty * disp * N[i]

    if rotation is not None:
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K[di, dj] += penalty * dN[i] * dN[j]
            f[di] += penalty * rotation * dN[i]


def compute_internal_forces(
    u: np.ndarray,
    E: float,
    I: float,
    L: float,
    n: int,
    n_pts: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate bending moment M(x) and shear force V(x) along the beam.

    Uses the Hermite shape-function derivatives element-by-element:
      M(x) = EI * d^2v/dx^2  = (EI/Le^2) * d^2N/dxi^2 @ u_e
      V(x) = EI * d^3v/dx^3  = (EI/Le^3) * d^3N/dxi^3 @ u_e

    The third derivatives are constant per element (piecewise-constant shear).

    Parameters
    ----------
    u     : (2*(n+1),) global displacement vector from solve()
    E     : Young's modulus [Pa]
    I     : second moment of area [m^4]
    L     : total beam length [m]
    n     : number of elements
    n_pts : evaluation points per element (default 10)

    Returns
    -------
    x : (n*n_pts,) positions [m]
    M : (n*n_pts,) bending moment [N*m]  (positive = sagging)
    V : (n*n_pts,) shear force [N]
    """
    Le = L / n
    EI = E * I
    x_list, M_list, V_list = [], [], []

    for e in range(n):
        x_e = e * Le
        ue  = u[[2*e, 2*e + 1, 2*e + 2, 2*e + 3]]
        # Constant shear (third derivative of cubic Hermite w.r.t. xi)
        d3N = np.array([12.0, 6.0*Le, -12.0, 6.0*Le])
        V_e = (EI / Le**3) * (d3N @ ue)

        # Avoid duplicate interior nodes: endpoint=False except last element
        xi_pts = np.linspace(0.0, 1.0, n_pts, endpoint=(e == n - 1))
        for xi in xi_pts:
            d2N = np.array([
                -6.0 + 12.0*xi,
                Le * (-4.0 + 6.0*xi),
                6.0 - 12.0*xi,
                Le * (-2.0 + 6.0*xi),
            ])
            M_xi = (EI / Le**2) * (d2N @ ue)
            x_list.append(x_e + xi * Le)
            M_list.append(M_xi)
            V_list.append(V_e)

    return np.array(x_list), np.array(M_list), np.array(V_list)


def apply_point_load(
    f: np.ndarray,
    x: float,
    L: float,
    n: int,
    force: float = 0.0,
    moment: float = 0.0,
) -> None:
    """
    Add a point force and/or moment at position x along the beam (in-place).

    Uses Hermite shape functions to distribute the load to the two nodes of
    the element containing x, so the result is exact for any x in [0, L].

    Parameters
    ----------
    f      : (2*(n+1),) global load vector  (modified in-place)
    x      : position along the beam [m],  0 <= x <= L
    L      : total beam length [m]
    n      : number of elements
    force  : transverse point force [N]   (positive = +v direction)
    moment : point moment [N*m]           (positive = +theta direction)
    """
    if not (0.0 <= x <= L):
        raise ValueError(f"x={x} is outside the beam [0, {L}]")

    Le = L / n
    e = min(int(x / Le), n - 1)     # element index; clamp so x=L lands on last element
    xi = (x - e * Le) / Le          # local coordinate in [0, 1]

    # Hermite shape functions N(xi): map [v1, th1, v2, th2] -> v(x)
    N1 =  1 - 3*xi**2 + 2*xi**3
    N2 =  Le * xi * (1 - xi)**2
    N3 =  3*xi**2 - 2*xi**3
    N4 =  Le * xi**2 * (xi - 1)

    # Shape function x-derivatives dN/dx: map [v1, th1, v2, th2] -> theta(x)
    dN1 = (-6*xi + 6*xi**2) / Le
    dN2 =  1 - 4*xi + 3*xi**2
    dN3 = ( 6*xi - 6*xi**2) / Le
    dN4 = -2*xi + 3*xi**2

    dofs = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
    contributions = [
        force * N1  + moment * dN1,
        force * N2  + moment * dN2,
        force * N3  + moment * dN3,
        force * N4  + moment * dN4,
    ]
    for dof, val in zip(dofs, contributions):
        f[dof] += val


def solve_mr(
    n: int,
    d: float,
    L: float,
    C10: float,
    C01: float,
    f_ext: np.ndarray,
    prescribed_dofs: list,
    prescribed_values: list,
    *,
    n_gauss_elem: int = 3,
    n_cs: int = 40,
    max_iter: int = 40,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Newton-Raphson solver for a circular-section EB beam with Mooney-Rivlin material.

    The constitutive law is the incompressible two-parameter Mooney-Rivlin model.
    For a fiber at distance y from the neutral axis with curvature kappa:
        lambda(y) = 1 - y * kappa
        sigma(lambda) = 2*C10*(lambda^2 - 1/lambda) + 2*C01*(lambda - 1/lambda^2)

    Section bending moment and tangent flexural stiffness via Gauss integration
    over the circular cross-section (n_cs points in y):
        M(kappa)      = -integral sigma(lambda(y)) * y * b(y) dy
        EI_eff(kappa) =  integral (dsigma/dlambda) * y^2 * b(y) dy
    where b(y) = 2*sqrt(R^2 - y^2) is the chord width at height y.

    Initialization uses the small-strain linear solution (E0 = 6*(C10+C01)) to
    provide consistent nodal rotations, avoiding artificially large element
    curvatures that would cause NR to diverge from a zero starting point.

    Parameters
    ----------
    n, d, L          : number of elements, rod diameter [m], beam length [m]
    C10, C01         : Mooney-Rivlin constants [Pa]
    f_ext            : (2*(n+1),) external force/moment vector
    prescribed_dofs  : list of DOF indices with prescribed values
    prescribed_values: corresponding prescribed displacement/rotation values
    n_gauss_elem     : Gauss points per element along beam axis (default 3)
    n_cs             : Gauss points for cross-section integration (default 40)
    max_iter         : max NR iterations (default 40)
    tol              : convergence tolerance on free-DOF residual norm (default 1e-8)

    Returns
    -------
    u : (2*(n+1),) displacement vector at full load
    R : (2*(n+1),) reaction vector  (non-zero only at prescribed DOFs)
    """
    Le    = L / n
    n_dof = 2 * (n + 1)
    p_dofs = np.array(prescribed_dofs, dtype=int)
    p_vals = np.array(prescribed_values, dtype=float)
    f_dofs = np.setdiff1d(np.arange(n_dof), p_dofs)

    # Gauss points on [0, 1] for element-axis integration
    _pts, _wts = np.polynomial.legendre.leggauss(n_gauss_elem)
    xi_g = (_pts + 1.0) / 2.0
    wt_g = _wts / 2.0

    # Cross-section Gauss points on [-R, R]
    _R = 0.5 * d
    _pts_cs, _wts_cs = np.polynomial.legendre.leggauss(n_cs)
    y_cs = _R * _pts_cs
    w_cs = _R * _wts_cs
    b_cs = 2.0 * np.sqrt(np.maximum(_R**2 - y_cs**2, 0.0))

    def _section(kappa):
        lam = np.clip(1.0 - y_cs * kappa, 1e-4, None)
        sig = 2.0*C10*(lam**2 - 1.0/lam) + 2.0*C01*(lam - 1.0/lam**2)
        Et  = 2.0*C10*(2.0*lam + 1.0/lam**2) + 2.0*C01*(1.0 + 2.0/lam**3)
        M   = -np.dot(sig * y_cs * b_cs, w_cs)
        EI  =  np.dot(Et  * y_cs**2 * b_cs, w_cs)
        return M, EI

    def _assemble(u_cur):
        K_t   = np.zeros((n_dof, n_dof))
        f_int = np.zeros(n_dof)
        for e in range(n):
            dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
            u_e  = u_cur[dofs]
            for xi, wt in zip(xi_g, wt_g):
                Bk = np.array([
                    -6.0 + 12.0*xi,
                    Le * (-4.0 + 6.0*xi),
                    6.0 - 12.0*xi,
                    Le * (-2.0 + 6.0*xi),
                ])
                kappa     = (Bk @ u_e) / Le**2
                M_k, EI_k = _section(kappa)
                f_int[dofs]              += (wt / Le) * M_k * Bk
                K_t[np.ix_(dofs, dofs)] += (wt / Le**3) * EI_k * np.outer(Bk, Bk)
        return K_t, f_int

    # Initialization: linear solution keeps rotations consistent with the
    # prescribed displacements, avoiding artificially large curvatures at
    # element boundaries that would cause NR to diverge from u=0.
    E0 = 6.0 * (C10 + C01)
    I0 = np.pi * d**4 / 64.0
    K_lin = assemble_K(n, E0, I0, L)
    u, _  = solve(K_lin, f_ext, prescribed_dofs, prescribed_values)

    for _ in range(max_iter):
        K_t, f_int = _assemble(u)
        r_f = (f_ext - f_int)[f_dofs]
        if np.linalg.norm(r_f) < tol:
            break
        u[f_dofs] += np.linalg.solve(K_t[np.ix_(f_dofs, f_dofs)], r_f)

    _, f_int_final = _assemble(u)
    R = np.zeros(n_dof)
    R[p_dofs] = f_int_final[p_dofs] - f_ext[p_dofs]
    return u, R


def l2_nodal_error(u, n, L, v_exact_func):
    """L2 displacement error evaluated at nodes."""
    x_nodes = np.linspace(0, L, n + 1)
    v_h  = u[0::2]
    v_ex = np.array([v_exact_func(x) for x in x_nodes])
    return float(np.sqrt(np.mean((v_h - v_ex)**2)))


def l2_interior_error(u, n, L, v_exact_func, n_pts=4):
    """
    Interior L2 displacement error for a uniform mesh.

    Samples n_pts uniformly spaced points per element, excluding element
    endpoints, to avoid nodal superconvergence in Hermite elements.
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


def F_analytical_3pt(d, L, delta, E):
    """
    Analytical midspan reaction force for three-point bending (no self-weight).

    F = 48 E I / L^3 * delta,  I = pi d^4 / 64.
    Accepts numpy arrays for MC propagation.
    """
    I = np.pi * d**4 / 64.0
    return 48.0 * E * I / L**3 * delta
