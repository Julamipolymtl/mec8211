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
          At free DOFs      : ≈ 0 (equilibrium residual)
          At prescribed DOFs: reaction force (load cell reading)
    """
    n_dof = K.shape[0]
    all_dofs = np.arange(n_dof)
    p_dofs = np.array(prescribed_dofs, dtype=int)
    p_vals = np.array(prescribed_values, dtype=float)
    f_dofs = np.setdiff1d(all_dofs, p_dofs)

    K_ff = K[np.ix_(f_dofs, f_dofs)]
    K_fp = K[np.ix_(f_dofs, p_dofs)]

    rhs = f_ext[f_dofs] - K_fp @ p_vals
    u_f = np.linalg.solve(K_ff, rhs)

    u = np.zeros(n_dof)
    u[f_dofs] = u_f
    u[p_dofs] = p_vals

    # Reaction vector: R = K @ u - f_ext
    #   At free DOFs    : R ≈ 0  (equilibrium residual, check for correctness)
    #   At prescribed DOFs: R = reaction force (what a load cell would read)
    R = K @ u - f_ext

    return u, R
