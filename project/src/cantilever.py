"""
Cantilever beam verification case.

Setup: clamped at node 0, prescribed transverse displacement at tip (node n).
SRQ  : reaction force at the tip.

Analytical solution: F = 3 E I delta / L^3
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from beam import assemble_K, assemble_distributed_load, solve

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
n         = 10       # number of elements
L         = 0.200    # beam length [m]
E         = 10e6     # Young's modulus [Pa]
d         = 0.010    # rod diameter [m]
mass      = 0.050    # rod mass [kg]  (used for self-weight distributed load)
delta     = 5e-3     # imposed tip displacement [m]

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
I  = np.pi * d**4 / 64          # second moment of area [m^4]
g  = 9.81                        # gravitational acceleration [m/s^2]
w  = mass * g / L                # self-weight per unit length [N/m]

# ---------------------------------------------------------------------------
# Assemble system
# ---------------------------------------------------------------------------
K     = assemble_K(n, E, I, L)
f_ext = assemble_distributed_load(n, L, w)

# ---------------------------------------------------------------------------
# Boundary conditions
#   node 0 : clamped  -> v=0, theta=0  (DOFs 0, 1)
#   node n : imposed  -> v=delta        (DOF 2n)
#   node n : tip rotation is FREE       (DOF 2n+1)
# ---------------------------------------------------------------------------
prescribed_dofs   = [0, 1, 2*n]
prescribed_values = [0.0, 0.0, delta]

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
u, R = solve(K, f_ext, prescribed_dofs, prescribed_values)

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
F_tip        = R[2*n]                                   # load cell reading at tip
F_analytical = 3*E*I*delta / L**3 - 3*w*L / 8          # superposition with self-weight

node_coords = np.linspace(0, L, n + 1)
v           = u[0::2]    # transverse displacements at all nodes
theta       = u[1::2]    # rotations at all nodes

print("=" * 50)
print("Cantilever beam — verification")
print("=" * 50)
print(f"  n elements      : {n}")
print(f"  Length L        : {L*1e3:.1f} mm")
print(f"  Diameter d      : {d*1e3:.1f} mm")
print(f"  E               : {E/1e6:.1f} MPa")
print(f"  I               : {I:.4e} m^4")
print(f"  Self-weight w   : {w:.4f} N/m")
print(f"  Tip displacement: {delta*1e3:.1f} mm")
print("-" * 50)
print(f"  FEM tip force   : {F_tip:.6f} N")
print(f"  Analytical      : {F_analytical:.6f} N")
print(f"  Relative error  : {abs(F_tip - F_analytical)/abs(F_analytical)*100:.4f} %")
print("=" * 50)
