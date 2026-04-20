"""
Cantilever beam verification case.

Setup: clamped at node 0, known point load at tip (node n).
SRQ  : tip transverse displacement.

Analytical solution (superposition of tip load + UDL self-weight):
  v(L) = F_tip * L^3 / (3 E I) + w * L^4 / (8 E I)
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from beam import assemble_K, assemble_distributed_load, apply_point_load, solve

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
n         = 10       # number of elements
L         = 0.200    # beam length [m]
E         = 10e6     # Young's modulus [Pa]
d         = 0.010    # rod diameter [m]
mass      = 0.050    # rod mass [kg]  (self-weight distributed load)
m_tip     = 0.020    # mass hung at tip [kg]

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
I      = np.pi * d**4 / 64          # second moment of area [m^4]
g      = 9.81                        # gravitational acceleration [m/s^2]
w      = -mass * g / L               # self-weight per unit length [N/m]  (gravity = -y)
F_tip  = -m_tip * g                  # tip point load [N]  (downward, -y)

# ---------------------------------------------------------------------------
# Assemble system
# ---------------------------------------------------------------------------
K     = assemble_K(n, E, I, L)
f_ext = assemble_distributed_load(n, L, w)
apply_point_load(f_ext, x=L, L=L, n=n, force=F_tip)

# ---------------------------------------------------------------------------
# Boundary conditions: clamped at node 0
# ---------------------------------------------------------------------------
prescribed_dofs   = [0, 1]
prescribed_values = [0.0, 0.0]

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
u, R = solve(K, f_ext, prescribed_dofs, prescribed_values)

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
v_tip        = u[2*n]
v_analytical = F_tip * L**3 / (3*E*I) + w * L**4 / (8*E*I)

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
print(f"  Tip load F_tip  : {F_tip:.4f} N  ({m_tip*1e3:.1f} g)")
print("-" * 50)
print(f"  FEM tip displ.  : {v_tip*1e3:.6f} mm")
print(f"  Analytical      : {v_analytical*1e3:.6f} mm")
print(f"  Relative error  : {abs(v_tip - v_analytical)/abs(v_analytical)*100:.4f} %")
print("=" * 50)
