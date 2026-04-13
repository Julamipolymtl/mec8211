"""
Three-point bending case.

Setup: simply supported at both ends of the loading span, prescribed transverse
       displacement at midspan.
SRQ  : reaction force at the midspan loading point.

n must be even so that a node falls exactly at midspan.

Analytical solution (no self-weight): F = 48 E I delta / L_span^3
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from beam import assemble_K, assemble_distributed_load, solve

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
n            = 10       # number of elements (must be even)
L_span       = 0.160    # loading span — distance between supports [m]
E            = 10e6     # Young's modulus [Pa]
d            = 0.010    # rod diameter [m]
mass         = 0.050    # rod mass [kg]  (self-weight over the modeled span)
delta        = -5e-3    # imposed midspan displacement [m]  (downward, -y)

# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------
if n % 2 != 0:
    raise ValueError(f"n must be even for midspan node to exist (got n={n})")

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
I  = np.pi * d**4 / 64          # second moment of area [m^4]
g  = 9.81                        # gravitational acceleration [m/s^2]
w  = -mass * g / L_span          # self-weight per unit length over span [N/m]  (gravity = -y)

mid_node = n // 2                # node index at midspan

# ---------------------------------------------------------------------------
# Assemble system
# ---------------------------------------------------------------------------
K     = assemble_K(n, E, I, L_span)
f_ext = assemble_distributed_load(n, L_span, w)

# ---------------------------------------------------------------------------
# Boundary conditions
#   node 0     : pin    -> v=0        (DOF 0),  theta FREE
#   node n     : roller -> v=0        (DOF 2n), theta FREE
#   node n//2  : imposed-> v=delta    (DOF 2*mid_node)
# ---------------------------------------------------------------------------
prescribed_dofs   = [0, 2*n, 2*mid_node]
prescribed_values = [0.0, 0.0, delta]

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
u, R = solve(K, f_ext, prescribed_dofs, prescribed_values)

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
F_mid        = R[2*mid_node]                                    # load cell reading at midspan
F_analytical = 48*E*I*delta / L_span**3 - 5*w*L_span / 8       # superposition with self-weight

node_coords = np.linspace(0, L_span, n + 1)
v           = u[0::2]    # transverse displacements
theta       = u[1::2]    # rotations

print("=" * 50)
print("Three-point bending — verification")
print("=" * 50)
print(f"  n elements      : {n}")
print(f"  Loading span    : {L_span*1e3:.1f} mm")
print(f"  Diameter d      : {d*1e3:.1f} mm")
print(f"  E               : {E/1e6:.1f} MPa")
print(f"  I               : {I:.4e} m^4")
print(f"  Self-weight w   : {w:.4f} N/m")
print(f"  Midspan displ.  : {delta*1e3:.1f} mm")
print("-" * 50)
print(f"  FEM midspan force  : {F_mid:.6f} N")
print(f"  Analytical         : {F_analytical:.6f} N")
print(f"  Relative error     : {abs(F_mid - F_analytical)/abs(F_analytical)*100:.4f} %")
print("=" * 50)
