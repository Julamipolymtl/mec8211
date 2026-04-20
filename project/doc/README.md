# MEC8211 — V&V Project: Three-Point Bending of TPU Rods

Euler-Bernoulli beam FEM solver with a complete V&V workflow following the
ASME V&V 20 standard. The physical system is a circular TPU rod loaded at
midspan in three-point bending.

---

## Physical problem

A circular TPU rod of nominal diameter $d \approx 5\ \text{mm}$ is tested in
three-point bending at two loading spans:

| Span | $L/d$ | Regime |
|------|-------|--------|
| $L = 60\ \text{mm}$ | 12 | small strains, linear |
| $L = 40\ \text{mm}$ |  8 | large strains, nonlinear |

The system response quantity (SRQ) is the midspan reaction force $F$.
The analytical limit of the 1D EB model is:

$$F = \frac{48\,E\,I}{L^3}\,\delta, \qquad I = \frac{\pi d^4}{64}$$

Material properties are modelled with a Mooney-Rivlin hyperelastic law
($C_{10} = 2.664 \times 10^5\ \text{Pa}$, $C_{01} = 6.601 \times 10^5\ \text{Pa}$),
which linearises to $E_0 = 6(C_{10}+C_{01}) \approx 5.56\ \text{MPa}$ at small strains.
The Young's modulus used in the validation is calibrated independently from
cantilever experiments on each specimen.

---

## Directory layout

```
project/
├── src/
│   ├── beam.py          # FEM solver (element, assembly, solve, error norms)
│   ├── cases.py         # Verification cases (cantilever, SS-UDL, MMS sine)
│   └── data_loaders.py  # CSV loaders for validation scripts
│
├── scripts/
│   ├── 1d_convergence.py               # [Step 1] 1D FEM GCI convergence study + figures
│   ├── 1d_mc_propagation.py            # [Step 1] MC input uncertainty for 1D model
│   ├── plot_1d_beam_fields.py          # [Step 1] Displacement and internal force plots
│   │
│   ├── ansys_convergence.py            # [Step 2] ANSYS mesh convergence + Richardson/GCI
│   ├── ansys_sensitivity.py            # [Step 2] ANSYS linear E-sweep vs experiment
│   │
│   ├── postprocess_exp_3pt_bending.py  # [Step 3] Extract SRQs from raw test files
│   ├── postprocess_exp_cantilever_E.py # [Step 3] Calibrate E from cantilever deflection
│   │
│   └── validate_asme.py                # [Step 4] Full ASME V&V 20 validation
│
├── tests/
│   └── test_beam.py     # Pytest unit tests for src/beam.py and src/cases.py
│
├── data/
│   ├── specimens.csv          # Per-specimen geometry: d (3 readings), mass, length
│   ├── experimental.csv       # F_exp [N] per (L_span, delta, specimen_id)
│   ├── simulation_ansys.csv   # ANSYS 2D results: nonlinear, linear E-sweep, convergence
│   ├── cantilever_exp.csv     # Tip deflection readings for E calibration
│   └── experimental_raw/      # Raw test-machine output (A-F conditions, 6 specimens)
│
├── results/                   # Generated figures and CSV outputs (git-tracked)
└── doc/
    └── README.md              # This file
```

---

## Workflow

Run all scripts from the **project root**:

```bash
python scripts/<script_name>.py
```

### Step 0 — Run the test suite

```bash
python -m pytest tests/ -v
```

Verifies the FEM solver before doing anything else. All tests cover element
stiffness, assembly, boundary conditions, and convergence rates for each
verification case.

---

### Step 1 — 1D FEM code verification

```bash
python scripts/1d_convergence.py
python scripts/1d_mc_propagation.py
python scripts/plot_1d_beam_fields.py
```

The solver uses Hermite cubic elements. For smooth exact solutions the
interior $L^2$ error converges at $O(h^4)$; nodal displacements are
superconvergent (machine precision for polynomial exact solutions).

The GCI follows Roache (1994) with safety factor $F_s = 1.25$:

$$\text{GCI}_\text{fine} = F_s \frac{|e_\text{fine} - e_\text{medium}|}{e_\text{fine}(r^{\hat{p}} - 1)}$$

**Outputs** — `results/convergence_cant.png`, `results/convergence_ss.png`,
`results/convergence_mms.png`, `results/convergence_3pt.png`,
`results/mms_solution.png`, `results/mms_source.png`,
`results/1d_gci.csv`, `results/1d_u_input.csv`,
`results/mc_distributions_L40.png`, `results/mc_distributions_L60.png`,
`results/beam_fields_cantilever.png`, `results/beam_fields_3pt.png`

---

### Step 2 — ANSYS 2D mesh convergence

```bash
python scripts/ansys_convergence.py
python scripts/ansys_sensitivity.py
```

`ansys_convergence.py` reads convergence rows from `data/simulation_ansys.csv`
(fixed $L$, $\delta$, varying $h$) and produces two figures: a total view of
relative error vs mesh size and an asymptotic Richardson extrapolation with GCI
bounds.

`ansys_sensitivity.py` overlays the linear-elastic ANSYS parametric $E$
sweep ($E = 6$–$14\ \text{MPa}$) against experimental means to identify the
best-fit modulus before validation.

**Outputs** — `results/ansys_conv_total.png`, `results/ansys_conv_asymptotic.png`,
`results/ansys_gci.csv` (Richardson/GCI for reference condition),
`results/ansys_u_input.csv` (input uncertainty per test condition).

---

### Step 3 — Raw data postprocessing and material calibration

These scripts are run **once**; their outputs are committed.

```bash
python scripts/postprocess_exp_3pt_bending.py
python scripts/postprocess_exp_cantilever_E.py
```

`postprocess_exp_3pt_bending.py` extracts per-specimen SRQs (peak reaction forces)
from raw test-machine `.txt` files and produces the values in
`data/experimental.csv` from the raw files in `data/experimental_raw/`.

`postprocess_exp_cantilever_E.py` back-calculates $E$ per specimen by inverting
the clamped-free tip-deflection formula including self-weight:

$$v_\text{tip} = \frac{F_\text{tip}\,L^3/3 + w\,L^4/8}{E\,I}
\quad \Rightarrow \quad
E = \frac{F_\text{tip}\,L^3/3 + w\,L^4/8}{I\,v_\text{tip}}$$

where $w = m_\text{free}\,g / L_\text{cant}$ is the beam self-weight per unit
length and $F_\text{tip} = m_\text{tip}\,g$.

**Output** — `results/calibrate_E.csv` (per-specimen $E_\text{mean}$, $E_\text{std}$, CV)

---

### Step 4 — ASME V&V 20 validation

```bash
python scripts/validate_asme.py
```

Requires `results/calibrate_E.csv` from Step 3 and
`results/ansys_gci.csv`, `results/ansys_u_input.csv` from Step 2.

Loads the cantilever-calibrated $E$ (mean and std across specimens) and uses
it as the model input distribution for Monte Carlo uncertainty propagation:

| Input | Distribution | Rationale |
|-------|-------------|-----------|
| $d$ | $\mathcal{N}(d_i,\,\sigma_{d,i})$ | 3 caliper readings per specimen |
| $L_\text{span}$ | $\mathcal{N}(L_\text{nom},\,0.25\ \text{mm})$ | machined fixture tolerance |
| $\delta$ | $\mathcal{N}(\delta_\text{nom},\,0.005\ \text{mm})$ | test-machine encoder |
| $E$ | $\mathcal{N}(\bar{E}_\text{cant},\,s_\text{cant})$ | cantilever calibration |

The validation error and combined uncertainty are (ASME V&V 20 eq. 1.5):

$$\mathcal{E} = \bar{F}_\text{exp} - F_\text{sim}, \qquad
U_\text{val} = \sqrt{u_\text{num}^2 + U_\text{input}^2 + U_D^2}$$

where $U_D = t_{0.975,5}\,s/\sqrt{6}$ is the 95 % CI on the experimental mean
and $u_\text{num} \approx 0$ for the 1D model (Hermite exactness for a
cubic-polynomial SRQ).  A condition is **validated** when $|\mathcal{E}| < U_\text{val}$.

For the **1D model**, $U_\text{input}$ is the 95 % half-width of the MC output
distribution (50 000 samples, inputs from the table above).
For the **ANSYS 2D model**, $U_\text{input} = |{\partial F}/{\partial E}|\,\sigma_E$
computed from the linear $E$-sweep in Step 2, and $u_\text{num}$ is the GCI
from mesh convergence.

**Outputs** — `results/validation_ansys.png`, `results/validation_1d.png`,
`results/validation_error_budget.png`, `results/validation_error_budget_ansys.png`,
`results/mc_distributions_L60.png`, `results/mc_distributions_L40.png`

---

## Key source files

### `src/beam.py`

| Function | Description |
|----------|-------------|
| `element_stiffness(E, I, Le)` | $4\times4$ Hermite element stiffness matrix |
| `assemble_K(n, E, I, L)` | Global stiffness matrix |
| `assemble_distributed_load(n, L, w)` | Consistent nodal load, uniform $w$ |
| `assemble_general_load(n, L, w_func)` | Consistent nodal load, arbitrary $w(x)$ |
| `solve(K, f, prescribed_dofs, values)` | Apply BCs and solve $\mathbf{K}\mathbf{u} = \mathbf{f}$ |
| `apply_point_load(f, x, L, n, ...)` | Add point force/moment at arbitrary $x$ |
| `apply_prescribed_displacement(...)` | Penalty-method soft constraint at arbitrary $x$ |
| `compute_internal_forces(u, E, I, L, n)` | $M(x)$ and $V(x)$ along beam |
| `solve_mr(n, d, L, C10, C01, ...)` | Newton-Raphson solver, Mooney-Rivlin material |
| `l2_nodal_error(u, n, L, v_exact)` | $L^2$ error at nodes |
| `l2_interior_error(u, n, L, v_exact)` | $L^2$ error at interior points (avoids superconvergence) |
| `F_analytical_3pt(d, L, delta, E)` | Analytical midspan force, array-safe |

### `src/cases.py`

| Name | Problem |
|------|---------|
| `CANTILEVER_UDL` | Clamped-free beam, uniform load |
| `SS_UDL` | Simply supported beam, uniform load |
| `MMS_SINE` | Method of manufactured solutions, $v_\text{mms} = \sin(3\pi x/L)$ |

### `data/simulation_ansys.csv`

Single file for all ANSYS 2D results.  The `model` column selects the row type:

| `model` | `E_MPa` | `h_mm` | Description |
|---------|---------|--------|-------------|
| `nonlinear` | — | production $h$ | Nonlinear (Mooney-Rivlin) runs at finest mesh |
| `linear` | set | production $h$ | Linear-elastic parametric $E$ sweep |
| `nonlinear` | — | varies | Mesh convergence study rows (fixed $L$, $\delta$) |

---

## Dependencies

```
numpy
scipy
matplotlib
pytest
```

Install with `pip install numpy scipy matplotlib pytest`.
