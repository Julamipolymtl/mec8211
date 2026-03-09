# HW2 — Transient Radial Diffusion Solver

Finite difference solver for transient salt diffusion in a cylindrical concrete pillar.

See docstrings in `src/` for detailed documentation.

## Project Structure

```
src/
  solver.py       # DiffusionParams dataclass and transient BDF1 finite difference solver
  mms.py          # Manufactured solution utilities (SymPy source term derivation)
  convergence.py  # Spatial and temporal convergence studies and error norms
  plots.py        # All plotting routines
  main.py         # Entry point — runs solver, convergence studies, and generates figures
data/
  params_physical.json  # Parameters for the physical HW2 problem
  params_mms.json       # Parameters for MMS verification and convergence studies
  params_template.json  # Annotated reference template for all parameters
tests/
  conftest.py     # Test import configuration
  test_solver.py  # Unit tests for boundary conditions, solver behaviour, and MMS
results/          # Output figures (generated)
doc/              # Documentation
```

## Usage

From the `hw2/` directory:

```bash
python src/main.py data/params_mms.json      # MMS verification and convergence studies
python src/main.py data/params_physical.json # Physical concentration profile
```

All physical, numerical, and convergence parameters are controlled through the JSON file.
See `data/params_template.json` for a fully annotated reference.

## Tests

From the `hw2/` directory:

```bash
pytest
```

## Notice on AI use

AI (Copilot and other LLMs) was used in the making of this project. Its inputs were mainly used in the code architecture planning and writing the documentation.
