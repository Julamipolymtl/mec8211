# HW2 — Transient Radial Diffusion Solver

Finite difference solver for transient salt diffusion in a cylindrical concrete pillar.

See docstrings in `src/` for detailed documentation.

## Project Structure

```
src/
  analytical.py   # Analytical solution
  solver.py       # Finite difference solver
  convergence.py  # Grid convergence study & error norms
  plots.py        # All plotting routines
  main.py         # Entry point — generates data
  post-process.py # Generates all plots from the data
tests/
  conftest.py     # Test import configuration
  test_solver.py  # Unit tests for analytical solution & solver
results/          # Output plots (generated)
data/             # Raw data output from code
doc/              # Documentation
```

## Usage

From the `src/` directory:

```bash
python main.py
```

## Tests

From the project root (`hw1/`):

```bash
python -m pytest tests/ -v
```

## Notice on AI use

AI (Copilot and other LLMs) was used in the making of this project. Its inputs were mainly used in the code architecture planning and writing the documentation. 
