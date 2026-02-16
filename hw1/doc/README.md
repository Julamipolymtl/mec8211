# HW1 — Radial Diffusion Solver

Finite difference solver for steady-state salt diffusion in a cylindrical concrete pillar. Two schemes are implemented: forward difference (1st order) and central difference (2nd order) for the dC/dr term.

See docstrings in `src/` for detailed documentation.

## Project Structure

```
src/
  analytical.py   # Analytical solution
  solver.py       # Finite difference solver
  convergence.py  # Grid convergence study & error norms
  plots.py        # All plotting routines
  main.py         # Entry point — generates all figures
results/          # Output plots (generated)
```

## Usage

From the `src/` directory:

```bash
python main.py
```

This produces all concentration profile and convergence plots in `results/`.
