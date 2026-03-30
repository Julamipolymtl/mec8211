# HW3 - Permeabilite d'un mat de fibres (LBM)

Solveur Lattice Boltzmann (D2Q9) pour calculer la permeabilite d'un mat de fibres 2D.

## Structure

```
src/
  devoir3_lbm_accelerated.py      # Solveur LBM (blackbox, fourni)
  convergence.py                  # Convergence single-seed
  run_convergence.py              # Convergence multi-seed (Part A)
  run_uncertainty_propagation.py  # Monte Carlo (Part B)
data/
results/
doc/
tests/
```

## Utilisation

Depuis `hw3/` :

```bash
python src/run_convergence.py
python src/run_uncertainty_propagation.py
```

Resultats dans `results/`.

## IA

Copilot et d'autres LLMs ont ete utilises pour la planification de l'architecture du code et la documentation.
