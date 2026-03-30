# HW3 - Permeabilite d'un mat de fibres (LBM)

Solveur Lattice Boltzmann (D2Q9) pour calculer la permeabilite d'un mat de fibres 2D.

## Structure

```
src/
  devoir3_lbm_accelerated.py      # Solveur LBM (blackbox, fourni)
  run_convergence.py              # Etude de convergence (Part A)
  run_uncertainty_propagation.py  # Propagation d'incertitude Monte Carlo (Part B)
  run_validation.py               # Validation ASME V&V20 (Part C-E)
results/
  convergence/                    # Resultats Part A
  montecarlo/                     # Resultats Part B
  validation/                     # Resultats Part C-E
doc/
```

## Dependances

```bash
pip install numpy matplotlib scipy Pillow
```

## Utilisation

Les scripts doivent etre executes dans l'ordre depuis `hw3/` :

```bash
python src/run_convergence.py              # Part A - genere results/convergence/
python src/run_uncertainty_propagation.py  # Part B - genere results/montecarlo/
python src/run_validation.py               # Part C-E - lit les resultats A et B
```

Resultats dans `results/` (fichiers `.dat`, non suivis par git).

## IA

Copilot et d'autres LLMs ont ete utilises pour la planification de l'architecture du code et la documentation.
