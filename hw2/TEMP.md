# Work plan for hw2

Switch from steady-state to transient.

dCdt -> Explicit or implicit? I would do imp. (lin so pretty ez pz + no stability criterion)

solver.py -> solve_step + solve

bdf1, bdf2! (check order in time). bdf2 IC? + order 1 & order 2

Add MMS calculation fct (?) (Choix de MMS?)

Q_bonus, switch from main script to .exe style code (solve.py prm1 prm2) + post-process script