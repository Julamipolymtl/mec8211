import numpy as np

import matplotlib.pyplot as plt

import devoir3_lbm_accelerated as lbm


# ============================================================
# PARAMETRES
# ============================================================

SEED = 101
DELTA_P = 0.1
PORO = 0.9
MEAN_FIBER_D = 12.5
STD_D = 2.85

DX_BASE = 2e-6
NX_BASE = 100

# Garder le domaine constant
DOMAINE = NX_BASE * DX_BASE

# Raffinnement de maillage constant
RAFF = [1, 0.5, 0.25, 0.125]

# ============================================================
# FONCTIONS
# ============================================================

def lancer_etude_convergence():
    """
    Lance une étude de convergence sur la porisité, k_in_micron2,
    en raffinant le maillage tout en gardant le domaine constant.
    """
    dx_vals = []
    k_vals = []

    for raffinement in RAFF:
        dx = DX_BASE / raffinement
        nx = int(DOMAINE / dx)

        nom_fichier = f"fiber_mat_{nx}.tiff"

        print("\n==============================")
        print(f"nx = {nx}, dx = {dx:.2e}")
        print("==============================")

        d_equivalent = lbm.Generate_sample(
            SEED,
            nom_fichier,
            MEAN_FIBER_D,
            STD_D,
            PORO,
            nx,
            dx,
        )

        k_val = lbm.LBM(
            nom_fichier,
            nx,
            DELTA_P,
            dx,
            d_equivalent,
        )

        dx_vals.append(dx)
        k_vals.append(k_val)

    dx_vals = np.array(dx_vals)
    k_vals = np.array(k_vals)

    calculer_ordre_convergence(k_vals)
    tracer_convergence(dx_vals, k_vals)

# ============================================================

def calculer_ordre_convergence(k_vals):
    """
    Calcule et affiche l'ordre de convergence observé.
    """
    ordres = []

    for i in range(len(k_vals) - 2):
        num = abs(k_vals[i] - k_vals[i + 1])
        den = abs(k_vals[i + 1] - k_vals[i + 2])

        ordre = np.log(num / den) / np.log(2)
        ordres.append(ordre)

    print("\nOrdres de convergence observés :")
    for i, ordre in enumerate(ordres):
        print(f"p{i} = {ordre:.4f}")

def tracer_convergence(dx_vals, k_vals):
    """
    Trace la courbe de convergence en échelle log-log.
    """
    k_ref = k_vals[-1]
    erreur = np.abs(k_vals - k_ref)

    plt.figure()
    plt.loglog(dx_vals, erreur, "o-", label="Erreur")
    plt.xlabel("dx")
    plt.ylabel("|k - k_ref|")
    plt.title("Convergence de k")
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    lancer_etude_convergence()