"""
Conversion Python 3 des scripts MATLAB : launch_simulationLBM.m, Generate_sample.m, LBM.m
Auteurs originaux : Sébastien Leclaire (2014), modifié par David Vidal

Parallélisation :
  - Generate_sample : remplissage de la grille entièrement vectorisé (NumPy, sans boucle Python)
  - LBM             : noyau de simulation compilé JIT par Numba avec parallel=True
                      → utilise automatiquement TOUS les cœurs disponibles via prange

Dépendances :
    pip install numpy pillow matplotlib numba
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit, prange


# ==============================================================================
# FONCTION : Generate_sample
# ==============================================================================

def Generate_sample(seed, filename, mean_d, std_d, poro, nx, dx):
    """
    Crée une structure 2D de fibres et l'exporte en format TIFF.

    Paramètres
    ----------
    seed     : int    - graine du générateur aléatoire (0 = aléatoire)
    filename : str    - nom du fichier TIFF de sortie
    mean_d   : float  - diamètre moyen des fibres [µm]
    std_d    : float  - écart-type des diamètres  [µm]
    poro     : float  - porosité cible
    nx       : int    - taille latérale du domaine [cellules]
    dx       : float  - taille d'une cellule [m]

    Retourne
    --------
    d_equivalent : float - diamètre équivalent [µm]
    """

    rng    = np.random.default_rng(None if seed == 0 else seed)
    dx_um  = dx * 1e6
    domain = nx * dx_um

    # -------------------------------------------------------------------------
    # Distribution des fibres
    # -------------------------------------------------------------------------
    dist_full = rng.normal(mean_d, std_d, 10000)

    nb_fiber     = 1
    poro_eff     = 1.0 - np.sum(dist_full[:nb_fiber] ** 2 / 4 * np.pi) / domain ** 2
    poro_eff_old = poro_eff

    while poro_eff >= poro:
        poro_eff_old = poro_eff
        nb_fiber    += 1
        poro_eff     = 1.0 - np.sum(dist_full[:nb_fiber] ** 2 / 4 * np.pi) / domain ** 2

    if abs(poro_eff - poro) > abs(poro_eff_old - poro):
        nb_fiber -= 1
        poro_eff  = poro_eff_old

    dist_d       = np.sort(dist_full[:nb_fiber])[::-1]
    d_equivalent = np.sum(dist_d ** 2) / np.sum(dist_d)
    print(f"d_equivalent     = {d_equivalent:.4f} µm")

    # -------------------------------------------------------------------------
    # Positionnement des fibres (sans chevauchement, conditions périodiques)
    # Vérification vectorisée sur les 9 images périodiques
    # -------------------------------------------------------------------------
    circles     = np.zeros((nb_fiber, 3))
    circles[0]  = [rng.random() * domain, rng.random() * domain, dist_d[0]]
    fiber_count = 1
    offsets     = np.array([0.0, domain, -domain])

    while fiber_count < nb_fiber:
        di = dist_d[fiber_count]
        xi = rng.random() * domain
        yi = rng.random() * domain

        xc = circles[:fiber_count, 0]
        yc = circles[:fiber_count, 1]
        dc = circles[:fiber_count, 2]
        r2 = (di + dc) ** 2                            # (fiber_count,)

        ox, oy = np.meshgrid(offsets, offsets, indexing='ij')
        ox = ox.ravel()                                  # (9,)
        oy = oy.ravel()

        # distances² : (fiber_count, 9)
        dx2 = (xi - xc[:, np.newaxis] + ox[np.newaxis, :]) ** 2
        dy2 = (yi - yc[:, np.newaxis] + oy[np.newaxis, :]) ** 2

        if np.any(dx2 + dy2 < r2[:, np.newaxis]):
            continue   # chevauchement → réessayer

        circles[fiber_count] = [xi, yi, di]
        fiber_count += 1

    print(f"number_of_fibres = {fiber_count}")

    # -------------------------------------------------------------------------
    # Remplissage de la grille — vectorisé NumPy (pas de boucle Python i,j)
    # -------------------------------------------------------------------------
    coords       = (0.5 + np.arange(nx)) * dx_um
    px, py       = np.meshgrid(coords, coords, indexing='ij')   # (nx, nx)
    poremat      = np.zeros((nx, nx), dtype=bool)

    xc = circles[:, 0]
    yc = circles[:, 1]
    r2 = (circles[:, 2] / 2) ** 2

    for k in range(nb_fiber):
        for ox in offsets:
            for oy in offsets:
                poremat |= (px - (xc[k] + ox)) ** 2 + (py - (yc[k] + oy)) ** 2 < r2[k]

    # -------------------------------------------------------------------------
    # Export TIFF et affichage
    # -------------------------------------------------------------------------
    # poremat est en convention (i=x, j=y) — on transpose pour l'export image
    # (lignes=y, cols=x), ce qui donne de vrais cercles à l'affichage.
    poremat_img = poremat.T   # (NY, NX) : lignes=y, colonnes=x
    Image.fromarray(poremat_img.astype(np.uint8) * 255).save(filename)

    plt.figure(1)
    plt.imshow(np.rot90(poremat_img, k=1), cmap='gray')
    plt.title("Structure de fibres générée")
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    return d_equivalent


# ==============================================================================
# NOYAU LBM COMPILÉ PAR NUMBA  (parallel=True → prange utilise tous les cœurs)
# ==============================================================================

@njit(parallel=True, cache=True)
def _lbm_step(N, SOLID, W, cx, cy, NX, NY, deltaP, dx, rho0, dt, OMEGA, bb_idx):
    """
    Un pas de temps LBM D2Q9 compilé JIT.

    parallel=True + prange → Numba distribue automatiquement les boucles
    sur tous les cœurs logiques disponibles (via OpenMP en arrière-plan).

    Retourne
    --------
    N_out    : ndarray (NX*NY, 9) - nouvelles fonctions de distribution
    ux_out   : ndarray (NX*NY,)   - vitesse x de chaque cellule
    FlowRate : float              - débit moyen sur la première rangée
    """
    NQ    = 9
    NCELL = NX * NY

    # ------------------------------------------------------------------
    # 1) Streaming périodique — parallélisé sur les directions q
    # ------------------------------------------------------------------
    N_stream = np.empty_like(N)
    N_stream[:, 0] = N[:, 0]   # direction au repos : aucun déplacement

    for q in prange(1, NQ):
        shift_x = int(cx[q])
        shift_y = int(cy[q])
        for idx in range(NCELL):
            i     = idx // NY
            j     = idx  % NY
            src_i = (i - shift_x) % NX
            src_j = (j - shift_y) % NY
            N_stream[idx, q] = N[src_i * NY + src_j, q]

    # ------------------------------------------------------------------
    # 2) Sauvegarde des nœuds solides (bounce-back avant collision)
    # ------------------------------------------------------------------
    N_solid_save = np.empty((NCELL, NQ), dtype=N.dtype)
    for idx in prange(NCELL):
        if SOLID[idx]:
            for q in range(NQ):
                N_solid_save[idx, q] = N_stream[idx, bb_idx[q]]

    # ------------------------------------------------------------------
    # 3) Moments macroscopiques + collision BGK — parallélisé sur NCELL
    # ------------------------------------------------------------------
    ux_out = np.empty(NCELL)

    for idx in prange(NCELL):
        rho_i = 0.0
        ux_i  = 0.0
        uy_i  = 0.0
        for q in range(NQ):
            f      = N_stream[idx, q]
            rho_i += f
            ux_i  += f * cx[q]
            uy_i  += f * cy[q]

        ux_i    = ux_i / rho_i + deltaP / (2.0 * NX * dx * rho0) * dt
        uy_i    = uy_i / rho_i
        ux_out[idx] = ux_i

        u2 = ux_i ** 2 + uy_i ** 2
        for q in range(NQ):
            cu  = ux_i * cx[q] + uy_i * cy[q]
            feq = rho_i * W[q] * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u2)
            N_stream[idx, q] += OMEGA * (feq - N_stream[idx, q])

    # ------------------------------------------------------------------
    # 4) Rétablissement des nœuds solides
    # ------------------------------------------------------------------
    for idx in prange(NCELL):
        if SOLID[idx]:
            for q in range(NQ):
                N_stream[idx, q] = N_solid_save[idx, q]

    # ------------------------------------------------------------------
    # 5) Débit sur la première rangée (indices 0..NY-1)
    # ------------------------------------------------------------------
    flow = 0.0
    for j in range(NY):
        flow += ux_out[j]
    flow /= (NX * dx)

    return N_stream, ux_out, flow


# ==============================================================================
# FONCTION : LBM
# ==============================================================================

def LBM(filename, NX, deltaP, dx, d_equivalent):
    """
    Calcule l'écoulement à travers le mat de fibres par la méthode LBM (D2Q9).

    Paramètres
    ----------
    filename     : str   - fichier TIFF de la structure de fibres
    NX           : int   - taille du domaine (carré NX×NX)
    deltaP       : float - chute de pression [Pa]
    dx           : float - taille d'une cellule [m]
    d_equivalent : float - diamètre équivalent des fibres [µm]
    """
    NY      = NX
    OMEGA   = 1.0
    rho0    = 1.0
    mu      = 1.8e-5
    epsilon = 1e-8

    dt = (1.0 / OMEGA - 0.5) * rho0 * dx ** 2 / 3.0 / mu

    # Lecture de la structure
    A     = np.array(Image.open(filename)).astype(bool)
    SOLID = A.flatten()

    # Paramètres D2Q9
    W  = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
    cx = np.array([0,   0,   1,    1,   1,    0,   -1,   -1,  -1  ], dtype=np.float64)
    cy = np.array([0,   1,   1,    0,  -1,   -1,   -1,    0,   1  ], dtype=np.float64)
    bb_idx = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4], dtype=np.int64)

    N            = np.outer(np.ones(NX * NY), rho0 * W)
    FlowRate_old = 1.0
    FlowRate     = 0.0
    t_           = 1

    print("Démarrage LBM (Numba JIT multi-cœur)...")
    print("Compilation JIT au 1er pas — quelques secondes d'attente normale.")

    # Boucle temporelle
    while FlowRate == 0.0 or abs(FlowRate_old - FlowRate) / abs(FlowRate) >= epsilon:
        N, ux, FlowRate_new = _lbm_step(
            N, SOLID, W, cx, cy, NX, NY, deltaP, dx, rho0, dt, OMEGA, bb_idx
        )
        FlowRate_old = FlowRate
        FlowRate     = FlowRate_new
        t_          += 1

        if t_ % 500 == 0:
            print(f"  pas {t_:6d} | FlowRate = {FlowRate:.6e}")

    print(f"Convergence après {t_} pas de temps.")

    # Résultats
    poro_eff = 1.0 - SOLID.sum() / (NX * NY)
    u_mean   = ux[:NY].mean()
    Re  = rho0 * u_mean * poro_eff * d_equivalent * 1e-6 / (mu * (1 - poro_eff))
    k   = u_mean * mu / deltaP * (NX * dx) * 1e12

    print(f"\nporo_eff         = {poro_eff:.6f}")
    print(f"Re               = {Re:.6e}")
    print(f"k_in_micron2     = {k:.6f} µm²")

    # Visualisation
    # origin='lower' : Y=1 en bas, croît vers le haut → cohérent avec Generate_sample
    ux_plot  = ux.copy()
    uy_plot  = np.zeros_like(ux)
    ux_plot[SOLID] = 0.0

    # reshape(NX,NY) → axe0=x, axe1=y ; .T → (NY,NX) pour imshow (lignes=y, cols=x)
    ux_2d    = ux_plot.reshape(NX, NY).T
    uy_2d    = uy_plot.reshape(NX, NY).T
    solid_2d = SOLID.reshape(NX, NY).T.astype(float)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(2 - solid_2d, cmap='gray', vmin=0, vmax=2,
              origin='lower', extent=[0.5, NX + 0.5, 0.5, NY + 0.5])
    X, Y = np.meshgrid(np.arange(1, NX + 1), np.arange(1, NY + 1))
    ax.quiver(X, Y, ux_2d, uy_2d, color='blue', scale=None, scale_units='xy')
    ax.set_xlim(0.5, NX + 0.5)
    ax.set_ylim(0.5, NY + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f"Champ de vitesse après {t_} pas de temps")
    plt.tight_layout()

    #ajout par emile
    os.makedirs("figures", exist_ok=True)

    nom_fig = f"figures/vitesse_NX{NX}_dx{dx:.1e}.png"

    fig.savefig(nom_fig, dpi=300, bbox_inches="tight")
    print(f"✔ Figure sauvegardée : {nom_fig}")


    return k


# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================

if __name__ == "__main__":

    seed         = 101
    deltaP       = 0.1
    NX           = 100
    poro         = 0.9
    mean_fiber_d = 12.5
    std_d        = 2.85
    dx           = 2e-6
    filename     = 'fiber_mat.tiff'

    d_equivalent = Generate_sample(seed, filename, mean_fiber_d, std_d, poro, NX, dx)
    LBM(filename, NX, deltaP, dx, d_equivalent)
