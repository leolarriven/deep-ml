"""Version DEPLOYABLE : le corps de `train` a coller tel quel dans le squelette.

Autonome (un seul import : numpy), deterministe, n'utilise NI `X_val` NI `y_val`.
C'est `solution_final.py` transforme en une unique fonction prete a l'emploi :

    predict = train(X_train, y_train, X_val, y_val)
    y_pred  = predict(X_val)          # -> (n,) float64

Deux mecanismes de degradation, deux remedes orthogonaux (cf. RAPPORT.md) :

1. DETECTION des colonnes de bruit. 100 des 264 colonnes sont du bruit pur,
   independant de tout ; un vrai monome est une fonction de 8 variables latentes
   donc fortement inter-correle avec les autres monomes. Le score de coherence
   (meilleure correlation absolue avec une autre colonne) separe les deux modes
   sans ambiguite (bruit 0.14-0.27, monomes 0.83-1.00) et le seuil est le point
   milieu du plus grand trou. Le filtre S'ABSTIENT si ce trou n'existe pas
   (design sans bruit pur : masque complet, aucune degradation).
2. ENSEMBLE de 40 ridge + 20 Huber IRLS sur des sous-espaces de colonnes (k=64)
   et des sous-echantillons de lignes (80 %), avec winsorisation par membre
   (les monomes de degre 3 ont des queues extremes : max|z| = 110 en validation
   contre 15.75 en train). Agregation par moyenne tronquee 20 %.

Mesure identique a `solution_final` (meme algorithme, memes constantes) :
20 seeds -> val R2 moyen 0.6930 / median 0.6988 / min 0.6233 / 20-20 > 0.5.
Cout : ~0.3 s pour 250x264.
"""
import numpy as np


def train(X_train, y_train, X_val, y_val):
    """Entraine une regression qui generalise malgre p > n (features bruitees).

    Args:
        X_train: (n_samples, n_features) standardise.
        y_train: (n_samples,).
        X_val:   (n_val, n_features) -- NON utilise (aucune fuite, pas de
                 transduction : le modele est fige avant de voir la validation).
        y_val:   (n_val,)            -- NON utilise.

    Returns:
        predict : callable X (n, p) -> y_pred (n,) float64.
    """
    # ------------------------------------------------------------- constantes
    # Figees, verifiees sur 20 seeds et 6 variantes du harness (cf. RAPPORT.md).
    N_RIDGE = 40                     # membres ridge
    N_HUBER = 20                     # membres Huber (queues lourdes)
    K_SUBSPACE = 64                  # colonnes par membre
    SUB_ROWS = 0.8                   # fraction de lignes par membre
    TRIM = 0.2                       # moyenne tronquee (coupe 10 % de chaque cote)
    QS = (0.005, 0.01, 0.02, 0.035, 0.05)   # quantiles de winsorisation par membre
    ALPHAS = np.logspace(-3.0, 6.0, 120)    # grille GCV
    MIN_GAP = 0.15                   # trou minimal pour appliquer le filtre
    MIN_SIDE = 0.03                  # taille minimale de chaque mode
    SEED = 0                         # determinisme
    HUBER_C = 1.345                  # constante de Huber (1.345 = 95 % gaussien)

    # --------------------------------------------------------------- detection
    def coherence(X):
        """Pour chaque colonne : sa meilleure correlation absolue avec une autre."""
        X = np.asarray(X, dtype=np.float64)
        if len(X) == 0:
            return np.zeros(X.shape[1])
        sd = X.std(axis=0)
        Z = (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)
        C = np.abs(Z.T @ Z) / len(X)
        np.fill_diagonal(C, 0.0)
        return C.max(axis=1)

    def noise_screen(X):
        """Masque des colonnes a garder (et seuil retenu ; nan = filtre non applique).

        Seuil = point milieu du plus grand trou des scores. S'il n'y a pas de trou
        net, on garde TOUT : le filtre ne s'applique que lorsqu'il est justifie.
        """
        s = coherence(X)
        p = len(s)
        if p < 3:
            return np.ones(p, bool), float("nan")
        sv = np.sort(s)
        gaps = np.diff(sv)
        i = int(np.argmax(gaps))
        n_left, n_right = i + 1, p - i - 1
        if not (gaps[i] > MIN_GAP and n_left >= MIN_SIDE * p and n_right >= MIN_SIDE * p):
            return np.ones(p, bool), float("nan")
        return s >= 0.5 * (sv[i] + sv[i + 1]), float(0.5 * (sv[i] + sv[i + 1]))

    # ------------------------------------------------------------- estimateurs
    def ridge_gcv(As, y, weights=None):
        """Ridge ferme par SVD (pondere si weights) ; alpha par GCV."""
        if weights is None:
            weights = np.ones(len(y))
        sw = np.sqrt(weights)
        ym = float(np.average(y, weights=weights))
        Asw = As * sw[:, None]
        ysw = (y - ym) * sw
        U, s, Vt = np.linalg.svd(Asw, full_matrices=False)
        Uy = U.T @ ysw
        yc2 = float(ysw @ ysw)
        n = len(y)
        best, alpha = np.inf, ALPHAS[0]
        for a in ALPHAS:
            dh = s ** 2 / (s ** 2 + a)          # diagonale de la hat-matrix
            z = dh * Uy
            res2 = max(yc2 - 2.0 * (z @ Uy) + float(z @ z), 0.0)
            gcv = (res2 / n) / max(1.0 - dh.sum() / n, 1e-12) ** 2
            if gcv < best:
                best, alpha = gcv, a
        return Vt.T @ ((s / (s ** 2 + alpha)) * Uy), ym

    def huber_irls(As, y, iters=7):
        """IRLS de Huber : ridge pondere reajuste, echelle MAD des residus."""
        w, ym = ridge_gcv(As, y)
        for _ in range(iters):
            r = y - (As @ w + ym)
            s = 1.4826 * np.median(np.abs(r - np.median(r)))
            s = s if s > 1e-9 else 1.0
            u = np.abs(r) / s
            wt = np.where(u <= HUBER_C, 1.0, HUBER_C / np.maximum(u, 1e-12))
            w, ym = ridge_gcv(As, y, wt)
        return w, ym

    def fit_member(X, y, cols, rows, q, use_huber):
        """Un membre : winsorisation (bornes du membre) + ridge ou Huber IRLS."""
        Xs = X[np.ix_(rows, cols)]
        lo = np.quantile(Xs, q, axis=0)
        hi = np.maximum(np.quantile(Xs, 1.0 - q, axis=0), lo)
        Xc = np.clip(Xs, lo, hi)
        mu = Xc.mean(axis=0)
        sc = np.sqrt((Xc ** 2).mean(axis=0))
        sc = np.where(sc < 1e-12, 1.0, sc)
        As = (Xc - mu) / sc
        w, ym = huber_irls(As, y[rows]) if use_huber else ridge_gcv(As, y[rows])
        w = w / sc

        def member(Q):
            return (np.clip(Q[:, cols], lo, hi) - mu) @ w + ym

        return member

    # ------------------------------------------------------------------- train
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        X = X.reshape(len(y), -1)

    # 1) criblage des colonnes de bruit (structure du design, sans X_val/y_val)
    keep, tau = noise_screen(X)
    X = X[:, keep]
    n, p = X.shape

    if n == 0 or p == 0:                       # repli : predicteur constant
        ym = float(y.mean()) if len(y) else 0.0

        def predict_const(Xq):
            Xq = np.asarray(Xq, dtype=np.float64)
            return np.full(Xq.shape[0] if Xq.ndim > 1 else 1, ym)

        return predict_const

    # 2) ensemble mixte ridge + Huber, diversite de COLONNES et de lignes
    # Taille des sous-espaces : k = 64 SEULEMENT si le criblage a identifie un mode
    # de bruit. S'il s'abstient (tau = nan : aucun mode separé), il n'y a pas de
    # bruit a eviter, et un sous-espace cacherait du signal a chaque membre :
    # mesure sur un design tout-informatif 250x264 -> k=64 donne 0.18, k=p donne 0.79.
    k = K_SUBSPACE if not np.isnan(tau) else p
    k = min(k, p)
    n_rows = min(n, max(2, int(SUB_ROWS * n)))
    total = N_RIDGE + N_HUBER
    rng = np.random.default_rng(SEED)
    members = []
    for i in range(total):
        cols = rng.choice(p, k, replace=False)
        rows = rng.choice(n, n_rows, replace=False)
        q = QS[int(rng.integers(len(QS)))]
        members.append(fit_member(X, y, cols, rows, q, use_huber=i >= N_RIDGE))

    def predict(Xq):
        Xq = np.asarray(Xq, dtype=np.float64)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        Q = Xq[:, keep]
        P = np.column_stack([m(Q) for m in members])
        if TRIM > 0:                           # moyenne tronquee : coupe les queues
            lo = np.quantile(P, TRIM / 2, axis=1)
            hi = np.quantile(P, 1 - TRIM / 2, axis=1)
            P = np.clip(P, lo[:, None], hi[:, None])
        return P.mean(axis=1)

    return predict