from scipy.spatial import cKDTree
import numpy as np
from numpy.typing import NDArray
# ---------------------------------------------------------------------------
# Pre-build helper — call ONCE per curve, outside the iteration loop
# ---------------------------------------------------------------------------

def buildPolylineTree(curve_xy, k=8):
    """
    Pre-compute everything needed for fast repeated projection onto a polyline.

    Call this once before the TTM iteration loop and pass the returned
    dict to adjustBoundary / projectPointsToPolyline via `tree_data`.

    Parameters
    ----------
    curve_xy : (M, 2) array
    k        : number of nearest vertices to examine per query point

    Returns
    -------
    dict with keys: curve_xy, A, B, AB, denom, tree, k
    """
    curve_xy = np.asarray(curve_xy, dtype=float)
    M = curve_xy.shape[0]
    if M < 2:
        raise ValueError("curve_xy must have at least 2 points.")

    A     = curve_xy[:-1]
    B     = curve_xy[1:]
    AB    = B - A
    denom = np.einsum("ij,ij->i", AB, AB)   # |AB|^2  (M-1,)

    return {
        "curve_xy": curve_xy,
        "A":     A,
        "B":     B,
        "AB":    AB,
        "denom": denom,
        "tree":  cKDTree(curve_xy),
        "k":     min(int(k), M),
    }


# ---------------------------------------------------------------------------
# Vectorised projection
# ---------------------------------------------------------------------------

def projectPointsToPolyline(P, curve_xy, k=8, tree_data=None):
    """
    Project N query points onto a dense polyline.

    Parameters
    ----------
    P          : (N, 2) array  — query points
    curve_xy   : (M, 2) array  — polyline vertices (ignored when tree_data given)
    k          : nearest-vertex count (ignored when tree_data given)
    tree_data  : dict from buildPolylineTree(), or None to build on the fly

    Returns
    -------
    Q       : (N, 2)  projected points
    seg_idx : (N,)    index of best segment
    dist2   : (N,)    squared projection distance
    """
    P = np.asarray(P, dtype=float)
    N = P.shape[0]

    if tree_data is None:
        tree_data = buildPolylineTree(curve_xy, k=k)

    A     = tree_data["A"]
    AB    = tree_data["AB"]
    denom = tree_data["denom"]
    tree  = tree_data["tree"]
    k_eff = tree_data["k"]
    M1    = A.shape[0]

    # Find k nearest vertices per query point
    _, v_idx = tree.query(P, k=k_eff)           # (N, k_eff)
    if v_idx.ndim == 1:
        v_idx = v_idx[:, None]

    # Expand to candidate segments: vertex v -> segments (v-1) and v
    v_idx_exp = np.stack([v_idx - 1, v_idx], axis=2).reshape(N, -1)  # (N, 2k)
    v_idx_exp = np.clip(v_idx_exp, 0, M1 - 1)

    # Gather segment geometry for all candidates
    A_c     = A[v_idx_exp]                        # (N, 2k, 2)
    AB_c    = AB[v_idx_exp]                       # (N, 2k, 2)
    denom_c = denom[v_idx_exp]                    # (N, 2k)

    # Closest-point parameter t, clamped to [0, 1]
    PA = P[:, None, :] - A_c                      # (N, 2k, 2)
    t  = np.einsum("nci,nci->nc", PA, AB_c)       # (N, 2k)
    safe_denom = np.where(denom_c > 0.0, denom_c, 1.0)
    t  = np.clip(t / safe_denom, 0.0, 1.0)

    Qc = A_c + t[:, :, None] * AB_c              # (N, 2k, 2)
    d2 = np.sum((P[:, None, :] - Qc) ** 2, axis=2)  # (N, 2k)
    d2 = np.where(denom_c > 0.0, d2, np.inf)

    best    = np.argmin(d2, axis=1)              # (N,)
    n_idx   = np.arange(N)
    Q       = Qc[n_idx, best]
    seg_idx = v_idx_exp[n_idx, best]
    dist2   = d2[n_idx, best]

    return Q, seg_idx, dist2