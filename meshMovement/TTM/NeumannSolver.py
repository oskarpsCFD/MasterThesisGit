import numpy as np
from numpy.typing import NDArray

from TTM.HelperFunctions import (
    createAlphaGamma_bnd,
    precompute_left_sides_bnd,
    createRightSide_neumann,
    TDMA_banded,
)
from TTM.BoundaryProjection import project_points_to_polyline


def compute_fixed_indices(neumann: NDArray, Nj: int, Ni: int,
                          frac_airfoil: float,
                          stitchBoundaryDict: dict | None = None) -> dict:
    """
    Compute the set of i (or j) indices on each boundary that must remain
    fixed — never modified by the Neumann solve.

    These are:
      - The two end-corners of every boundary (always fixed).
      - For the bottom boundary only: the two trailing-edge points where
        the wake cut meets the airfoil.
      - For any boundary present in stitchBoundaryDict: the four stitch
        anchor indices (left block start/end + right block start/end).
    """
    stitch = stitchBoundaryDict or {}

    def stitch_cols(name):
        arr = stitch[name]
        return {int(r[c]) % Ni for r in arr for c in (0, 1)}

    def stitch_rows(name):
        arr = stitch[name]
        return {int(r[c]) % Nj for r in arr for c in (0, 1)}

    fixed = {}

    if neumann[0]:
        N_air = int(round(frac_airfoil * Ni))
        if N_air % 2 == 0:
            N_air += 1
        N_wake = (Ni + 2) - N_air
        N_wL = N_wake // 2
        i_te_lower = N_wL - 1
        i_te_upper = N_wL + N_air - 2
        fixed[0] = {0, i_te_lower, i_te_upper, Ni - 1}
        if "bot" in stitch:
            fixed[0] |= stitch_cols("bot")

    if neumann[1]:
        fixed[1] = {0, Ni - 1}
        if "top" in stitch:
            fixed[1] |= stitch_cols("top")

    if neumann[2]:
        fixed[2] = {0, Nj - 1}
        if "left" in stitch:
            fixed[2] |= stitch_rows("left")

    if neumann[3]:
        fixed[3] = {0, Nj - 1}
        if "right" in stitch:
            fixed[3] |= stitch_rows("right")

    return fixed


def calculateLi(P: NDArray) -> NDArray:
    edges = P[1:] - P[:-1]
    edge_lengths = np.sqrt(np.sum(edges**2, axis=1))
    li = np.zeros(len(P))
    li[1:-1] = 0.5 * (edge_lengths[:-1] + edge_lengths[1:])
    li[0]    = 0.5 * edge_lengths[0]
    li[-1]   = 0.5 * edge_lengths[-1]
    return li


def _pava_nondecreasing(s: NDArray) -> NDArray:
    """Pool Adjacent Violators: closest non-decreasing sequence under L2."""
    s = np.asarray(s, dtype=float)
    blocks: list = [[float(v), 1] for v in s]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            n = blocks[i][1] + blocks[i + 1][1]
            m = (blocks[i][0] * blocks[i][1]
                 + blocks[i + 1][0] * blocks[i + 1][1]) / n
            blocks[i] = [m, n]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out: list = []
    for mean, count in blocks:
        out.extend([mean] * count)
    return np.array(out, dtype=float)


def _uncross_segment(P_corr: NDArray, P_old: NDArray) -> NDArray:
    """Ensure corrected positions keep the same arc-length order as P_old."""
    M = len(P_old)
    if M < 3:
        return P_corr

    edges    = P_old[1:] - P_old[:-1]
    elen     = np.maximum(np.linalg.norm(edges, axis=1, keepdims=True), 1e-12)
    tangents = edges / elen

    T        = np.zeros_like(P_old)
    T[1:-1]  = 0.5 * (tangents[:-1] + tangents[1:])
    T[0]     = tangents[0]
    T[-1]    = tangents[-1]
    T       /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)

    s_old = np.concatenate([[0.0], np.cumsum(elen.ravel())])
    s_new = s_old + np.einsum('ij,ij->i', P_corr - P_old, T)

    if np.all(np.diff(s_new) >= 0.0):
        return P_corr

    s_fixed  = _pava_nondecreasing(s_new)
    seg_lens = np.diff(s_old)
    s_k      = np.clip(s_fixed, s_old[0], s_old[-1])
    j_idx    = np.clip(np.searchsorted(s_old, s_k, side='right') - 1, 0, M - 2)
    frac     = np.clip((s_k - s_old[j_idx]) / (seg_lens[j_idx] + 1e-30), 0.0, 1.0)
    return P_old[j_idx] + frac[:, None] * (P_old[j_idx + 1] - P_old[j_idx])


def limitAverageVelocitySegments(P_new: NDArray, P_old: NDArray,
                                  i0: int = 0, i1: int = -1,
                                  K: int = 2, tol: float = 1e-6) -> tuple:
    P_newCalc, P_oldCalc = P_new[i0:i1], P_old[i0:i1]
    if K == 0:
        return P_new, None
    N = len(P_oldCalc)

    stationary_mask = np.zeros(N, dtype=bool)
    segments = np.array_split(np.arange(N), K)
    P_corrected = P_newCalc.copy()

    for idx in segments:
        P_seg_new = P_corrected[idx]
        P_seg_old = P_oldCalc[idx]

        if len(idx) < 2:
            continue

        li      = calculateLi(P_seg_old)
        delta_P = P_seg_new - P_seg_old

        edges        = P_seg_old[1:] - P_seg_old[:-1]
        edge_lengths = np.maximum(np.sqrt(np.sum(edges**2, axis=1, keepdims=True)), 1e-12)
        tangents     = edges / edge_lengths

        T        = np.zeros_like(P_seg_old)
        T[1:-1]  = 0.5 * (tangents[:-1] + tangents[1:])
        T[0]     = tangents[0]
        T[-1]    = tangents[-1]
        T       /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)

        delta_tan_scalar = np.sum(delta_P * T, axis=1)

        moving = np.abs(delta_tan_scalar) >= tol
        if not np.any(moving):
            stationary_mask[idx] = True
            P_corrected[idx] = P_seg_old
            continue

        li_sum = np.sum(li[moving])
        delta_tan_bar = np.sum(delta_tan_scalar[moving] * li[moving]) / (li_sum + 1e-6)
        P_corrected[idx] = P_seg_new - delta_tan_bar * T

        if len(idx) >= 3:
            P_corrected[idx] = _uncross_segment(P_corrected[idx], P_seg_old)

        stationary = np.abs(delta_tan_scalar) < tol
        P_corrected[idx[stationary]] = P_seg_old[stationary]
        stationary_mask[idx[stationary]] = True

    # Force junction nodes to their average when they share a degenerate (zero-length) edge
    for k in range(len(segments) - 1):
        if len(segments[k]) == 0 or len(segments[k + 1]) == 0:
            continue
        j_last  = segments[k][-1]
        j_first = segments[k + 1][0]
        if np.linalg.norm(P_oldCalc[j_last] - P_oldCalc[j_first]) < 1e-8:
            avg = 0.5 * (P_corrected[j_last] + P_corrected[j_first])
            P_corrected[j_last]  = avg
            P_corrected[j_first] = avg

    P_corrected = _uncross_segment(P_corrected, P_oldCalc)
    P_new[i0:i1] = P_corrected
    return P_new, stationary_mask


def solve_neumann_row(X: NDArray, Y: NDArray,
                      dxi: float, deta: float,
                      weight: NDArray,
                      j: int,
                      fixed: set,
                      omega: float,
                      helperCurves: dict,
                      side_name: str,
                      tree_data,
                      stitch_pairs=None) -> float:
    fixed_list = sorted(fixed)
    X_fixed = X[j, fixed_list].copy()
    Y_fixed = Y[j, fixed_list].copy()
    X_old   = X[j, 1:-1].copy()
    Y_old   = Y[j, 1:-1].copy()

    alpha_b, gamma_b, _ = createAlphaGamma_bnd(X, Y, dxi, deta, j)
    a_b, b_b, c_b = precompute_left_sides_bnd(alpha_b, gamma_b, weight)

    dX = createRightSide_neumann(gamma_b, X, j)
    dY = createRightSide_neumann(gamma_b, Y, j)
    dX[0]  -= a_b[0]  * X[j, 0];  dX[-1] -= c_b[-1] * X[j, -1]
    dY[0]  -= a_b[0]  * Y[j, 0];  dY[-1] -= c_b[-1] * Y[j, -1]

    xj = TDMA_banded(a_b[1:], b_b, c_b[:-1], dX)
    yj = TDMA_banded(a_b[1:], b_b, c_b[:-1], dY)

    x_new = (1 - omega) * X_old + omega * xj
    y_new = (1 - omega) * Y_old + omega * yj

    if stitch_pairs:
        Ni_loc = X.shape[1]
        for ig0, ig1 in stitch_pairs:
            k0 = int(ig0) % Ni_loc - 1
            k1 = int(ig1) % Ni_loc - 1
            if 0 <= k0 < len(x_new) and 0 <= k1 < len(x_new):
                avg_x = 0.5 * (x_new[k0] + x_new[k1])
                avg_y = 0.5 * (y_new[k0] + y_new[k1])
                x_new[k0] = x_new[k1] = avg_x
                y_new[k0] = y_new[k1] = avg_y

    P_old_stack = np.column_stack((X_old, Y_old))
    P_new_stack = np.column_stack((x_new, y_new))

    # Airfoil arc = largest gap between consecutive fixed-node indices
    if len(fixed_list) >= 2:
        gaps    = [fixed_list[k + 1] - fixed_list[k] for k in range(len(fixed_list) - 1)]
        gap_idx = int(np.argmax(gaps))
        i0 = fixed_list[gap_idx]
        i1 = fixed_list[gap_idx + 1]
    else:
        i0 = 0
        i1 = X.shape[1] - 1

    P_corrected, stationary_mask = limitAverageVelocitySegments(P_new_stack, P_old_stack, i0, i1)
    x_new = P_corrected[:, 0]
    y_new = P_corrected[:, 1]

    max_delta = max(np.max(np.abs(x_new - X_old)),
                    np.max(np.abs(y_new - Y_old)))

    X[j, 1:-1] = x_new
    Y[j, 1:-1] = y_new

    xc, yc = helperCurves[side_name]
    Q, _, _ = project_points_to_polyline(
        np.column_stack((X[j, 1:-1], Y[j, 1:-1])),
        np.column_stack((xc, yc)),
        tree_data=tree_data)
    X[j, 1:-1] = Q[:, 0]
    Y[j, 1:-1] = Q[:, 1]

    for k, i in enumerate(fixed_list):
        X[j, i] = X_fixed[k]
        Y[j, i] = Y_fixed[k]

    # Post-restoration crossing repair on the airfoil arc
    if len(fixed_list) >= 2 and i1 > i0 + 2:
        ps_lo = i0 - 1
        ps_hi = i1 - 1
        if ps_hi > ps_lo + 2:
            P_fixed_rgn = _uncross_segment(
                np.column_stack((X[j, i0:i1 + 1], Y[j, i0:i1 + 1])),
                P_old_stack[ps_lo:ps_hi + 1])
            X[j, i0 + 1:i1] = P_fixed_rgn[1:-1, 0]
            Y[j, i0 + 1:i1] = P_fixed_rgn[1:-1, 1]

    # Restore nodes that were stationary before projection and crossing repair
    if stationary_mask is not None:
        for k, is_stat in enumerate(stationary_mask):
            if is_stat:
                abs_int = i0 + k
                X[j, abs_int + 1] = X_old[abs_int]
                Y[j, abs_int + 1] = Y_old[abs_int]

    return max_delta


def solve_neumann_col(X: NDArray, Y: NDArray,
                      dxi: float, deta: float,
                      weight: NDArray,
                      i_col: int,
                      fixed: set,
                      omega: float,
                      helperCurves: dict,
                      side_name: str,
                      tree_data,
                      stitch_pairs=None) -> float:
    return solve_neumann_row(X.T, Y.T, deta, dxi, weight, i_col,
                             fixed, omega, helperCurves, side_name,
                             tree_data, stitch_pairs)


def _make_boundary_configs(Nj, Ni, phi, psi,
                           phi_bot=None, phi_top=None,
                           psi_left=None, psi_right=None,
                           omega_bnd_bot=None, omega_bnd_top=None,
                           omega_bnd_left=None, omega_bnd_right=None):
    return [
        dict(side="bot",   is_row=True,  j_or_i=0,      weight=phi_bot   if phi_bot   is not None else phi[0,  :], omega_bnd=omega_bnd_bot),
        dict(side="top",   is_row=True,  j_or_i=Nj - 1, weight=phi_top   if phi_top   is not None else phi[-1, :], omega_bnd=omega_bnd_top),
        dict(side="left",  is_row=False, j_or_i=0,       weight=psi_left  if psi_left  is not None else psi[:,  0], omega_bnd=omega_bnd_left),
        dict(side="right", is_row=False, j_or_i=Ni - 1,  weight=psi_right if psi_right is not None else psi[:, -1], omega_bnd=omega_bnd_right),
    ]


def _apply_neumann_boundaries(X, Y, dxi, deta, configs,
                               neumann, has_curves, helperCurves,
                               fixed_indices, tree_cache, omega,
                               stitchBoundaryDict=None):
    stitch = stitchBoundaryDict or {}
    max_delta = 0.0
    for idx, cfg in enumerate(configs):
        if not neumann[idx]:
            continue
        if not has_curves or cfg["side"] not in helperCurves:
            continue

        omega_eff = cfg.get("omega_bnd") or omega

        side = cfg["side"]
        stitch_pairs = None
        if side in stitch:
            arr = stitch[side]
            if len(arr) > 2:
                stitch_pairs = [(int(a), int(b)) for a, b in arr[1:-1]]

        kwargs = dict(
            X=X, Y=Y, dxi=dxi, deta=deta,
            weight=cfg["weight"],
            fixed=fixed_indices[idx],
            omega=omega_eff,
            helperCurves=helperCurves,
            side_name=cfg["side"],
            tree_data=tree_cache.get(idx),
            stitch_pairs=stitch_pairs,
        )

        if cfg["is_row"]:
            d = solve_neumann_row(**kwargs, j=cfg["j_or_i"])
        else:
            d = solve_neumann_col(**kwargs, i_col=cfg["j_or_i"])

        if d > max_delta:
            max_delta = d

    return max_delta
