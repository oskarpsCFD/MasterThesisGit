import numpy as np
from numpy.typing import NDArray

from TTM.HelperFunctions import (
    createAlphaGammaBnd,
    precomputeLeftSidesBnd,
    createRightSideNeumann,
    TDMABanded,
)
from TTM.BoundaryProjection import projectPointsToPolyline


def computeFixedIndices(neumann: NDArray, Nj: int, Ni: int,
                          frac_airfoil: float,
                          stitchBoundaryDict: dict | None = None) -> dict:
    """
    Compute the set of i (or j) indices on each boundary that must remain
    fixed — never modified by the Neumann solve.

    These are:
      - The two end-corners of every boundary (always fixed).
      - For any boundary present in stitchBoundaryDict: the four stitch
        anchor indices (left block start/end + right block start/end).

    Parameters
    ----------
    neumann           : (4,) bool array  [bot, top, left, right]
    Nj, Ni            : grid dimensions
    frac_airfoil      : fraction of Ni points that lie on the airfoil surface
    stitchBoundaryDict: optional dict mapping side name ('bot','top','left',
                        'right') to stitch array of shape (M, 2).
                        Each row is [left_idx, right_idx]; right indices are
                        negative and resolved via % Ni (or % Nj for columns).

    Returns
    -------
    dict mapping boundary index (0-3) -> set of fixed integer indices
    """
    stitch = stitchBoundaryDict or {}

    def stitch_cols(name):
        """Absolute column indices of all stitched pairs on a row boundary."""
        arr = stitch[name]
        return {int(r[c]) % Ni for r in arr for c in (0, 1)}

    def stitch_rows(name):
        """Absolute row indices of all stitched pairs on a column boundary."""
        arr = stitch[name]
        return {int(r[c]) % Nj for r in arr for c in (0, 1)}

    fixed = {}

    # bot (index 0): corners + trailing-edge kinks + stitch anchors
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

    # top (index 1): corners + stitch anchors
    if neumann[1]:
        fixed[1] = {0, Ni - 1}
        if "top" in stitch:
            fixed[1] |= stitch_cols("top")

    # left (index 2): corners + stitch anchors
    if neumann[2]:
        fixed[2] = {0, Nj - 1}
        if "left" in stitch:
            fixed[2] |= stitch_rows("left")

    # right (index 3): corners + stitch anchors
    if neumann[3]:
        fixed[3] = {0, Nj - 1}
        if "right" in stitch:
            fixed[3] |= stitch_rows("right")

    return fixed


# ---------------------------------------------------------------------------
# Row solve  (bottom j=0,  top j=Nj-1)
# ---------------------------------------------------------------------------

def solveNeumannRow(X: NDArray, Y: NDArray,
                      dxi: float, deta: float,
                      weight: NDArray,
                      j: int,
                      fixed: set,
                      omega: float,
                      helperCurves: dict,
                      side_name: str,
                      tree_data) -> float:
    """
    Solve one Neumann boundary row in-place and project onto the curve.

    The ghost-cell symmetry (∂P/∂η = 0) gives the RHS:
        RHS = -gamma * 2 * P[neighbour_row, 1:-1]

    The tridiagonal coefficients use only phi (not psi).  The psi forcing
    term disappears because  P_jp - P_jm = 0  under ghost substitution.

    The system sweeps over i = 1 … Ni-2  (the Ni-2 interior columns of the
    boundary row).  Columns i=0 and i=Ni-1 are always Dirichlet corners;
    their contribution is subtracted from the first and last RHS entries.

    Parameters
    ----------
    X, Y        : full grid  (Nj, Ni),  modified in-place
    dxi, deta   : logical-space step sizes
    phi_row     : (Ni-2,)  phi weight at the innermost interior row
                  adjacent to this boundary  (phi[0,:] for bot,
                  phi[-1,:] for top)
    j           : 0 (bottom) or Nj-1 (top)
    fixed_i     : set of i-indices that must not move (corners + TE kinks)
    omega       : SOR relaxation factor
    helperCurves: dict with 'bot'/'top' keys pointing to (xc, yc) arrays
    side_name   : 'bot' or 'top'
    tree_data   : pre-built KDTree dict for the curve

    Returns
    -------
    max_delta : float — maximum point displacement this row
    """
    Ni = X.shape[1]

    # Save positions of fixed points before the solve
    fixed_list = sorted(fixed)
    X_fixed = X[j, fixed_list].copy()
    Y_fixed = Y[j, fixed_list].copy()

    # Metric coefficients at the boundary row  (Ni-2,) each
    alpha_b, gamma_b, _ = createAlphaGammaBnd(X, Y, dxi, deta, j)

    # Tridiagonal coefficients using only phi  (psi does NOT appear here)
    #   a = alpha * (1 - phi/2)
    #   b = -2*(alpha + gamma)
    #   c = alpha * (1 + phi/2)
    a_b, b_b, c_b = precomputeLeftSidesBnd(alpha_b, gamma_b, weight)

    # RHS via ghost-cell Neumann:  -gamma * 2 * P[neighbour_row]
    dX = createRightSideNeumann(gamma_b, X, j)
    dY = createRightSideNeumann(gamma_b, Y, j)

    # Subtract Dirichlet corner contributions from the boundary RHS
    # (i=0 and i=Ni-1 are always fixed — they are not unknowns)
    dX[0]  -= a_b[0]   * X[j, 0]
    dX[-1] -= c_b[-1]  * X[j, -1]
    dY[0]  -= a_b[0]   * Y[j, 0]
    dY[-1] -= c_b[-1]  * Y[j, -1]

    # TDMA solve over i = 1 … Ni-2
    a_sub = a_b[1:]
    c_sup = c_b[:-1]

    xj = TDMABanded(a_sub, b_b, c_sup, dX)
    yj = TDMABanded(a_sub, b_b, c_sup, dY)

    x_new = (1 - omega) * X[j, 1:-1] + omega * xj
    y_new = (1 - omega) * Y[j, 1:-1] + omega * yj

    max_delta = max(np.max(np.abs(x_new - X[j, 1:-1])),
                    np.max(np.abs(y_new - Y[j, 1:-1])))

    X[j, 1:-1] = x_new
    Y[j, 1:-1] = y_new

    # Project all sliding points onto the boundary curve
    xc, yc = helperCurves[side_name]
    curve_xy = np.column_stack((xc, yc))
    P_bnd = np.column_stack((X[j, 1:-1], Y[j, 1:-1]))
    Q, _, _ = projectPointsToPolyline(P_bnd, curve_xy, tree_data=tree_data)
    X[j, 1:-1] = Q[:, 0]
    Y[j, 1:-1] = Q[:, 1]

    # Restore fixed points (corners + trailing-edge kinks)
    for k, i in enumerate(fixed_list):
        X[j, i] = X_fixed[k]
        Y[j, i] = Y_fixed[k]

    return max_delta


# ---------------------------------------------------------------------------
# Column solve  (left i=0,  right i=Ni-1)
# ---------------------------------------------------------------------------

def solveNeumannCol(X: NDArray, Y: NDArray,
                      dxi: float, deta: float,
                      weight: NDArray,
                      i_col: int,
                      fixed: set,
                      omega: float,
                      helperCurves: dict,
                      side_name: str,
                      tree_data) -> float:

    return solveNeumannRow(X.T, Y.T,
                           dxi, deta,
                           weight, i_col,
                           fixed,omega,
                           helperCurves,side_name,
                           tree_data)


def _makeBoundaryConfigs(Nj, Ni, phi, psi):
    """
    Return a list of four BoundaryConfig dicts, one per side.

    Each dict contains every constant needed to call the correct solver
    and is indexed 0=bot, 1=top, 2=left, 3=right — matching the neumann/snap
    arrays throughout the rest of the code.

    Fields
    ------
    side       : str    — key into helperCurves / tree_cache
    is_row     : bool   — True → solveNeumannRow, False → solveNeumannCol
    j_or_i     : int    — j for rows, i_col for columns
    weight     : ndarray — phi slice (rows) or psi slice (cols)
                  Rows sweep along ξ  → phi weights ξ-direction attraction
                  Cols sweep along η  → psi weights η-direction attraction
    """
    return [
        # bot  (j=0)     — row,    phi from innermost interior row
        dict(side="bot",   is_row=True,  j_or_i=0,      weight=phi[0,  :]),
        # top  (j=Nj-1)  — row,    phi from innermost interior row
        dict(side="top",   is_row=True,  j_or_i=Nj - 1, weight=phi[-1, :]),
        # left (i=0)     — column, psi from innermost interior col
        dict(side="left",  is_row=False, j_or_i=0,      weight=psi[:,  0]),
        # right(i=Ni-1)  — column, psi from innermost interior col
        dict(side="right", is_row=False, j_or_i=Ni - 1, weight=psi[:, -1]),
    ]


def _applyNeumannBoundaries(X, Y, dxi, deta, configs,
                               neumann, has_curves, helperCurves,
                               fixed_indices, tree_cache, omega):
    """
    Apply Neumann BC solves for all active boundaries and return max_delta.

    Iterates over the four boundary configs; for each side that is both
    enabled in `neumann` and has a curve available, dispatches to the
    correct row or column solver.
    """
    max_delta = 0.0
    for idx, cfg in enumerate(configs):
        if not neumann[idx]:
            continue
        if not has_curves or cfg["side"] not in helperCurves:
            continue
        kwargs = dict(
            X=X, Y=Y, dxi=dxi, deta=deta,
            weight=cfg["weight"],
            fixed=fixed_indices[idx],
            omega=omega,
            helperCurves=helperCurves,
            side_name=cfg["side"],
            tree_data=tree_cache.get(idx),
        )

        if cfg["is_row"]:
            d = solveNeumannRow(**kwargs, j=cfg["j_or_i"])
        else:
            d = solveNeumannCol(**kwargs, i_col=cfg["j_or_i"])

        if d > max_delta:
            max_delta = d

    return max_delta
