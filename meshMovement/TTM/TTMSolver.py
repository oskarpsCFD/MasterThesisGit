"""
TTMSolver.py
----------------
Entry-point for the Winslow elliptic mesh smoother.
Helper routines are in their own modules:
  HelperFunctions.py  — metric coefficients, tridiagonal builders, TDMA
  NeumannSolver.py    — ghost-cell Neumann BC (boundary row/col solves)
  BoundaryProjection.py — KDTree construction, curve projection, snap
"""
import numpy as np
from numpy.typing import NDArray

from TTM.HelperFunctions import (
    createAlphaGamma,
    createRightSide,
    precompute_left_sides,
    TDMA_banded,
    buildTreeCache,
    SolveRow,
)

from TTM.BoundaryProjection import (
    build_polyline_tree,
    generateBoundaryPhiPsi,
)

from TTM.NeumannSolver import (
    compute_fixed_indices,
    _apply_neumann_boundaries,
    _make_boundary_configs,
)




def WinslowSolver(X: NDArray, Y: NDArray,
                  dxi: float, deta: float,
                  psi: NDArray, phi: NDArray,
                  frac: NDArray = np.array([1,1,1,1]),
                  tol: float = 5e-3,
                  maxIter: int = 1000,
                  helperCurves=None,
                  neumann=None,
                  neumann_iters=100,
                  omega: float = 1.0,
                  stichedBoundaryDict: dict[NDArray] | None = None,
                  mode: str = "XY",
                  phi_psi_updater=None,
                  recompute_interval: int = 50,
                  w1: NDArray | None = None,
                  w2: NDArray | None = None,
                  neumann_bot_fraction: float = 1.0) -> tuple:
    """
    Winslow elliptic mesh smoother with optional Neumann boundary conditions.
    Parameters
    ----------
    X, Y         : (Nj, Ni) physical grid coordinates, modified in-place
    dxi, deta    : logical-space step sizes
    psi          : (Nj-2, Ni-2) eta-direction attraction weight
                   — used in the interior RHS forcing term only.
                   NOT used in Neumann boundary solves (the forcing term
                   vanishes under ghost-cell substitution).
    phi          : (Nj-2, Ni-2) xi-direction attraction weight
                   — used in both interior and Neumann boundary tridiagonals.
    neumann      : (4,) bool array [bot, top, left, right].
                   If True for a side, that boundary row/col is solved with
                   a ghost-cell Neumann BC (dP/dn = 0), allowing points to
                   slide freely along the curve.
    frac_airfoil : fraction of Ni points on the airfoil surface.
                   Required when neumann[0] is True so that trailing-edge
                   fixed indices can be computed.
    neumann_iters: number of iterations for which the Neumann BC is active.
    snap         : (4,) bool array — legacy Dirichlet snap.
    omega        : SOR relaxation factor (1.0 = no over-relaxation).
    mode         : which coordinate(s) to update in the interior sweep.
                   "XY" — solve both X and Y (default, standard Winslow).
                   "X"  — solve X only; Y is held fixed at its current values.
                          Use for rectangular domains where Y = ETA exactly:
                          with psi ≠ 0 the Y-equation has no equilibrium at
                          Y = ETA, so solving it permanently perturbs Y and
                          prevents X from converging (see Anderson 1987).
                   "Y"  — solve Y only; X is held fixed at its current values.
                          Symmetric counterpart for eta-direction clustering.
    """


    if mode not in ("XY", "X", "Y"):
        raise ValueError(f"mode must be 'XY', 'X', or 'Y'; got {mode!r}")
    solve_X = mode in ("XY", "X")
    solve_Y = mode in ("XY", "Y")

    if neumann is None:
        neumann = np.array([False, False, False, False])

    sides      = np.array(["bot", "top", "left", "right"])
    has_curves = helperCurves is not None and len(helperCurves) > 0
    Nj, Ni     = X.shape

    # Pre-build KDTree for every boundary that needs snapping or Neumann
    tree_cache = buildTreeCache(has_curves, neumann, sides, helperCurves)

    # Fixed indices per Neumann boundary
    frac_Geometry = 1.0
    for i in frac:
        if i != 1.0:
            frac_Geometry = i
            break

    fixed_indices = compute_fixed_indices(neumann, Nj, Ni,
                                          frac_Geometry,
                                          stitchBoundaryDict=stichedBoundaryDict)

    # Build a per-node omega array for the bottom Neumann boundary when the
    # mesh has an airfoil + wake-cut topology (frac < 1).  The entire stitch
    # region (both wake-cut halves + airfoil) uses the Neumann BC; stitch-pair
    # averaging in solve_neumann_row keeps both wake-cut sides equal after the
    # TDMA step.  neumann_bot_fraction controls the active fraction of the
    # airfoil arc; the wake-cut segments are always fully active.
    omega_bnd_bot = None
    if neumann[0] and frac_Geometry < 1.0:
        N_air = int(round(frac_Geometry * Ni))
        if N_air % 2 == 0:
            N_air += 1
        N_wake     = (Ni + 2) - N_air
        N_wL       = N_wake // 2
        i_te_lower = N_wL - 1          # global column index on the bottom row
        i_te_upper = N_wL + N_air - 2  # global column index on the bottom row
        # Interior array (Ni-2,): interior index k  ↔  global column k+1.
        n_airfoil = i_te_upper - i_te_lower
        n_active  = max(1, int(round(neumann_bot_fraction * n_airfoil)))
        omega_bnd_bot = np.zeros(Ni - 2)
        # Left wake cut (columns 1..i_te_lower-1, interior 0..i_te_lower-2)
        omega_bnd_bot[:i_te_lower - 1] = omega
        # Airfoil surface (neumann_bot_fraction fraction of arc from i_te_lower)
        lo = max(0, i_te_lower - 1)
        hi = min(Ni - 2, lo + n_active)
        omega_bnd_bot[lo:hi] = omega
        # Right wake cut (columns i_te_upper+1..Ni-2, interior i_te_upper..Ni-3)
        omega_bnd_bot[i_te_upper:] = omega

    # Compute boundary-specific phi/psi from the weight matrices at the
    # boundary rows/cols so the Neumann solve enforces d/dxi(w·dS/dxi)=0
    # with w evaluated at the boundary, not at the first interior row.
    phi_bot_bnd = phi_top_bnd = psi_left_bnd = psi_right_bnd = None
    if w1 is not None and w2 is not None:
        phi_bot_bnd, phi_top_bnd, psi_left_bnd, psi_right_bnd = \
            generateBoundaryPhiPsi(w1, w2, dxi, deta)

    bnd_configs = _make_boundary_configs(Nj, Ni, phi, psi,
                                         phi_bot=phi_bot_bnd,
                                         phi_top=phi_top_bnd,
                                         psi_left=psi_left_bnd,
                                         psi_right=psi_right_bnd,
                                         omega_bnd_bot=omega_bnd_bot)

    for it in range(maxIter):
        max_delta = 0.0

        if phi_psi_updater is not None and it > 0 and it % recompute_interval == 0 and it < neumann_iters/2:
            phi, psi = phi_psi_updater(X, Y)
            bnd_configs = _make_boundary_configs(Nj, Ni, phi, psi,
                                                  phi_bot=phi_bot_bnd,
                                                  phi_top=phi_top_bnd,
                                                  psi_left=psi_left_bnd,
                                                  psi_right=psi_right_bnd,
                                                  omega_bnd_bot=omega_bnd_bot)

        alpha, gamma, beta = createAlphaGamma(X, Y, dxi, deta)
        a_full, b_all, c_full = precompute_left_sides(alpha, gamma, phi)

        for j in range(1, Nj - 1):
            # Compute relevant values to solve
            j_int = j - 1
            aj = a_full[j_int]
            bj = b_all [j_int]
            cj = c_full[j_int]
            a_sub = aj[1:]
            c_sup = cj[:-1]

            # Solve row in x direction
            if solve_X:
                max_delta = SolveRow(X, beta, gamma, psi, j, omega,
                                     aj, bj, cj, a_sub, c_sup, max_delta)

            # Solve row in y direction
            if solve_Y:
                max_delta = SolveRow(Y, beta, gamma, psi, j, omega,
                                     aj, bj, cj, a_sub, c_sup, max_delta)



        if it < neumann_iters:
            d = _apply_neumann_boundaries(
                X, Y, dxi, deta, bnd_configs,
                neumann, has_curves, helperCurves,
                fixed_indices, tree_cache, omega,
                stitchBoundaryDict=stichedBoundaryDict)
            if d > max_delta:
                max_delta = d

        r = max_delta
        if r < tol:
            print(f"Mesh converged after {it} iterations")
            break

    return X, Y
