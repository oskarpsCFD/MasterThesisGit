"""
TTMSolver.py
----------------
Entry-point for the TTM elliptic mesh smoother.
Helper routines are in their own modules:
  HelperFunctions.py  — metric coefficients, tridiagonal builders, TDMA
  NeumannSolver.py    — ghost-cell Neumann BC (boundary row/col solves)
  BoundaryProjection.py — KDTree construction, curve projection, snap
"""
import numpy as np
from numpy.typing import NDArray

from TTM.HelperFunctions import (
    createAlphaGamma,
    precomputeLeftSides,
    buildTreeCache,
    SolveRow,
)


from TTM.NeumannSolver import (
    computeFixedIndices,
    _applyNeumannBoundaries,
    _makeBoundaryConfigs,
)


def TTMSolver(X: NDArray, Y: NDArray,
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
              te_clustering: float = 1.0,
              mode: str = "XY") -> tuple:
    """
    TTM elliptic mesh smoother with optional Neumann boundary conditions.
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
                   "XY" — solve both X and Y (default, standard TTM).
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

    fixed_indices = computeFixedIndices(neumann, Nj, Ni, frac_Geometry, stichedBoundaryDict)
    bnd_configs = _makeBoundaryConfigs(Nj, Ni, phi, psi)

    for it in range(maxIter):
        max_delta = 0.0

        alpha, gamma, beta = createAlphaGamma(X, Y, dxi, deta)
        a_full, b_all, c_full = precomputeLeftSides(alpha, gamma, phi)

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
            d = _applyNeumannBoundaries(
                X, Y, dxi, deta, bnd_configs,
                neumann, has_curves, helperCurves,
                fixed_indices, tree_cache, omega)
            if d > max_delta:       # ← inside the if, uses fresh d
                max_delta = d

        r = max_delta
        if r < tol:
            print(f"Mesh converged after {it} iterations")
            break

    return X, Y
