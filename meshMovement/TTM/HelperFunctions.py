from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded


# ---------------------------------------------------------------------------
# Alpha / Gamma / Beta  (unchanged — must stay per-iteration as X,Y evolve)
# ---------------------------------------------------------------------------

def createAlphaGamma(X: NDArray, Y: NDArray, dxi: float, deta: float
                     ) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compute metric coefficients on the interior of the grid.
    Returns alpha, gamma, beta — all shape (Nj-2, Ni-2).
    """
    x_xi  = (X[1:-1, 2:] - X[1:-1, :-2]) / (2 * dxi)
    y_xi  = (Y[1:-1, 2:] - Y[1:-1, :-2]) / (2 * dxi)
    gamma = x_xi ** 2 + y_xi ** 2

    x_eta = (X[2:, 1:-1] - X[:-2, 1:-1]) / (2 * deta)
    y_eta = (Y[2:, 1:-1] - Y[:-2, 1:-1]) / (2 * deta)
    alpha = x_eta ** 2 + y_eta ** 2

    beta = x_xi * x_eta + y_xi * y_eta
    return alpha, gamma, beta


def createAlphaGamma_bnd(X: NDArray, Y: NDArray,
                          dxi: float, deta: float,
                          j: int) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Metric coefficients at a boundary row j, shape (Ni-2,) each.

    xi-derivative  : central difference along the boundary row.
    eta-derivative : one-sided (forward at j=0, backward at j=Nj-1).
    """
    Nj = X.shape[0]

    x_xi = (X[j, 2:] - X[j, :-2]) / (2 * dxi)
    y_xi = (Y[j, 2:] - Y[j, :-2]) / (2 * dxi)
    gamma = x_xi ** 2 + y_xi ** 2

    if j == 0:
        x_eta = (X[1, 1:-1] - X[0, 1:-1]) / deta
        y_eta = (Y[1, 1:-1] - Y[0, 1:-1]) / deta
    else:  # j == Nj-1
        x_eta = (X[-1, 1:-1] - X[-2, 1:-1]) / deta
        y_eta = (Y[-1, 1:-1] - Y[-2, 1:-1]) / deta

    alpha = x_eta ** 2 + y_eta ** 2
    beta  = x_xi * x_eta + y_xi * y_eta
    return alpha, gamma, beta


# ---------------------------------------------------------------------------
# OPT 1 — Pre-build ALL row tridiagonals in one vectorised pass
# ---------------------------------------------------------------------------

def precompute_left_sides(alpha: NDArray, gamma: NDArray, phi: NDArray
                          ) -> Tuple[NDArray, NDArray, NDArray]:
    a_full = alpha * (1.0 - phi / 2.0)
    c_full = alpha * (1.0 + phi / 2.0)
    b      = -2.0 * (alpha + gamma)
    return a_full, b, c_full


def precompute_left_sides_bnd(alpha_bnd: NDArray, gamma_bnd: NDArray,
                               phi_bnd: NDArray
                               ) -> Tuple[NDArray, NDArray, NDArray]:
    """Same formula as interior, but for a single boundary row (Ni-2,)."""
    a = alpha_bnd * (1.0 - phi_bnd / 2.0)
    c = alpha_bnd * (1.0 + phi_bnd / 2.0)
    b = -2.0 * (alpha_bnd + gamma_bnd)
    return a, b, c


# ---------------------------------------------------------------------------
# OPT 2 — Compiled TDMA via scipy.linalg.solve_banded
# ---------------------------------------------------------------------------

def TDMA_banded(a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> NDArray:
    n = len(d)
    ab = np.zeros((3, n), dtype=float)
    ab[0, 1:]  = c
    ab[1, :]   = b
    ab[2, :-1] = a
    return solve_banded((1, 1), ab, d)


def TDMA(a, b, c, d):
    return TDMA_banded(a, b, c, d)


# ---------------------------------------------------------------------------
# RHS — interior rows
# ---------------------------------------------------------------------------

def createRightSide(beta: NDArray, gamma: NDArray, P: NDArray,
                    psi: NDArray, j_global: int) -> NDArray:
    j_int = j_global - 1

    P_jp = P[j_global + 1, 1:-1]
    P_jm = P[j_global - 1, 1:-1]

    P_ip_jp = P[j_global + 1, 2:]
    P_im_jp = P[j_global + 1, :-2]
    P_ip_jm = P[j_global - 1, 2:]
    P_im_jm = P[j_global - 1, :-2]

    bet = beta [j_int, :]
    gam = gamma[j_int, :]
    ps  = psi  [j_int, :]

    cross = (bet / 2.0) * (P_ip_jp - P_ip_jm - P_im_jp + P_im_jm)
    ns    = -gam * (P_jp + P_jm)
    force = -(gam * ps / 2.0) * (P_jp - P_jm)

    return cross + ns + force


# ---------------------------------------------------------------------------
# RHS — Neumann boundary rows (ghost-cell)
#
# Ghost-cell rule:   P[ghost] = P[interior_neighbour]
#   j=0    (bot): ghost at j=-1  →  P[-1,:] := P[1,:]
#   j=Nj-1 (top): ghost at j=Nj →  P[Nj,:] := P[Nj-2,:]
#
# With ghost substitution:
#   P_jp + P_jm = 2 * P_nb    (P_nb = interior neighbour row)
#   P_jp - P_jm = 0            → forcing term vanishes
#   cross term  = 0            → antisymmetric, cancels
#
# Result: RHS = -gamma * 2 * P_nb
# ---------------------------------------------------------------------------

def createRightSide_neumann(gamma_bnd: NDArray, P: NDArray,
                             j_global: int) -> NDArray:
    """
    Build RHS for a Neumann boundary row using ghost-cell symmetry.

    Parameters
    ----------
    gamma_bnd : (Ni-2,) gamma at this boundary row
    P         : (Nj, Ni) full grid (X or Y)
    j_global  : 0 (bottom) or Nj-1 (top)
    """
    Nj = P.shape[0]
    j_nb = 1 if j_global == 0 else Nj - 2
    P_nb = P[j_nb, 1:-1]
    return -gamma_bnd * 2.0 * P_nb

def createLeftSide(alpha: NDArray, gamma: NDArray, phi: NDArray, j_int: int
                   ) -> Tuple[NDArray, NDArray, NDArray]:
    aj   = alpha[j_int, :]
    gj   = gamma[j_int, :]
    phij = phi  [j_int, :]
    b      = -2.0 * (aj + gj)
    a_full = aj * (1.0 - phij / 2.0)
    c_full = aj * (1.0 + phij / 2.0)
    return a_full[1:], b, c_full[:-1]

def buildTreeCache(curveBool, BoundaryArray, sides, helpers,):
    treeCache = {}
    from TTM.BoundaryProjection import build_polyline_tree
    if curveBool:
        for idx in range(4):
            if BoundaryArray[idx]:
                key = sides[idx]
                if key in helpers:
                    xc, yc = helpers[key]
                    treeCache[idx] = build_polyline_tree(
                        np.column_stack((xc, yc)))
    return treeCache


def SolveRow(Z, beta, gamma, psi, index, omega,
             aj, bj, cj, a_sub, c_sup, max_delta):

    dX = createRightSide(beta, gamma, Z, psi, index)
    dX[0] -= aj[0] * Z[index, 0]
    dX[-1] -= cj[-1] * Z[index, -1]

    xj = TDMA_banded(a_sub, bj, c_sup, dX)

    x_new = (1 - omega) * Z[index, 1:-1] + omega * xj

    dx = np.max(np.abs(x_new - Z[index, 1:-1]))
    if dx > max_delta: max_delta = dx

    Z[index, 1:-1] = x_new
    return max_delta
