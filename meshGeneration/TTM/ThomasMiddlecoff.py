import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter
from typing import Callable, Tuple


def testField(X, Y, plot=False):
    U = np.zeros_like(X)
    ramp_width = 2.0

    X_phys = X * 24
    Y_phys = Y * 24

    x_left = 0.5 * (Y_phys + 10.0)
    x_right = x_left + ramp_width

    mask1 = X_phys <= x_left
    mask2 = (X_phys > x_left) & (X_phys <= x_right)

    U[mask1] = 1.0
    U[mask2] = 1.0 - (X_phys[mask2] - x_left[mask2]) / ramp_width

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.pcolormesh(X, Y, U, cmap='RdBu_r', shading='auto')
        plt.colorbar(label='U')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.ylim(-2, 2)
        plt.title('Oblique shock field')
        plt.tight_layout()
        plt.show()

    return U

def clusterLeft(x: NDArray, A: float, B: float):
    return 1 + A * np.exp(-B * x)


def clusterRight(x: NDArray, A: float, B: float):
    return 1 + A * np.exp(-B * (1 - x))


def clusterBottom(y: NDArray, A: float, B: float):
    return 1 + A * np.exp(-B * y)


def clusterTop(y: NDArray, A: float, B: float):
    return 1 + A * np.exp(-B * (1 - y))


def clusterXiPoints(Xi: NDArray, xi_vals: list, A: float, B: float):
    """
    Gaussian peaks at each xi value in xi_vals.
    Drives w1 to cluster grid lines toward those ξ locations.
    """
    w = np.ones_like(Xi)
    for xi0 in xi_vals:
        w += A * np.exp(-B * (Xi - xi0) ** 2)
    return w



def smoothField(w: NDArray, n_passes: int = 5) -> NDArray:
    """
    Apply n_passes of a 3x3 box filter using scipy.ndimage.uniform_filter.

    Boundary rows/columns are preserved (matching the original smoother
    semantics where only interior points are updated).
    """
    boundary = (
        w[0, :].copy(),
        w[-1, :].copy(),
        w[:, 0].copy(),
        w[:, -1].copy(),
    )

    for _ in range(n_passes):
        w = uniform_filter(w, size=3, mode='nearest')

    # Restore boundary values
    w[0, :]  = boundary[0]
    w[-1, :] = boundary[1]
    w[:, 0]  = boundary[2]
    w[:, -1] = boundary[3]

    return w


def generateWeightFunctions(
    Xi: NDArray,
    Eta: NDArray,
    clusters: list[dict] = None,
    smoothIt: int = 5,
) -> tuple[NDArray, NDArray]:
    """
    Generate weight functions w1 (Xi-axis) and w2 (Eta-axis).

    Parameters
    ----------
    clusters : list of dicts, each with keys:
        'side' : 'left' | 'right' | 'bottom' | 'top'
        'A'    : float  (amplitude)
        'B'    : float  (decay rate)

    Example
    -------
    clusters = [
        {'side': 'bottom', 'A': 25, 'B': 5},
    ]
    """
    w1 = np.ones_like(Xi)
    w2 = np.ones_like(Eta)

    for c in (clusters or []):
        side = c['side']
        A    = c['A']
        B    = c['B']

        if side == 'left':
            w1 *= clusterLeft(Xi, A, B)
        elif side == 'right':
            w1 *= clusterRight(Xi, A, B)
        elif side == 'bottom':
            w2 *= clusterBottom(Eta, A, B)
        elif side == 'top':
            w2 *= clusterTop(Eta, A, B)
        elif side == 'xi_points':
            xi_vals = c['xi_vals']
            w1 *= clusterXiPoints(Xi, xi_vals, A, B)
        else:
            raise ValueError(f"Unknown side '{side}'. Use: left, right, bottom, top, xi_points")

    w1 = smoothField(w1, n_passes=smoothIt)
    w2 = smoothField(w2, n_passes=smoothIt)

    return w1, w2


def computeGridControlFunctions(
        X: NDArray, Y: NDArray,
        field_func: Callable,
        w1Base:NDArray, w2Base:NDArray,
        A_xi: float, A_eta: float,
        dxi: float, deta: float,
        smooth_passes: int = 1,isotropic: bool = False,
        testMode=True
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    U = field_func(X, Y)
    if isotropic:
        dU_xi  = np.gradient(U, dxi,  axis=1)
        dU_eta = np.gradient(U, deta, axis=0)

        # Normalise by peak gradient so phi stays bounded
        hx = max(float(np.max(np.abs(dU_xi))),  1e-12)
        he = max(float(np.max(np.abs(dU_eta))), 1e-12)
        if testMode:
            w1 = smoothField(1.0 + A_xi  * (dU_xi  / hx) ** 2, n_passes=smooth_passes)
            w2 = smoothField(1.0 + A_eta * (dU_eta / he) ** 2, n_passes=smooth_passes)


        else:
            w1, w2 = np.ones_like(w1Base), np.ones_like(w2Base)


    else:


        U_xi = np.gradient(U, dxi, axis=1)
        U_eta = np.gradient(U, deta, axis=0)

        X_xi = np.gradient(X, dxi, axis=1)
        X_eta = np.gradient(X, deta, axis=0)
        Y_xi = np.gradient(Y, dxi, axis=1)
        Y_eta = np.gradient(Y, deta, axis=0)

        J = X_xi * Y_eta - X_eta * Y_xi
        J = np.where(np.abs(J) < 1e-12, 1e-12, J)

        U_x = (U_xi * Y_eta - U_eta * Y_xi) / J
        U_y = (-U_xi * X_eta + U_eta * X_xi) / J

        G2 = U_x ** 2 + U_y ** 2
        G2max = max(float(np.max(G2)), 1e-12)

        if testMode:
            w = 1.0 + A_xi * G2 / G2max
            w = smoothField(w, n_passes=smooth_passes)

        else:
            w = np.ones_like(w1Base)

        w1 = w.copy()
        w2 = w.copy()

    w1 *= w1Base
    w2 *= w2Base

    phi = (w1[1:-1, 2:] - w1[1:-1, :-2]) / 2.0 / w1[1:-1, 1:-1]
    psi = (w2[2:,  1:-1] - w2[:-2, 1:-1]) / 2.0 / w2[1:-1, 1:-1]

    return phi, psi, w1, w2


