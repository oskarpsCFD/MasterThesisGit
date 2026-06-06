import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from Geometry.clusterFunctins import f, fAirfoilCluster, fRadian


"""
All parametric curves to be used in a design of a mesh.
"""


def horisontal(length: float, s: NDArray, origin: Tuple[float, float] = (0, 0)) -> Tuple[NDArray, NDArray, str]:
    """
    length: Length of the line
    s: Parametric variable
    """
    s = f(s) * length
    x = s + origin[0]
    y = 0 * s + origin[1]
    return x, y

def vertical(length: float, s: NDArray, origin: Tuple[float, float] = (0, 0)) -> Tuple[NDArray, NDArray]:
    """
    length: Length of the line
    s: Parametric variable
    """
    s = f(s) * length
    x = 0 * s + origin[0]
    y = s + origin[1]
    return x, y


def NACA00XX(t: float, s: NDArray) -> Tuple[NDArray, NDArray]:
    """
    t: Thickness of the airfoil
    s: Parametric variable
    """

    # ---- coefficients
    a0 = 0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = 0.2843
    p = 0.5
    # enforce closed trailing edge:
    a4 = -(a0 + a1 + a2 + a3)

    x = fAirfoilCluster(s, p=p)
    y = np.zeros_like(x)
    leadingEdge = np.argmin(np.abs(x))
    xL = x[:leadingEdge]
    xT = x[leadingEdge:]

    y[:leadingEdge] = 5 * t * (a0 * np.sqrt(xL) + a1 * xL + a2 * xL ** 2 + a3 * xL ** 3 + a4 * xL ** 4)
    y[leadingEdge:] = -5 * t * (a0 * np.sqrt(xT) + a1 * xT + a2 * xT ** 2 + a3 * xT ** 3 + a4 * xT ** 4)

    return x, y

def wake_line(x_start: float, x_end: float, y: float, n: int, direction: str):
    if n < 2:
        raise ValueError("wake line needs at least 2 points")

    if direction == "in":
        x = np.linspace(x_start, x_end, n)
    elif direction == "out":
        x = np.linspace(x_end, x_start, n)
    else:
        raise ValueError("direction must be 'in' or 'out'")

    y_arr = np.full_like(x, y, dtype=float)
    return x, y_arr


def airfoil_with_wake_cut(
    t: float,
    N_total: int,
    backOrigin: Tuple[float, float],
    airfoilOrigin: Tuple[float, float],
    frac_airfoil: float = 0.7,
) -> Tuple[NDArray, NDArray]:
    """
    Composite curve:
      back boundary -> trailing edge -> around airfoil -> trailing edge -> back boundary
    """

    if N_total < 9:
        raise ValueError("Need N_total >= 9")
    if not (0.1 < frac_airfoil < 0.95):
        raise ValueError("Choose frac_airfoil sensibly, e.g. 0.6-0.85")

    x_back, _ = backOrigin
    x_te, y_te = airfoilOrigin

    Lwake = x_back - x_te
    if Lwake <= 0:
        raise ValueError("Airfoil trailing edge must lie left of the back boundary.")

    # --- choose N_air (odd)
    N_air = int(round(frac_airfoil * N_total))
    if N_air % 2 == 0:
        N_air += 1

    N_wake_total = (N_total + 2) - N_air
    if N_wake_total < 6:
        N_air -= 2
        N_wake_total = (N_total + 2) - N_air
        if N_wake_total < 6:
            raise ValueError("N_total too small for chosen frac_airfoil")

    N_wL = N_wake_total // 2
    N_wU = N_wake_total - N_wL
    if N_wL < 3:
        N_wL = 3
        N_wU = N_wake_total - N_wL
    if N_wU < 3:
        N_wU = 3
        N_wL = N_wake_total - N_wU

    if N_wL < 3 or N_wU < 3:
        raise ValueError("Could not allocate wake points robustly.")

    # --- wake pieces
    x_wL, y_wL = wake_line(x_back, x_te, y_te, N_wL, "in")
    x_wU, y_wU = wake_line(x_back, x_te, y_te, N_wU, "out")

    def cosspace(a: float, b: float, n: int) -> NDArray:
        theta = np.linspace(0.0, np.pi, n)
        return a + 0.5 * (b - a) * (1.0 - np.cos(theta))

    # --- airfoil
    s_air = cosspace(0.0, 1.0, N_air)
    x_a, y_a = NACA00XX(t=t, s=s_air)

    # shift from old TE=(1,0) to new TE=(x_te,y_te)
    x_a = x_a + (x_te - 1.0)
    y_a = y_a + y_te

    # concatenate, dropping duplicated TE points
    x = np.concatenate([x_wL, x_a[1:], x_wU[1:]])
    y = np.concatenate([y_wL, y_a[1:], y_wU[1:]])


    if len(x) != N_total:
        raise RuntimeError(f"airfoil_with_wake_cut produced {len(x)} points, expected {N_total}")

    return x, y


def C_shape(r: float, L: float, s: NDArray, origin: Tuple[float, float] = (0, 0)) -> Tuple[NDArray, NDArray]:
    """
    t: Thickness of the airfoil
    s: Parametric variable
    """
    s = np.sort(s)

    x0, y0 = origin
    xBack = x0 + L / 2.0
    xFront = x0 - L / 2.0

    # total length of the OPEN path (no back vertical)
    L_tot = L + np.pi * r + L

    frac_top = L / L_tot
    frac_circle = (np.pi * r) / L_tot
    frac_bot = L / L_tot  # same as top

    b0 = 0.0
    b1 = frac_top
    b2 = b1 + frac_circle
    b3 = 1.0  # end

    l_top = (s >= b0) & (s < b1)
    l_circle = (s >= b1) & (s < b2)
    l_bot = (s >= b2) & (s <= b3)

    x = np.zeros_like(s)
    y = np.zeros_like(s)

    # 1) Top horizontal: back -> front
    t1 = (s[l_top] - b0) / (b1 - b0 + 1e-15)
    x[l_top] = xBack - L * t1
    y[l_top] = y0 + r

    # 2) Front semicircle: top -> bottom bulging left
    t2 = (s[l_circle] - b1) / (b2 - b1 + 1e-15)
    theta = 0.5 * np.pi - np.pi * t2  # +pi/2 -> -pi/2
    x[l_circle] = xFront - r * np.cos(theta)
    y[l_circle] = y0 + r * np.sin(theta)

    # 3) Bottom horizontal: front -> back
    t3 = (s[l_bot] - b2) / (b3 - b2 + 1e-15)
    x[l_bot] = xFront + L * t3
    y[l_bot] = y0 - r

    return x, y


def circle(r: float, s: NDArray, origin: Tuple[float, float] = (0,0)) -> Tuple[NDArray, NDArray]:
    """
    t: Thickness of the airfoil
    s: Parametric variable
    """
    s = fRadian(s)

    x = r * np.cos(s) + origin[0]
    y = r * np.sin(s) + origin[1]

    return x, y