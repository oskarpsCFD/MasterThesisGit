import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from Geometry.clusterFunctins import f, fAirfoilCluster, fRadian, clusterTowardStart, clusterTowardEnd


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
    
    
def wakeLine(x_start: float, x_end: float, y: float, n: int, direction: str, beta: float = 0.0):
    if n < 2:
        raise ValueError("wake line needs at least 2 points")

    t = np.linspace(0.0, 1.0, n)
    if beta > 0.0:
        if direction == "in":
            # "in" runs x_start→x_end (back→trailing edge); cluster toward end (t=1)
            t = clusterTowardEnd(t, beta)
        elif direction == "out":
            # "out" runs x_end→x_start (trailing edge→back); cluster toward start (t=0)
            t = clusterTowardStart(t, beta)

    if direction == "in":
        x = x_start + t * (x_end - x_start)
    elif direction == "out":
        x = x_end + t * (x_start - x_end)
    else:
        raise ValueError("direction must be 'in' or 'out'")

    y_arr = np.full_like(x, y, dtype=float)
    return x, y_arr



def airfoilWithWakeCut(
    t: float,
    N_total: int,
    backOrigin: Tuple[float, float],
    airfoilOrigin: Tuple[float, float],
    frac_airfoil: float = 0.7,
    wake_cluster_beta: float = 0.0,
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
    x_wL, y_wL = wakeLine(x_back, x_te, y_te, N_wL, "in",  beta=wake_cluster_beta)
    x_wU, y_wU = wakeLine(x_back, x_te, y_te, N_wU, "out", beta=wake_cluster_beta)

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
        raise RuntimeError(f"airfoilWithWakeCut produced {len(x)} points, expected {N_total}")

    return x, y


def cShape(r: float, L: float, s: NDArray, origin: Tuple[float, float] = (0, 0), beta: float = 0.0,
            b1: float = None, b2: float = None) -> Tuple[NDArray, NDArray]:

    #t: Thickness of the airfoil
    #s: Parametric variable

    s = np.sort(s)

    x0, y0 = origin
    xBack = x0 + L / 2.0
    xFront = x0 - L / 2.0

    # total length of the OPEN path (no back vertical)
    L_tot = L + np.pi * r + L

    frac_top = L / L_tot
    frac_circle = (np.pi * r) / L_tot

    b0 = 0.0
    if b1 is None:
        b1 = frac_top
    if b2 is None:
        b2 = frac_top + frac_circle
    b3 = 1.0  # end

    l_top = (s >= b0) & (s < b1)
    l_circle = (s >= b1) & (s < b2)
    l_bot = (s >= b2) & (s <= b3)

    x = np.zeros_like(s)
    y = np.zeros_like(s)

    # 1) Top horizontal: back -> front; cluster toward inner (semicircle) end
    t1 = (s[l_top] - b0) / (b1 - b0 + 1e-15)
    if beta > 0.0:
        t1 = clusterTowardEnd(t1, beta)
    x[l_top] = xBack - L * t1
    y[l_top] = y0 + r

    # 2) Front semicircle: top -> bottom bulging left
    t2 = (s[l_circle] - b1) / (b2 - b1 + 1e-15)
    theta = 0.5 * np.pi - np.pi * t2  # +pi/2 -> -pi/2
    x[l_circle] = xFront - r * np.cos(theta)
    y[l_circle] = y0 + r * np.sin(theta)

    # 3) Bottom horizontal: front -> back; cluster toward inner (semicircle) end
    t3 = (s[l_bot] - b2) / (b3 - b2 + 1e-15)
    if beta > 0.0:
        t3 = clusterTowardStart(t3, beta)
    x[l_bot] = xFront + L * t3
    y[l_bot] = y0 - r
    print(x.min())
    print(x.max())
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


def fullCircle(r: float, s: NDArray, origin: Tuple[float, float] = (0, 0)) -> Tuple[NDArray, NDArray]:
    """
    Full circle from θ=0 to θ=2π (s=0 → s=1). First and last point
    coincide in physical space, enabling stitched/periodic meshes.
    No odd-length restriction.
    """
    s = np.asarray(s, dtype=float)
    theta = s * 2 * np.pi
    x = r * np.cos(theta) + origin[0]
    y = r * np.sin(theta) + origin[1]
    return x, y


def radial(r_inner: float, r_outer: float, angle: float, s: NDArray,
           origin: Tuple[float, float] = (0, 0)) -> Tuple[NDArray, NDArray]:
    """
    Radial line at fixed angle from r_inner (s=0) to r_outer (s=1).
    """
    s = np.asarray(s, dtype=float)
    r = r_inner + s * (r_outer - r_inner)
    x = r * np.cos(angle) + origin[0]
    y = r * np.sin(angle) + origin[1]
    return x, y
