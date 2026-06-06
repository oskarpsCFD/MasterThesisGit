from Geometry.ParametrisizedCurves import *

# Domain between two concentric C-shaped (horseshoe) curves.
# xi  (NX direction) → along the horseshoe curves, from open end → around → back to open end
# eta (NY direction) → radial, from inner horseshoe to outer horseshoe
r_inner = 0.5
r_outer = 1.5
L = 1.0          # length of the straight arms (same for both curves so backOrigin is shared)
origin = (0.0, 0.0)
backOrigin = (origin[0] + L / 2.0, origin[1])   # (2.0, 0.0)

NSample = 10001
frac = np.array([1, 1, 1, 1])


def logicalSpace(NX: int, NY: int):
    xi = np.linspace(0, 1, NX)
    eta = np.linspace(0, 1, NY)
    dxi, deta = np.average(np.diff(xi)), np.average(np.diff(eta))
    X, Y = np.meshgrid(xi, eta)
    return xi, eta, X, Y, dxi, deta


def xb(nx: NDArray, NX: int):
    """Inner horseshoe curve, from (backOrigin, +r_inner) around to (backOrigin, -r_inner)."""
    s = nx / (NX - 1)
    x, y = cShape(r_inner, L, s, origin)
    return x, y


def xt(nx: NDArray, NX: int):
    """Outer horseshoe curve, from (backOrigin, +r_outer) around to (backOrigin, -r_outer)."""
    s = nx / (NX - 1)
    x, y = cShape(r_outer, L, s, origin)
    return x, y


def xl(ny: NDArray, NY: int):
    """Upper connection at the open end: (backOrigin, r_inner) → (backOrigin, r_outer)."""
    s = ny / (NY - 1)
    x, y = vertical(r_outer - r_inner, s, (backOrigin[0], r_inner))
    return x, y


def xr(ny: NDArray, NY: int):
    """Lower connection at the open end: (backOrigin, -r_inner) → (backOrigin, -r_outer)."""
    s = ny / (NY - 1)
    x, y = vertical(-(r_outer - r_inner), s, (backOrigin[0], -r_inner))
    return x, y


def generateGeometry(NX, NY, plot=False):
    nx = np.arange(NX)
    ny = np.arange(NY)

    Xb, Yb = xb(nx, NX)
    Xt, Yt = xt(nx, NX)
    Xl, Yl = xl(ny, NY)
    Xr, Yr = xr(ny, NY)

    Cb = np.column_stack([Xb, Yb])
    Ct = np.column_stack([Xt, Yt])
    Cl = np.column_stack([Xl, Yl])
    Cr = np.column_stack([Xr, Yr])

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(Xb, Yb, label='inner (Cb)')
        ax.plot(Xt, Yt, label='outer (Ct)')
        ax.plot(Xl, Yl, label='upper side (Cl)')
        ax.plot(Xr, Yr, label='lower side (Cr)')
        ax.set_aspect('equal')
        ax.legend()
        plt.title('Horseshoe domain boundary')
        plt.show()

    return Cb, Cl, Cr, Ct


def consistency(Cb, Cl, Cr, Ct, tol=1e-12):
    ok = True

    if not np.allclose(Cb[0], Cl[0], atol=tol):
        print("Consistency does not match Cb[0], Cl[0]", Cb[0], Cl[0])
        ok = False
    if not np.allclose(Cb[-1], Cr[0], atol=tol):
        print("Consistency does not match Cb[-1], Cr[0]", Cb[-1], Cr[0])
        ok = False
    if not np.allclose(Ct[0], Cl[-1], atol=tol):
        print("Consistency does not match Ct[0], Cl[-1]", Ct[0], Cl[-1])
        ok = False
    if not np.allclose(Ct[-1], Cr[-1], atol=tol):
        print("Consistency does not match Ct[-1], Cr[-1]", Ct[-1], Cr[-1])
        ok = False

    if ok:
        print("Consistency matches table 1.3")
    else:
        print("Consistency does not match table 1.3")

    return ok


def generateHelperCurves(Sample: int, genCurves: list[str] = ["bot", "top", "left", "right"]):
    if len(genCurves) == 0:
        return None
    curveDict = {}
    n = np.arange(Sample)
    for key in genCurves:
        if key == "right":
            curveDict[key] = xr(n, Sample)
        elif key == "left":
            curveDict[key] = xl(n, Sample)
        elif key == "top":
            curveDict[key] = xt(n, Sample)
        elif key == "bot":
            curveDict[key] = xb(n, Sample)
        else:
            raise ValueError(f"Unknown key value of {key} was given")
    return curveDict
