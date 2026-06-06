from Geometry.ParametrisizedCurves import *

# Annular O-mesh between two concentric cylinders.
# xi  (NX direction) → angular, 0 to 2π, stitched at cut line θ=0
# eta (NY direction) → radial,  r_inner to r_outer
r_inner = 1.0
r_outer = 4.0
origin = (0.0, 0.0)

NSample = 10001
frac = np.array([1, 1, 1, 1])


def logicalSpace(NX: int, NY: int):
    xi = np.linspace(0, 1, NX)
    eta = np.linspace(0, 1, NY)
    dxi, deta = np.average(np.diff(xi)), np.average(np.diff(eta))
    X, Y = np.meshgrid(xi, eta)
    return xi, eta, X, Y, dxi, deta


def xb(nx: NDArray, NX: int):
    """Inner circle, θ: 0 → 2π.  xb[0] == xb[-1] in physical space (stitched)."""
    s = nx / (NX - 1)
    return fullCircle(r_inner, s, origin)


def xt(nx: NDArray, NX: int):
    """Outer circle, θ: 0 → 2π."""
    s = nx / (NX - 1)
    return fullCircle(r_outer, s, origin)


def xl(ny: NDArray, NY: int):
    """Radial cut line at θ=0, from inner to outer radius."""
    s = ny / (NY - 1)
    return radial(r_inner, r_outer, 0.0, s, origin)


def xr(ny: NDArray, NY: int):
    """Radial cut line at θ=2π (same physical location as xl — stitched boundary)."""
    s = ny / (NY - 1)
    return radial(r_inner, r_outer, 2 * np.pi, s, origin)


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
        theta = np.linspace(0, 2 * np.pi, 500)
        fig, ax = plt.subplots()
        ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta), 'k--', lw=0.8, label='inner')
        ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), 'k--', lw=0.8, label='outer')
        ax.plot(Xl, Yl, label='left (θ=0)')
        ax.plot(Xr, Yr, '--', label='right (θ=2π, stitched)')
        ax.set_aspect('equal')
        ax.legend()
        plt.show()

    return Cb, Cl, Cr, Ct


def consistency(Cb, Cl, Cr, Ct, tol=1e-10):
    """
    Corner checks (table 1.3):
        Cb[0]  == Cl[0]    inner circle start  == cut line inner end
        Cb[-1] == Cr[0]    inner circle end    == cut line inner end  (stitched → same point)
        Ct[0]  == Cl[-1]   outer circle start  == cut line outer end
        Ct[-1] == Cr[-1]   outer circle end    == cut line outer end  (stitched → same point)
    """
    ok = True

    if not np.allclose(Cb[0], Cl[0], atol=tol):
        print("Consistency fail: Cb[0] != Cl[0]", Cb[0], Cl[0])
        ok = False
    if not np.allclose(Cb[-1], Cr[0], atol=tol):
        print("Consistency fail: Cb[-1] != Cr[0]", Cb[-1], Cr[0])
        ok = False
    if not np.allclose(Ct[0], Cl[-1], atol=tol):
        print("Consistency fail: Ct[0] != Cl[-1]", Ct[0], Cl[-1])
        ok = False
    if not np.allclose(Ct[-1], Cr[-1], atol=tol):
        print("Consistency fail: Ct[-1] != Cr[-1]", Ct[-1], Cr[-1])
        ok = False

    if ok:
        print("Consistency matches table 1.3")
    else:
        print("Consistency does not match table 1.3")

    return ok


def generateHelperCurves(Sample: int, genCurves: list[str] = ["bot", "top", "left", "right"]):
    if not genCurves:
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
            raise ValueError(f"Unknown key: {key}")
    return curveDict
