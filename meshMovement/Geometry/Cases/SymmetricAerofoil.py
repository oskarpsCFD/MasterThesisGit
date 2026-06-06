from Geometry.ParametrisizedCurves import *

C_radius =  30
topLength = C_radius
COrigin = (3.0, 0.0)
frac_airfoil = 0.73 # 57
airfoilThickness = 0.12

airfoilOrigin = (-11  , 0.0)   # trailing edge location
offset = 0.1


sideLength = C_radius
backOrigin = (COrigin[0] + topLength / 2.0, COrigin[1])
WAKE_CLUSTER_BETA = 3.0  # tanh clustering of wake nodes toward the trailing edge; larger = tighter
wakeLength = backOrigin[0] - airfoilOrigin[0]

if wakeLength <= 0:
    raise ValueError("Airfoil trailing edge must lie left of the back boundary.")


# Generation of mesh bounadries and mesh boundary movement arrays
NSample = 20001
frac = np.array([frac_airfoil, 1, 1, 1])


def logicalSpace(NX: int, NY: int):
    xi = np.linspace(0, 1, NX)
    eta = np.linspace(0, 1, NY)
    dxi, deta = np.average(np.diff(xi)), np.average(np.diff(eta))
    X, Y = np.meshgrid(xi, eta)

    return xi, eta, X, Y, dxi, deta


def xb(nx: NDArray, NX: int):
    x, y = airfoil_with_wake_cut(
        t=airfoilThickness,
        N_total=NX,
        backOrigin=backOrigin,
        airfoilOrigin=airfoilOrigin,
        frac_airfoil=frac_airfoil
    )
    return x, y


def xt(nx: NDArray, NX: int):
    s = nx / (NX - 1)
    x, y = C_shape(C_radius, topLength, s, COrigin)

    return x, y


def xr(ny: NDArray, NY: int):
    s = ny / (NY - 1)
    x, y = vertical(-sideLength, s, backOrigin)
    return x, y


def xl(ny: NDArray, NY: int):
    s = ny / (NY - 1)
    x, y = vertical(sideLength, s, backOrigin)
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


    if plot == True:
        import matplotlib.pyplot as plt
        plt.plot(Xl, Yl)
        plt.plot(Xr, Yr)
        plt.plot(Xt, Yt)
        plt.plot(Xb, Yb)
        plt.show()
    return Cb,Cl,Cr,Ct


def consistency(Cb, Cl, Cr, Ct, tol=1e-12):
    """
    Curves are given as (N, 2) arrays:
    [ [x, y], ... ]

    Checks:
        Cb(0)  == Cl(0)
        Cb(1)  == Cr(0)
        Ct(0)  == Cl(1)
        Ct(1)  == Cr(1)
    """
    ok = True

    # bottom-left
    if not np.allclose(Cb[0],  Cl[0],  atol=tol):
        print("Consistency does not match Cb[0],  Cl[0]", Cb[0],  Cl[0])
        ok = False
    # bottom-right
    if not np.allclose(Cb[-1], Cr[0],  atol=tol):
        print("Consistency does not match Cb[-1], Cr[0]", Cb[-1], Cr[0])
        ok = False
    # top-left
    if not np.allclose(Ct[0],  Cl[-1], atol=tol):
        print("Consistency does not match Ct[0],  Cl[-1]", Ct[0],  Cl[-1])
        ok = False
    # top-right
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
    else:
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
