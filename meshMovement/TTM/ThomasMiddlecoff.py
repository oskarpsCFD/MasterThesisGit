import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from typing import Tuple


def cluster_left(x: NDArray, A: float, B: float):
    return 1 + A * np.exp(-B * x)


def cluster_right(x: NDArray, A: float, B: float):
    return 1 + A * np.exp(-B * (1 - x))


def cluster_bottom(y: NDArray, A: float, B: float):
    return 1 + A * np.exp(-B * y)


def cluster_top(y: NDArray, A: float, B: float):
    return 1 + A * np.exp(-B * (1 - y))


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
            w1 *= cluster_left(Xi, A, B)
        elif side == 'right':
            w1 *= cluster_right(Xi, A, B)
        elif side == 'bottom':
            w2 *= cluster_bottom(Eta, A, B)
        elif side == 'top':
            w2 *= cluster_top(Eta, A, B)
        else:
            raise ValueError(f"Unknown side '{side}'. Use: left, right, bottom, top")

    w1 = smoothField(w1, n_passes=smoothIt)
    w2 = smoothField(w2, n_passes=smoothIt)

    return w1, w2


def physicalGradToLogical(
        gradFieldX: NDArray, gradFieldY: NDArray,
        X: NDArray, Y: NDArray,
        dxi: float, deta: float,
        eps: float = 1e-12,
) -> Tuple[NDArray, NDArray]:
    """
    Project a physical gradient field (gx, gy) onto logical xi and eta
    directions at every grid point, returning full-grid arrays of the
    same shape as X.
    """
    # --- metric tensors on the full grid with appropriate differencing ---
    x_xi  = np.empty_like(X)
    y_xi  = np.empty_like(Y)
    x_eta = np.empty_like(X)
    y_eta = np.empty_like(Y)

    # Interior: central differences
    x_xi [:, 1:-1] = (X[:, 2:] - X[:, :-2]) / (2 * dxi)
    y_xi [:, 1:-1] = (Y[:, 2:] - Y[:, :-2]) / (2 * dxi)
    x_eta[1:-1, :] = (X[2:, :] - X[:-2, :]) / (2 * deta)
    y_eta[1:-1, :] = (Y[2:, :] - Y[:-2, :]) / (2 * deta)

    # Boundaries: one-sided differences
    x_xi [:, 0]  = (X[:, 1]  - X[:, 0])  / dxi
    x_xi [:, -1] = (X[:, -1] - X[:, -2]) / dxi
    y_xi [:, 0]  = (Y[:, 1]  - Y[:, 0])  / dxi
    y_xi [:, -1] = (Y[:, -1] - Y[:, -2]) / dxi

    x_eta[0,  :] = (X[1,  :] - X[0,  :]) / deta
    x_eta[-1, :] = (X[-1, :] - X[-2, :]) / deta
    y_eta[0,  :] = (Y[1,  :] - Y[0,  :]) / deta
    y_eta[-1, :] = (Y[-1, :] - Y[-2, :]) / deta

    # --- arc-length normalisation ---
    len_xi  = np.sqrt(x_xi  ** 2 + y_xi  ** 2) + eps
    len_eta = np.sqrt(x_eta ** 2 + y_eta ** 2) + eps

    # --- project ---
    grad_xi  = (gradFieldX * x_xi  + gradFieldY * y_xi)  / len_xi
    grad_eta = (gradFieldX * x_eta + gradFieldY * y_eta) / len_eta

    return grad_xi, grad_eta


def computeThomasMiddlecoff(
        X: NDArray, Y: NDArray,
        gradFieldX: NDArray, gradFieldY: NDArray,
        w1Base: NDArray, w2Base: NDArray,
        C2_xi: float,
        dxi: float, deta: float,
        smooth_passes: int = 3,
        isotropic: bool = False,
        percentile_clip: float = 99.0,
        xi_filter: tuple | None = None,
        eta_filter: tuple | None = None,
        C2_eta: float | None = None,   # If None, falls back to C2. Set 0.0 to disable eta grading.
        xi_cutoff: float = 1.0,   # Logical xi beyond which field has no effect (1.0 = disabled)
        xi_ramp_width: float = 0.1,
        eta_cutoff: float = 1.0,  # Logical eta beyond which field has no effect (1.0 = disabled)
        eta_ramp_width: float = 0.1,
        ducros_sensor: NDArray | None = None,  # Ducros shock sensor field, shape (Nj, Ni), values in [0, 1]
        C_ducros: float = 1.0,                 # Amplification factor for the Ducros boost
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:

    Nj, Ni = X.shape

    C2_eta_val = C2_xi if C2_eta is None else C2_eta

    # --- build the spatial mask (logical space) ---
    xi_coords  = np.linspace(0.0, 1.0, Ni)
    eta_coords = np.linspace(0.0, 1.0, Nj)

    mask = np.ones((Nj, Ni), dtype=float)

    if xi_filter is not None:
        xi_lo, xi_hi = xi_filter
        col_mask = (xi_coords >= xi_lo) & (xi_coords <= xi_hi)
        mask[:, col_mask] = 0.0

    if eta_filter is not None:
        eta_lo, eta_hi = eta_filter
        row_mask = (eta_coords >= eta_lo) & (eta_coords <= eta_hi)
        mask[row_mask, :] = 0.0

    # --- logical-space xi cutoff with smooth cosine ramp (applied column-wise) ---
    if xi_cutoff < 1.0:
        xi_ramp_start = xi_cutoff - xi_ramp_width
        xi_col_mask = np.where(
            xi_coords < xi_ramp_start, 1.0,
            np.where(
                xi_coords > xi_cutoff, 0.0,
                0.5 * (1.0 + np.cos(np.pi * (xi_coords - xi_ramp_start) / xi_ramp_width))
            )
        )
        mask *= xi_col_mask[np.newaxis, :]  # broadcast over rows

    # --- logical-space eta cutoff with smooth cosine ramp (applied row-wise) ---
    if eta_cutoff < 1.0:
        eta_ramp_start = eta_cutoff - eta_ramp_width
        eta_row_mask = np.where(
            eta_coords < eta_ramp_start, 1.0,
            np.where(
                eta_coords > eta_cutoff, 0.0,
                0.5 * (1.0 + np.cos(np.pi * (eta_coords - eta_ramp_start) / eta_ramp_width))
            )
        )
        mask *= eta_row_mask[:, np.newaxis]  # broadcast over columns

    # Do NOT apply the spatial mask to the raw gradient field.
    # Masking gx/gy before computing w creates a large d(ln w)/dxi spike at
    # the ramp edge because phi = (w[k+1]-w[k-1])/(2*w[k]) amplifies the
    # feature in w caused by the mask transition.  Instead, compute phi/psi
    # from the full unmasked field and multiply them by the mask afterward:
    # scaling phi directly is smooth and causes no secondary spike.
    def MonitorNormalisation(raw: NDArray, c2: float) -> NDArray:
        if c2 == 0.0:
            # Short-circuit: no field-driven grading, return uniform 1
            return np.ones_like(raw)
        g      = np.abs(raw)
        g_max  = np.percentile(g, percentile_clip)
        g      = np.clip(g, 0.0, g_max)
        g_norm = g / (g_max + 1e-12)
        return smoothField(1.0 + c2 * g_norm, n_passes=smooth_passes)

    ducros_boost = (
        smoothField(1.0 + C_ducros * ducros_sensor, n_passes=smooth_passes)
        if ducros_sensor is not None else 1.0
    )

    if isotropic:
        mag     = np.sqrt(gradFieldX ** 2 + gradFieldY ** 2)
        monitor = MonitorNormalisation(mag * ducros_boost, C2_xi)
        w1 = w1Base * monitor
        w2 = w2Base * monitor
    else:
        grad_xi, grad_eta = physicalGradToLogical(
            gradFieldX, gradFieldY, X, Y, dxi, deta
        )
        w1 = w1Base * MonitorNormalisation(grad_xi * ducros_boost, C2_xi)
        w2 = w2Base * MonitorNormalisation(grad_eta * ducros_boost, C2_eta_val)

    phi = (w1[1:-1, 2:] - w1[1:-1, :-2]) / (2.0 * w1[1:-1, 1:-1])
    psi = (w2[2:,  1:-1] - w2[:-2, 1:-1]) / (2.0 * w2[1:-1, 1:-1])

    # Apply the spatial mask to phi/psi directly.  phi appears as a
    # coefficient in the Winslow tridiagonal (not through its derivative),
    # so zeroing it in the dead zone simply turns off node attraction there
    # without creating any spike.
    mask_int = mask[1:-1, 1:-1]
    phi *= mask_int
    psi *= mask_int

    return phi, psi, w1, w2


def build_field_interpolator(
        P_OF: NDArray,
        field_OF: NDArray,
) -> tuple:
    """
    Build callable interpolators for gradFieldX and gradFieldY from the
    original OpenFOAM point cloud, so they can be re-evaluated at updated
    mesh positions during Winslow iterations.

    Parameters
    ----------
    P_OF      : (N, 2 or 3)  OpenFOAM point coordinates (physical space)
    field_OF  : (N, 2 or 3)  gradient field at those points

    Returns
    -------
    interp_gx, interp_gy : callables  (M, 2) query_pts -> (M,) values
        LinearNDInterpolator with nearest-neighbour fallback for any
        query points that fall outside the convex hull.
    """
    pts = np.asarray(P_OF)[:, :2]
    fld = np.asarray(field_OF)[:, :2]

    lin_gx = LinearNDInterpolator(pts, fld[:, 0])
    lin_gy = LinearNDInterpolator(pts, fld[:, 1])
    nn_gx  = NearestNDInterpolator(pts, fld[:, 0])
    nn_gy  = NearestNDInterpolator(pts, fld[:, 1])

    def _make(lin, nn):
        def _call(query_pts):
            vals = lin(query_pts)
            nans = np.isnan(vals)
            if nans.any():
                vals[nans] = nn(query_pts[nans])
            return vals
        return _call

    return _make(lin_gx, nn_gx), _make(lin_gy, nn_gy)


def reinterpolate_phi_psi(
        X: NDArray, Y: NDArray,
        interp_gx,
        interp_gy,
        w1Base: NDArray,
        w2Base: NDArray,
        dxi: float, deta: float,
        **compute_kwargs,
) -> Tuple[NDArray, NDArray]:
    """
    Re-interpolate gradFieldX/gradFieldY onto the current (X, Y) mesh
    positions and recompute phi and psi via compute_phi_psi_liu_monitor.

    Parameters
    ----------
    X, Y        : current structured grid  (Nj, Ni)
    interp_gx,
    interp_gy   : callables from build_field_interpolator
    w1Base,
    w2Base      : base weight arrays (Nj, Ni)
    dxi, deta   : logical-space step sizes
    **compute_kwargs : forwarded to computeThomasMiddlecoff
                       (C2_xi, C2_eta, smooth_passes, isotropic, …)

    Returns
    -------
    phi, psi : (Nj-2, Ni-2) updated attraction weights
    """
    query_pts = np.column_stack((X.ravel(), Y.ravel()))
    gx = interp_gx(query_pts).reshape(X.shape)
    gy = interp_gy(query_pts).reshape(Y.shape)
    phi, psi, _, _ = computeThomasMiddlecoff(
        X, Y, gx, gy, w1Base, w2Base,
        dxi=dxi, deta=deta,
        **compute_kwargs,
    )
    return phi, psi
