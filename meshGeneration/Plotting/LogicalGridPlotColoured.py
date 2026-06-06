import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ---------- color helpers ----------

def _normalize01(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A)
    amin = float(A.min())
    amax = float(A.max())
    if abs(amax - amin) < 1e-14:
        return np.zeros_like(A)
    return (A - amin) / (amax - amin)


def _colors_from_sides(
    XI: np.ndarray,
    ETA: np.ndarray,
    c_bottom,
    c_top,
    c_left,
    c_right,
) -> np.ndarray:
    """
    Smooth RGB interpolation from four side colors in logical space.

    Uses a symmetric blend:
      C = 0.5 * [ (1-eta)*Cb + eta*Ct + (1-xi)*Cl + xi*Cr ]
    """
    xi = _normalize01(XI)
    eta = _normalize01(ETA)

    cb = np.array(c_bottom, float)
    ct = np.array(c_top, float)
    cl = np.array(c_left, float)
    cr = np.array(c_right, float)

    C = 0.5 * (
        (1.0 - eta)[..., None] * cb
        + eta[..., None] * ct
        + (1.0 - xi)[..., None] * cl
        + xi[..., None] * cr
    )
    return np.clip(C, 0.0, 1.0)  # (Ny, Nx, 3)


# ---------- segment + color building ----------

def _row_segments(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Ny, Nx = X.shape
    segs = np.empty((Ny * (Nx - 1), 2, 2), float)
    k = 0
    for j in range(Ny):
        x0, y0 = X[j, :-1], Y[j, :-1]
        x1, y1 = X[j, 1:],  Y[j, 1:]
        n = Nx - 1
        segs[k:k+n, 0, 0] = x0
        segs[k:k+n, 0, 1] = y0
        segs[k:k+n, 1, 0] = x1
        segs[k:k+n, 1, 1] = y1
        k += n
    return segs


def _col_segments(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Ny, Nx = X.shape
    segs = np.empty((Nx * (Ny - 1), 2, 2), float)
    k = 0
    for i in range(Nx):
        x0, y0 = X[:-1, i], Y[:-1, i]
        x1, y1 = X[1:,  i], Y[1:,  i]
        n = Ny - 1
        segs[k:k+n, 0, 0] = x0
        segs[k:k+n, 0, 1] = y0
        segs[k:k+n, 1, 0] = x1
        segs[k:k+n, 1, 1] = y1
        k += n
    return segs


def _row_segment_colors(C: np.ndarray) -> np.ndarray:
    Ny, Nx, _ = C.shape
    cols = np.empty((Ny * (Nx - 1), 4), float)
    k = 0
    for j in range(Ny):
        cc = 0.5 * (C[j, :-1, :] + C[j, 1:, :])  # avg endpoints
        n = Nx - 1
        cols[k:k+n, :3] = cc
        cols[k:k+n, 3] = 1.0
        k += n
    return cols


def _col_segment_colors(C: np.ndarray) -> np.ndarray:
    Ny, Nx, _ = C.shape
    cols = np.empty((Nx * (Ny - 1), 4), float)
    k = 0
    for i in range(Nx):
        cc = 0.5 * (C[:-1, i, :] + C[1:, i, :])
        n = Ny - 1
        cols[k:k+n, :3] = cc
        cols[k:k+n, 3] = 1.0
        k += n
    return cols


# ---------- plotting primitives ----------

def _plot_grid_colored(ax, X, Y, Cnodes,
                       lw=0.8,
                       boundary_lw=2.0,
                       boundary_alpha=1.0,
                       interior_alpha=0.95,
                       show_points=False,
                       point_size=8):
    """
    Plot structured grid using LineCollections with per-segment RGB.
    Cnodes is (Ny,Nx,3) node colors. Segment colors are averages.
    """

    Ny, Nx = X.shape

    # Interior grid (all rows/cols)
    row_lc = LineCollection(_row_segments(X, Y), linewidths=lw, alpha=interior_alpha)
    col_lc = LineCollection(_col_segments(X, Y), linewidths=lw, alpha=interior_alpha)
    row_lc.set_colors(_row_segment_colors(Cnodes))
    col_lc.set_colors(_col_segment_colors(Cnodes))
    ax.add_collection(row_lc)
    ax.add_collection(col_lc)

    # Boundaries thicker (bottom, top, left, right)
    # bottom (j=0), top (j=Ny-1)
    ax.plot(X[0, :],     Y[0, :],     color=Cnodes[0, :, :].mean(axis=0), linewidth=boundary_lw, alpha=boundary_alpha)
    ax.plot(X[-1, :],    Y[-1, :],    color=Cnodes[-1, :, :].mean(axis=0), linewidth=boundary_lw, alpha=boundary_alpha)
    # left (i=0), right (i=Nx-1)
    ax.plot(X[:, 0],     Y[:, 0],     color=Cnodes[:, 0, :].mean(axis=0), linewidth=boundary_lw, alpha=boundary_alpha)
    ax.plot(X[:, -1],    Y[:, -1],    color=Cnodes[:, -1, :].mean(axis=0), linewidth=boundary_lw, alpha=boundary_alpha)

    if show_points:
        ax.scatter(X.ravel(), Y.ravel(), s=point_size, c=Cnodes.reshape(-1, 3), marker=".", linewidths=0)

    ax.set_aspect("equal")


# ---------- user-facing function ----------

def plot_logical_and_real_colored(
    XI, ETA, X, Y,
    show_points=False,
    lw=0.8,
    boundary_lw=2.2,
    dpi=300,
    figsize=(10, 4),
    side_colors=None,
):
    """
    Two-panel plot (logical + real), both color-coded.
    Colors are defined in logical space from four side colors and carried to real space.
    """

    if side_colors is None:
        # nicer, muted palette (works on white background)
        side_colors = {
            "bottom": (0.20, 0.55, 0.80),  # muted blue
            "top":    (0.90, 0.45, 0.15),  # muted orange
            "left":   (0.25, 0.70, 0.45),  # muted green
            "right":  (0.60, 0.45, 0.80),  # muted purple
        }

    Cnodes = _colors_from_sides(
        XI, ETA,
        c_bottom=side_colors["bottom"],
        c_top=side_colors["top"],
        c_left=side_colors["left"],
        c_right=side_colors["right"],
    )

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
    })

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Logical
    _plot_grid_colored(
        axs[0], XI, ETA, Cnodes,
        lw=lw, boundary_lw=boundary_lw,
        show_points=show_points
    )
    axs[0].set_xlabel(r'$\xi$')
    axs[0].set_ylabel(r'$\eta$')
    axs[0].set_title('Logical Space')

    # Real
    _plot_grid_colored(
        axs[1], X, Y, Cnodes,
        lw=lw, boundary_lw=boundary_lw,
        show_points=show_points
    )
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title('Real Space')

    # Tight and nice
    fig.tight_layout()
    plt.show()
    return fig, axs


# Example:
# plot_logical_and_real_colored(XI, ETA, X, Y, show_points=False)