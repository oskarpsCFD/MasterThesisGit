import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection


def ease_in_out(t: float) -> float:
    # smoothstep
    return t * t * (3.0 - 2.0 * t)


def _normalize01(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A)
    amin = A.min()
    amax = A.max()
    if abs(amax - amin) < 1e-14:
        return np.zeros_like(A)
    return (A - amin) / (amax - amin)


def _colors_from_sides(
    XI: np.ndarray,
    ETA: np.ndarray,
    c_bottom=(0.90, 0.20, 0.20),
    c_top=(0.20, 0.40, 0.95),
    c_left=(0.20, 0.80, 0.35),
    c_right=(0.95, 0.70, 0.20),
) -> np.ndarray:
    """
    Interpolate an RGB color field from four side colors using logical coordinates.
    C = 0.5 * [ (1-eta)Cb + eta Ct + (1-xi)Cl + xi Cr ]
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
    return C  # (Ny, Nx, 3)


def _row_segments(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Ny, Nx = X.shape
    segs = np.empty((Ny * (Nx - 1), 2, 2), float)
    k = 0
    for j in range(Ny):
        x0 = X[j, :-1]
        y0 = Y[j, :-1]
        x1 = X[j, 1:]
        y1 = Y[j, 1:]
        n = Nx - 1
        segs[k : k + n, 0, 0] = x0
        segs[k : k + n, 0, 1] = y0
        segs[k : k + n, 1, 0] = x1
        segs[k : k + n, 1, 1] = y1
        k += n
    return segs


def _col_segments(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Ny, Nx = X.shape
    segs = np.empty((Nx * (Ny - 1), 2, 2), float)
    k = 0
    for i in range(Nx):
        x0 = X[:-1, i]
        y0 = Y[:-1, i]
        x1 = X[1:, i]
        y1 = Y[1:, i]
        n = Ny - 1
        segs[k : k + n, 0, 0] = x0
        segs[k : k + n, 0, 1] = y0
        segs[k : k + n, 1, 0] = x1
        segs[k : k + n, 1, 1] = y1
        k += n
    return segs


def _row_segment_colors(C: np.ndarray) -> np.ndarray:
    Ny, Nx, _ = C.shape
    cols = np.empty((Ny * (Nx - 1), 4), float)
    k = 0
    for j in range(Ny):
        c0 = C[j, :-1, :]
        c1 = C[j, 1:, :]
        cc = 0.5 * (c0 + c1)
        n = Nx - 1
        cols[k : k + n, :3] = cc
        cols[k : k + n, 3] = 1.0
        k += n
    return cols


def _col_segment_colors(C: np.ndarray) -> np.ndarray:
    Ny, Nx, _ = C.shape
    cols = np.empty((Nx * (Ny - 1), 4), float)
    k = 0
    for i in range(Nx):
        c0 = C[:-1, i, :]
        c1 = C[1:, i, :]
        cc = 0.5 * (c0 + c1)
        n = Ny - 1
        cols[k : k + n, :3] = cc
        cols[k : k + n, 3] = 1.0
        k += n
    return cols


def animate_logical_to_real_grid_colored(
    XI: np.ndarray,
    ETA: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    out_gif: str = "mesh_morph_colored.gif",
    frames: int = 90,
    fps: int = 30,
    start_pause_s: float = 0.5,
    end_pause_s: float = 0.5,
    dpi: int = 200,
    lw: float = 1.0,
    show_points: bool = False,
    point_size: float = 6.0,
    side_colors=None,
):
    """
    Animates the structured grid from logical (XI,ETA) to real (X,Y) with:
      - start pause (hold first frame)
      - end pause (hold last frame)
      - color field interpolated from four side colors (boundary-defined RGB)

    XI,ETA,X,Y must have identical shapes (Ny,Nx).
    """

    if XI.shape != ETA.shape or X.shape != Y.shape or XI.shape != X.shape:
        raise ValueError("XI, ETA, X, Y must all have the same shape (Ny, Nx).")

    if side_colors is None:
        side_colors = {
            "bottom": (0.90, 0.20, 0.20),
            "top": (0.20, 0.40, 0.95),
            "left": (0.20, 0.80, 0.35),
            "right": (0.95, 0.70, 0.20),
        }

    # Start and end states
    X0, Y0 = XI, ETA
    X1, Y1 = X, Y

    Ny, Nx = XI.shape

    # Node colors fixed in logical space
    Cnodes = _colors_from_sides(
        XI,
        ETA,
        c_bottom=side_colors["bottom"],
        c_top=side_colors["top"],
        c_left=side_colors["left"],
        c_right=side_colors["right"],
    )
    row_cols = _row_segment_colors(Cnodes)
    col_cols = _col_segment_colors(Cnodes)

    start_hold = int(round(start_pause_s * fps))
    end_hold = int(round(end_pause_s * fps))
    total_frames = start_hold + frames + end_hold

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=dpi)
    ax.set_aspect("equal")

    # Limits to include both states
    xmin = min(X0.min(), X1.min())
    xmax = max(X0.max(), X1.max())
    ymin = min(Y0.min(), Y1.min())
    ymax = max(Y0.max(), Y1.max())
    pad_x = 0.05 * (xmax - xmin + 1e-12)
    pad_y = 0.05 * (ymax - ymin + 1e-12)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    title = ax.set_title("Logical → Real (t = 0.00)")
    ax.set_xlabel("x / ξ")
    ax.set_ylabel("y / η")

    # Initial collections (at t=0)
    row_lc = LineCollection(_row_segments(X0, Y0), linewidths=lw)
    col_lc = LineCollection(_col_segments(X0, Y0), linewidths=lw)
    row_lc.set_colors(row_cols)
    col_lc.set_colors(col_cols)
    ax.add_collection(row_lc)
    ax.add_collection(col_lc)

    scat = None
    if show_points:
        scat = ax.scatter(
            X0.ravel(),
            Y0.ravel(),
            s=point_size,
            c=Cnodes.reshape(-1, 3),
            marker=".",
            linewidths=0,
        )

    def update(frame: int):
        # --- time parameter with start+end pauses ---
        if frame < start_hold:
            t = 0.0
        elif frame < start_hold + frames:
            f = frame - start_hold
            t = f / (frames - 1) if frames > 1 else 1.0
        else:
            t = 1.0

        tt = ease_in_out(t)

        Xt = (1.0 - tt) * X0 + tt * X1
        Yt = (1.0 - tt) * Y0 + tt * Y1

        row_lc.set_segments(_row_segments(Xt, Yt))
        col_lc.set_segments(_col_segments(Xt, Yt))

        if scat is not None:
            scat.set_offsets(np.column_stack([Xt.ravel(), Yt.ravel()]))

        title.set_text(f"Logical → Real (t = {t:0.2f})")

        if scat is not None:
            return row_lc, col_lc, scat, title
        return row_lc, col_lc, title

    ani = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=True)
    ani.save(out_gif, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_gif


# Example usage:
# gif_path = animate_logical_to_real_grid_colored(
#     XI, ETA, X, Y,
#     out_gif="mesh_morph_colored.gif",
#     frames=100, fps=30,
#     start_pause_s=0.5, end_pause_s=0.5,
#     lw=1.0, show_points=False
# )
# print("Saved:", gif_path)