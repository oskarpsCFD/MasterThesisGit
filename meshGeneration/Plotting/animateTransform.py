import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def ease_in_out(t):
    return t*t*(3 - 2*t)

def animate_logical_to_real_grid(
    XI, ETA, X, Y,
    out_gif="logical_to_real_grid.gif",
    frames=80,
    fps=20,
    dpi=200,
    lw=0.8,
    show_points=False,
    point_size=6
):
    """
    Animate the full structured grid (lines) from logical to real.
    """

    Ny, Nx = XI.shape

    # precompute endpoints as arrays for interpolation
    X0, Y0 = XI, ETA
    X1, Y1 = X, Y

    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    ax.set_aspect("equal")

    # limits to include both
    xmin = min(X0.min(), X1.min()); xmax = max(X0.max(), X1.max())
    ymin = min(Y0.min(), Y1.min()); ymax = max(Y0.max(), Y1.max())
    pad_x = 0.05*(xmax-xmin + 1e-12)
    pad_y = 0.05*(ymax-ymin + 1e-12)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    title = ax.set_title("Logical → Real (t = 0.00)")
    ax.set_xlabel("x / ξ")
    ax.set_ylabel("y / η")

    # create line artists once
    row_lines = [ax.plot([], [], "-k", linewidth=lw)[0] for _ in range(Ny)]
    col_lines = [ax.plot([], [], "-k", linewidth=lw)[0] for _ in range(Nx)]

    if show_points:
        scat = ax.scatter([], [], s=point_size, marker=".", linewidths=0)
    else:
        scat = None

    def update(frame):
        t = frame/(frames-1)
        tt = ease_in_out(t)
        Xt = (1-tt)*X0 + tt*X1
        Yt = (1-tt)*Y0 + tt*Y1

        # update rows
        for j in range(Ny):
            row_lines[j].set_data(Xt[j, :], Yt[j, :])

        # update cols
        for i in range(Nx):
            col_lines[i].set_data(Xt[:, i], Yt[:, i])

        if scat is not None:
            P = np.column_stack([Xt.ravel(), Yt.ravel()])
            scat.set_offsets(P)

        title.set_text(f"Logical → Real (t = {t:0.2f})")

        if scat is not None:
            return (*row_lines, *col_lines, scat, title)
        return (*row_lines, *col_lines, title)

    ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)

    ani.save(out_gif, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_gif