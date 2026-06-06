def plot_logical_grid(XI, ETA, ax=None, show_points=True):
    """
    Plot a structured logical grid given meshgrid matrices XI, ETA.

    XI, ETA : arrays of shape (Ny, Nx)
    """
    linew = 0.4

    if ax is None:
        fig, ax = plt.subplots()

    Ny, Nx = XI.shape

    # constant eta (rows)
    for j in range(Ny):
        ax.plot(XI[j, :], ETA[j, :], '-k', linewidth=linew)

    # constant xi (columns)
    for i in range(Nx):
        ax.plot(XI[:, i], ETA[:, i], '-k', linewidth=linew)

    if show_points:
        ax.plot(XI, ETA, 'k.', markersize=4)

    ax.set_aspect("equal")
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$\eta$')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title('Logical Space')
    return ax

def plot_real_grid(X, Y, ax=None, show_points=True):
    """
    Plot a structured logical grid given meshgrid matrices XI, ETA.

    XI, ETA : arrays of shape (Ny, Nx)
    """
    import matplotlib.pyplot as plt
    linew = 0.4
    if ax is None:
        fig, ax = plt.subplots()

    Ny, Nx = X.shape

    # constant eta (rows)
    for j in range(Ny):
        ax.plot(X[j, :], Y[j, :], '-k', linewidth=linew)

    # constant xi (columns)
    for i in range(Nx):
        ax.plot(X[:, i], Y[:, i], '-k', linewidth=linew)

    if show_points:
        ax.plot(X, Y, 'k.', markersize=0.1)

    ax.set_aspect("equal")
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title('Real Space')
    plt.show()
    return ax

def plot_real_grid_local(X, Y,
                         xlim=None, ylim=None,
                         ax=None,
                         show_points=False,
                         lw=0.4,
                         point_size=2,
                         title=None,
                         dpi=200,
                         figsize=(6, 6)):
    """
    Plot the real-space structured grid with optional axis limits for zooming
    into a local region.

    Parameters
    ----------
    X, Y    : (Ny, Nx) node coordinate arrays
    xlim    : (xmin, xmax) or None  — horizontal window
    ylim    : (ymin, ymax) or None  — vertical window
    ax      : existing Axes to draw on, or None to create a new figure
    show_points : mark grid nodes with dots
    lw      : line width for grid lines
    point_size  : dot size when show_points=True
    title   : axes title (default "Real Space")
    dpi, figsize : figure resolution and size (ignored when ax is given)
    """
    import matplotlib.pyplot as plt

    standalone = ax is None
    if standalone:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": dpi,
        })
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    Ny, Nx = X.shape

    for j in range(Ny):
        ax.plot(X[j, :], Y[j, :], '-k', linewidth=lw)
    for i in range(Nx):
        ax.plot(X[:, i], Y[:, i], '-k', linewidth=lw)

    if show_points:
        ax.plot(X, Y, 'k.', markersize=point_size)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_aspect("equal")
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(title if title is not None else 'Real Space')

    if standalone:
        fig.tight_layout()
        plt.show()
        return fig, ax

    return ax


def plot_boundaries_logical_and_real(XI, ETA, X, Y, show_points=False):
    """
    Plot only the four boundary curves (bottom, top, left, right) in logical
    space (left panel) and real space (right panel).
    """
    import matplotlib.pyplot as plt

    side_colors = {
        "bottom": (0.20, 0.55, 0.80),
        "top":    (0.90, 0.45, 0.15),
        "left":   (0.25, 0.70, 0.45),
        "right":  (0.60, 0.45, 0.80),
    }

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    })

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    boundaries = [
        ("bottom", XI[0, :],  ETA[0, :],  X[0, :],  Y[0, :]),
        ("top",    XI[-1, :], ETA[-1, :], X[-1, :], Y[-1, :]),
        ("left",   XI[:, 0],  ETA[:, 0],  X[:, 0],  Y[:, 0]),
        ("right",  XI[:, -1], ETA[:, -1], X[:, -1], Y[:, -1]),
    ]

    for name, xi, eta, x, y in boundaries:
        c = side_colors[name]
        axs[0].plot(xi, eta, color=c, linewidth=2.0, label=name)
        axs[1].plot(x,  y,   color=c, linewidth=2.0, label=name)
        if show_points:
            axs[0].plot(xi, eta, '.', color=c, markersize=4)
            axs[1].plot(x,  y,   '.', color=c, markersize=4)

    for ax, xlabel, ylabel, title in [
        (axs[0], r'$\xi$', r'$\eta$', 'Logical Space'),
        (axs[1], 'x',     'y',        'Real Space'),
    ]:
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    plt.show()
    return fig, axs


def plot_logical_and_real(XI, ETA, X, Y, show_points=False):
    """
    Plot logical grid (XI, ETA) and real grid (X, Y) side by side.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.figsize": [6.5, 4],
        "figure.dpi": 300,
        "image.cmap": "Grays"
    })
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    plot_logical_grid(XI, ETA, ax=axs[0], show_points=show_points)
    plot_real_grid(X, Y, ax=axs[1], show_points=show_points)

    fig.tight_layout()
    plt.show()
    return fig, axs
