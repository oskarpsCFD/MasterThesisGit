def debug_plot_cells(
    points: NDArray,
    cells: NDArray,
    show_point_ids: bool = False,
    show_cell_ids: bool = False,
    close_cells: bool = False,
    ax=None,
):
    """
    Plot a 2D structured/unstructured quad mesh from points and cell connectivity.

    Parameters
    ----------
    points : (N, 2) array
        Point coordinates.
    cells : (M, 4) array
        Quad connectivity. Each row contains 4 point indices.
    show_point_ids : bool
        If True, annotate point indices.
    show_cell_ids : bool
        If True, annotate cell indices at the quad center.
    close_cells : bool
        If True, draw the final closing edge from last point to first point.
    ax : matplotlib axis or None
        Existing axis to plot into. If None, create a new figure.
    """
    points = np.asarray(points)
    cells = np.asarray(cells, dtype=int)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"'points' must have shape (N, 2), got {points.shape}")

    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError(f"'cells' must have shape (M, 4), got {cells.shape}")

    n_points = len(points)
    if np.any(cells < 0) or np.any(cells >= n_points):
        bad = np.argwhere((cells < 0) | (cells >= n_points))
        raise ValueError(f"Connectivity contains invalid point indices at entries: {bad[:10]}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], s=20)

    # Plot point ids
    if show_point_ids:
        for i, (x, y) in enumerate(points):
            ax.text(x, y, str(i), fontsize=8, ha="left", va="bottom")

    # Plot cells
    for ci, cell in enumerate(cells):
        poly_pts = points[cell]

        # Draw polygon
        polygon = Polygon(poly_pts, closed=close_cells, fill=False, linewidth=1.2)
        ax.add_patch(polygon)

        # Mark vertices in order
        for local_id, pidx in enumerate(cell):
            x, y = points[pidx]
            ax.text(
                x, y,
                f"{pidx}",
                fontsize=8,
                ha="center",
                va="center"
            )

        # Plot cell id at centroid
        if show_cell_ids:
            cx = np.mean(poly_pts[:, 0])
            cy = np.mean(poly_pts[:, 1])
            ax.text(cx, cy, f"C{ci}", fontsize=9, ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_title("Cell Connectivity Debug Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    plt.show()