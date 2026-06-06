import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


def generatePoints(X: NDArray, Y: NDArray) -> NDArray:
    return np.stack((X, Y), axis=-1).reshape(-1, 2)


def quadArea(p):
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def generateConnectivity(Xi: NDArray) -> list:
    Nj, Ni = Xi.shape

    def idx(i, j):
        return i + Ni * j

    cells = []
    for j in range(Nj - 1):
        for i in range(Ni - 1):
            v0 = idx(i,     j)
            v1 = idx(i + 1, j)
            v2 = idx(i + 1, j + 1)
            v3 = idx(i,     j + 1)
            cells.append((v0, v1, v2, v3))
    return cells

def stitchBoundaryPoints(points, cells, shape, stitchMapDictionary, tol=1e-10):
    """
    Merge coincident boundary points and update connectivity.

    Uses a precomputed stitchMapDictionary to determine which boundary index
    pairs should be merged. The dictionary maps boundary side names
    ('bot', 'top', 'left', 'right') to an (M, 2) array of index pairs,
    where each row [i0, i1] means the point at local boundary index i0
    should be unioned with the point at local boundary index i1.

    A KDTree step then handles any remaining coincident boundary pairs
    (corners, back-boundary endpoints) using coordinate tolerance.

    Parameters
    ----------
    points            : (N, 2) or (N, 3)
    cells             : (M, 4)
    shape             : (Nj, Ni)
    stitchMapDictionary : dict mapping side -> (M, 2) int array of index pairs
    tol               : coordinate tolerance for KDTree merge of non-stitched pairs

    Returns
    -------
    new_points, new_cells, old_to_new
    """
    points = np.asarray(points, dtype=float)
    cells  = np.asarray(cells,  dtype=np.int64)

    Nj, Ni = shape
    N = len(points)

    if N != Nj * Ni:
        raise ValueError(f"points has length {N}, but shape {shape} implies {Nj * Ni}")

    def idx(i, j):
        return i + Ni * j

    # Collect boundary point indices
    boundary_set = set()
    for i in range(Ni):
        boundary_set.add(idx(i, 0))
        boundary_set.add(idx(i, Nj - 1))
    for j in range(Nj):
        boundary_set.add(idx(0, j))
        boundary_set.add(idx(Ni - 1, j))
    boundary = np.array(sorted(boundary_set), dtype=np.int64)

    # Union-find
    rep = np.arange(N, dtype=np.int64)

    def find(a):
        while rep[a] != a:
            rep[a] = rep[rep[a]]
            a = rep[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                rep[rb] = ra
            else:
                rep[ra] = rb

    # ------------------------------------------------------------------
    # Union stitched pairs from the dictionary.
    # For each side, the index pairs are local boundary-row/column indices.
    # We convert them to flat point indices using idx().
    # Negative indices are supported (e.g. -1 means Ni-1 or Nj-1).
    # ------------------------------------------------------------------
    for side, index_pairs in stitchMapDictionary.items():
        for i0, i1 in index_pairs:
            if side == "bot":
                # bot boundary: j=0, i varies
                i0_norm = int(i0) % Ni
                i1_norm = int(i1) % Ni
                union(idx(i0_norm, 0), idx(i1_norm, 0))
            elif side == "top":
                # top boundary: j=Nj-1, i varies
                i0_norm = int(i0) % Ni
                i1_norm = int(i1) % Ni
                union(idx(i0_norm, Nj - 1), idx(i1_norm, Nj - 1))
            elif side == "left":
                # left boundary: i=0, j varies
                j0_norm = int(i0) % Nj
                j1_norm = int(i1) % Nj
                union(idx(0, j0_norm), idx(0, j1_norm))
            elif side == "right":
                # right boundary: i=Ni-1, j varies
                j0_norm = int(i0) % Nj
                j1_norm = int(i1) % Nj
                union(idx(Ni - 1, j0_norm), idx(Ni - 1, j1_norm))
            else:
                raise ValueError(f"Unknown boundary side '{side}' in stitchMapDictionary")

    # ------------------------------------------------------------------
    # KDTree merge for all remaining coincident boundary pairs
    # (corners, any floating-point exact matches)
    # ------------------------------------------------------------------
    bpts = points[boundary]
    tree = cKDTree(bpts)
    for i, j in tree.query_pairs(r=tol):
        union(boundary[i], boundary[j])

    old_to_rep  = np.array([find(i) for i in range(N)], dtype=np.int64)
    kept        = np.array(sorted(set(old_to_rep.tolist())), dtype=np.int64)
    rep_to_new  = {r: i for i, r in enumerate(kept)}
    old_to_new  = np.array([rep_to_new[r] for r in old_to_rep], dtype=np.int64)

    new_points = points[kept]
    new_cells  = old_to_new[cells]

    # Remove degenerate cells (fewer than 4 unique vertices)
    mask      = np.array([len(set(cell)) == 4 for cell in new_cells], dtype=bool)
    new_cells = new_cells[mask]

    return new_points, new_cells, old_to_new


def extrudeTo3D(points, cells, onlyP: bool = True, offset: float = 0.1):
    if not onlyP:
        shapeP = points.shape
        shapeC = cells.shape

        P3D = np.zeros((shapeP[0] * 2, 3))
        P3D[:shapeP[0], :2] = points
        P3D[shapeP[0]:, :2] = points
        P3D[shapeP[0]:, 2]  = offset

        for i, c in enumerate(cells):
            quad = points[c]
            if quadArea(quad) < 0:
                cells[i] = c[::-1]

        C3D = np.zeros((shapeC[0], shapeC[1] * 2), dtype=int)
        C3D[:, :shapeC[1]] = cells
        C3D[:, shapeC[1]:] = cells + shapeP[0]

        return P3D, C3D

    else:
        shapeP = points.shape
        P3D = np.zeros((shapeP[0] * 2, 3))
        P3D[:shapeP[0], :2] = points
        P3D[shapeP[0]:, :2] = points
        P3D[shapeP[0]:, 2]  = 0.1
        return P3D


