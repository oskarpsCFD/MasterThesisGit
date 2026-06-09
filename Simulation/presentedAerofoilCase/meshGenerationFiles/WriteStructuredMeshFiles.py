from pathlib import Path
import numpy as np


def _foam_header(cls: str, location: str, obj: str) -> str:
    return f"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2312                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
    FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{location}";
    object      {obj};
}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def write_point_field(points, file_path, location="constant/structuredMesh", object_name=None):
    """
    Write OpenFOAM pointField file.

    points: array-like of shape (N, 3) or (N, 2)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if object_name is None:
        object_name = file_path.name

    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError("points must have shape (N,2) or (N,3)")

    if pts.shape[1] == 2:
        z = np.zeros((pts.shape[0], 1), dtype=float)
        pts = np.hstack((pts, z))

    with open(file_path, "w") as f:
        f.write(_foam_header("vectorField", location, object_name))
        f.write(f"{len(pts)}\n(\n")
        for p in pts:
            f.write(f"({p[0]:.16g} {p[1]:.16g} {p[2]:.16g})\n")
        f.write(")\n")


def write_label_list(labels, file_path, location="constant/structuredMesh", object_name=None):
    """
    Write OpenFOAM labelList file.

    labels: array-like of shape (N,)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if object_name is None:
        object_name = file_path.name

    arr = np.asarray(labels)

    if arr.ndim != 1:
        raise ValueError("labels must be a 1D array/list")

    arr = arr.astype(np.int64)

    with open(file_path, "w") as f:
        f.write(_foam_header("labelList", location, object_name))
        f.write(f"{len(arr)}\n(\n")
        for v in arr:
            f.write(f"{int(v)}\n")
        f.write(")\n")


def make_logical_ij(nx, ny, ordering="rowMajor"):
    """
    Returns logicalI, logicalJ for point ordering.

    rowMajor: k = j*nx + i
    colMajor: k = i*ny + j
    """
    if ordering == "rowMajor":
        logical_i = []
        logical_j = []
        for j in range(ny):
            for i in range(nx):
                logical_i.append(i)
                logical_j.append(j)

    elif ordering == "colMajor":
        logical_i = []
        logical_j = []
        for i in range(nx):
            for j in range(ny):
                logical_i.append(i)
                logical_j.append(j)
    else:
        raise ValueError("ordering must be 'rowMajor' or 'colMajor'")

    return np.asarray(logical_i, dtype=np.int64), np.asarray(logical_j, dtype=np.int64)


def write_structured_mesh_files(
    original_points,
    original_to_foam_point,
    nx,
    ny,
    folder,
    ordering="rowMajor"
):
    """
    Write:
      - originalPoints
      - originalToFoamPoint
      - logicalI
      - logicalJ
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    logical_i, logical_j = make_logical_ij(nx, ny, ordering=ordering)

    npts = len(np.asarray(original_points))
    if len(original_to_foam_point) != npts:
        raise ValueError("original_to_foam_point length must equal number of original points")
    if len(logical_i) != npts:
        raise ValueError("nx*ny must equal number of original points")

    write_point_field(original_points, folder / "originalPoints")
    write_label_list(original_to_foam_point, folder / "originalToFoamPoint")
    write_label_list(logical_i, folder / "logicalI")
    write_label_list(logical_j, folder / "logicalJ")
