import numpy as np
from typing import Tuple
from numpy.typing import NDArray


def scatter_to_grid(
    values: NDArray,
    logicalI: NDArray,
    logicalJ: NDArray,
    Nx: int,
    Ny: int,
    fill_value=np.nan,
) -> NDArray:
    """
    Scatter 1D structured values onto a logical (Ny, Nx) grid.

    values:
        shape (N,) or (N, ncomp)
    logicalI, logicalJ:
        shape (N,)
    """
    values = np.asarray(values)
    logicalI = np.asarray(logicalI, dtype=int)
    logicalJ = np.asarray(logicalJ, dtype=int)

    if values.shape[0] != logicalI.shape[0] or values.shape[0] != logicalJ.shape[0]:
        raise ValueError(
            f"Inconsistent lengths: values={values.shape[0]}, "
            f"logicalI={logicalI.shape[0]}, logicalJ={logicalJ.shape[0]}"
        )

    if np.any(logicalI < 0) or np.any(logicalI >= Nx):
        raise ValueError("logicalI contains out-of-range indices")
    if np.any(logicalJ < 0) or np.any(logicalJ >= Ny):
        raise ValueError("logicalJ contains out-of-range indices")

    if values.ndim == 1:
        out = np.full((Ny, Nx), fill_value, dtype=float if np.issubdtype(values.dtype, np.integer) else values.dtype)
        out[logicalJ, logicalI] = values
        return out

    elif values.ndim == 2:
        out = np.full(
            (Ny, Nx, values.shape[1]),
            fill_value,
            dtype=float if np.issubdtype(values.dtype, np.integer) else values.dtype,
        )
        out[logicalJ, logicalI, :] = values
        return out

    else:
        raise ValueError(f"values must have ndim 1 or 2, got {values.ndim}")


def _compute_mesh_metrics(X: NDArray, Y: NDArray):
    """
    Jacobian components of the (ξ,η) → (x,y) coordinate transformation.
    All derivatives are with respect to grid indices (unit computational step).
    """
    dXdxi  = np.gradient(X, axis=1)
    dYdxi  = np.gradient(Y, axis=1)
    dXdeta = np.gradient(X, axis=0)
    dYdeta = np.gradient(Y, axis=0)
    detJ   = dXdxi * dYdeta - dXdeta * dYdxi
    n_neg = int(np.sum(detJ <= 0))
    if n_neg > 0:
        print(f"[WARNING] {n_neg}/{detJ.size} grid cells have non-positive Jacobian (detJ ≤ 0)")
    return dXdxi, dYdxi, dXdeta, dYdeta, detJ


def _physical_gradient(
    phi: NDArray,
    dXdxi: NDArray, dYdxi: NDArray,
    dXdeta: NDArray, dYdeta: NDArray,
    detJ: NDArray,
) -> Tuple[NDArray, NDArray]:
    """
    Physical gradient (∂φ/∂x, ∂φ/∂y) via inverse Jacobian.

    Cells with |detJ| below 1e-30 (inverted or degenerate) are set to zero
    gradient so they do not produce NaN/inf values that corrupt the weight
    function and break the downstream TDMA solve.
    """
    dphi_dxi  = np.gradient(phi, axis=1)
    dphi_deta = np.gradient(phi, axis=0)
    safe_detJ = np.where(np.abs(detJ) > 1e-30, detJ,
                         np.copysign(1e-30, detJ + 1e-300))
    dphi_dx = ( dYdeta * dphi_dxi  - dYdxi  * dphi_deta) / safe_detJ
    dphi_dy = (-dXdeta * dphi_dxi  + dXdxi  * dphi_deta) / safe_detJ
    # Zero out gradient at inverted/degenerate cells rather than propagate garbage.
    bad = np.abs(detJ) <= 1e-30
    dphi_dx[bad] = 0.0
    dphi_dy[bad] = 0.0
    return dphi_dx, dphi_dy


def rewriteOpenFOAMPoints(
    P_OF: NDArray,
    P_Structured: NDArray,
    structToFOAM: NDArray,
    logicalI: NDArray,
    logicalJ: NDArray,
    Nx: int,
    Ny: int,
    Ux_OF: NDArray,
    Uy_OF: NDArray,
    p_OF: NDArray,
    w_OF: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Reconstruct OpenFOAM fields onto the structured grid, transform velocity
    to local grid-aligned coordinates, and compute physical gradient fields.

    Parameters
    ----------
    P_OF : (Nfoam, 3)
        OpenFOAM mesh points.
    P_Structured : (Nstruct, 3)
        Structured logical points (including seam duplicates).
    structToFOAM : (Nstruct,)
        Maps structured point j → OpenFOAM point index.
    logicalI, logicalJ : (Nstruct,)
        Logical grid indices for each structured point.
    Nx, Ny : int
        Structured grid dimensions.
    Ux_OF, Uy_OF : (Nfoam,)
        Global Cartesian velocity components from OpenFOAM.
    p_OF : (Nfoam,)
        Static pressure from OpenFOAM.
    w_OF : (Nfoam,)
        Specific dissipation rate ω from OpenFOAM.

    Returns
    -------
    X, Y : (Ny, Nx)
        Physical coordinates on the structured grid.
    Ux_local, Uy_local : (Ny, Nx)
        Velocity components projected onto the local ξ and η grid directions.
    p, w : (Ny, Nx)
        Pressure and ω on the structured grid.
    grad_U_mag : (Ny, Nx)
        Frobenius norm of the physical velocity gradient tensor |∇U|.
    grad_p_x, grad_p_y : (Ny, Nx)
        Physical pressure gradient vector components ∂p/∂x and ∂p/∂y.
    f_ducros : (Ny, Nx)
        Ducros shock sensor (div u)² / ((div u)² + |curl u|² + ε), in [0, 1].
        Values near 1 indicate a shock; values near 0 indicate vortex-dominated flow.
    """
    P_OF          = np.asarray(P_OF)
    P_Structured  = np.asarray(P_Structured)
    structToFOAM  = np.asarray(structToFOAM, dtype=int)
    logicalI      = np.asarray(logicalI, dtype=int)
    logicalJ      = np.asarray(logicalJ, dtype=int)
    Ux_OF         = np.asarray(Ux_OF, dtype=float)
    Uy_OF         = np.asarray(Uy_OF, dtype=float)
    p_OF          = np.asarray(p_OF,  dtype=float)
    w_OF          = np.asarray(w_OF,  dtype=float)

    n_struct = P_Structured.shape[0]
    if structToFOAM.shape[0] != n_struct:
        raise ValueError("structToFOAM length must match number of structured points")
    if logicalI.shape[0] != n_struct or logicalJ.shape[0] != n_struct:
        raise ValueError("logicalI/logicalJ length must match number of structured points")
    if Nx * Ny != n_struct:
        raise ValueError(f"Nx*Ny = {Nx*Ny}, but number of structured points is {n_struct}")
    if np.any(structToFOAM < 0) or np.any(structToFOAM >= P_OF.shape[0]):
        raise ValueError("structToFOAM contains out-of-range OpenFOAM point indices")

    # Map OpenFOAM fields to structured ordering
    Ux_s = Ux_OF[structToFOAM]
    Uy_s = Uy_OF[structToFOAM]
    p_s  = p_OF[structToFOAM]
    w_s  = w_OF[structToFOAM]

    # Scatter coordinates and fields onto the logical grid
    XY = scatter_to_grid(P_Structured[:, :2], logicalI, logicalJ, Nx, Ny)
    X  = XY[:, :, 0]
    Y  = XY[:, :, 1]

    Ux = scatter_to_grid(Ux_s, logicalI, logicalJ, Nx, Ny)
    Uy = scatter_to_grid(Uy_s, logicalI, logicalJ, Nx, Ny)
    p  = scatter_to_grid(p_s,  logicalI, logicalJ, Nx, Ny)
    w  = scatter_to_grid(w_s,  logicalI, logicalJ, Nx, Ny)

    # Mesh metrics
    dXdxi, dYdxi, dXdeta, dYdeta, detJ = _compute_mesh_metrics(X, Y)

    # Local velocity: project global (Ux, Uy) onto unit tangent vectors of the
    # ξ (i-index) and η (j-index) grid directions
    mag_xi  = np.maximum(np.sqrt(dXdxi**2  + dYdxi**2),  1e-30)
    mag_eta = np.maximum(np.sqrt(dXdeta**2 + dYdeta**2), 1e-30)
    Ux_local = (Ux * dXdxi  + Uy * dYdxi)  / mag_xi
    Uy_local = (Ux * dXdeta + Uy * dYdeta) / mag_eta

    # Physical gradient magnitudes
    dUx_dx, dUx_dy = _physical_gradient(Ux, dXdxi, dYdxi, dXdeta, dYdeta, detJ)
    dUy_dx, dUy_dy = _physical_gradient(Uy, dXdxi, dYdxi, dXdeta, dYdeta, detJ)
    grad_U_mag = np.sqrt(dUx_dx**2 + dUx_dy**2 + dUy_dx**2 + dUy_dy**2)

    grad_p_x, grad_p_y = _physical_gradient(p, dXdxi, dYdxi, dXdeta, dYdeta, detJ)

    # Ducros shock sensor: (div u)² / ((div u)² + |curl u|² + ε)
    # High near shocks (compressive, irrotational); low in turbulent/vortical regions.
    div_u  = dUx_dx + dUy_dy
    curl_u = dUy_dx - dUx_dy
    f_ducros = div_u**2 / (div_u**2 + curl_u**2 + 1e-10)

    return X, Y, Ux_local, Uy_local, p, w, grad_U_mag, grad_p_x, grad_p_y, f_ducros
