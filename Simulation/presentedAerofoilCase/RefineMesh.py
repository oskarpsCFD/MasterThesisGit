from pathlib import Path

from Geometry.Cases.SymmetricAerofoil import *

from TTM.TTMSolver import WinslowSolver
from TTM.ThomasMiddlecoff import *
from TTM.stitchBoundaryDictionary import createStitchBoundaryDict
from TTM.BoundaryProjection import build_polyline_tree

from meshGenerationFiles.CellPointGeneration import generatePoints
from meshGenerationFiles.WriteStructuredMeshFiles import write_structured_mesh_files
from meshGenerationFiles.generatePointFields import rewriteOpenFOAMPoints
from meshGenerationFiles.loadData import loadOrGenerateHelpers

CASE_PATH = "/home/oskarps/MasterOppgave/Simulation/refining/testingBoundary/"

CURVE_CACHE_PATH  = Path(CASE_PATH).parent / "helperCurves_cache.npz"
DUCROS_CACHE_DIR  = Path(CASE_PATH).parent / "ducros_cache"
PHI_CACHE_DIR     = Path(CASE_PATH).parent / "phi_cache"

MeshResolution       = (121, 75)
N_INNER              = 1000     # max Winslow iterations per outer step
NEUMANN_ITERATIONS   = 0        # iterations where Neumann BC is applied
TOL                  = 1e-5     # convergence tolerance
ISOTROPIC            = False
OMEGA                = 1.0      # SOR relaxation factor
WINSLOW_MODE         = "XY"
RECOMPUTE_INTERVAL   = 10       # re-interpolate phi/psi every N iterations (0 = disabled)
C2_Xi                = 2
C2_Eta               = 0
C_DUCROS             = 1.0      # Ducros sensor amplification factor
XI_CUTOFF            = 1.0      # logical xi beyond which field has no effect
XI_RAMP_WIDTH        = 0.1
ETA_CUTOFF           = 1.0      # logical eta beyond which field has no effect
ETA_RAMP_WIDTH       = 0.1

Weight_Function_Cluster = [
    {'side': 'bottom', 'A': 100, 'B': 15}
]

NEUMANN_BOUNDARY     = np.array([True, False, False, False])
NEUMANN_BOT_FRACTION = 1

_refine_step: int = 0


def _jameson_sensor(p: NDArray) -> NDArray:
    """Jameson pressure-based shock sensor, active only in compression regions."""
    p_im1 = p[:, :-2]
    p_i   = p[:, 1:-1]
    p_ip1 = p[:, 2:]
    f_int = np.abs(p_ip1 - 2.0*p_i + p_im1) / (np.abs(p_ip1 + 2.0*p_i + p_im1) + 1e-12)
    f_raw = np.zeros_like(p)
    f_raw[:, 1:-1] = np.where(p_im1 > p_ip1, f_int, 0.0)
    nonzero = f_raw[f_raw > 0]
    p99 = float(np.percentile(nonzero, 99.99)) if nonzero.size > 0 else 1.0
    return np.clip(f_raw / (p99 + 1e-12), 0.0, 1.0)


def _save_step_cache(step: int, X: NDArray, Y: NDArray,
                     f_ducros: NDArray, f_jameson: NDArray, phi: NDArray):
    DUCROS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PHI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DUCROS_CACHE_DIR / f"step_{step:04d}.npz",
             X=X, Y=Y, ducros=f_ducros, jameson=f_jameson)
    np.savez(PHI_CACHE_DIR / f"step_{step:04d}.npz",
             X=X, Y=Y, phi=phi)


def RefineMesh(P_Structured: NDArray,
               P_OpenFOAM: NDArray,
               structToFOAM: NDArray,
               Field: NDArray,
               logicalI: NDArray,
               logicalJ: NDArray):

    Nx, Ny = MeshResolution

    xi, eta, XI, ETA, dxi, deta = logicalSpace(Nx, Ny)

    N = Field[0].shape[0]
    Ux, Uy, p, w = (Field[0][:N//2, 0],
                    Field[0][:N//2, 1],
                    Field[1][:N//2],
                    Field[2][:N//2])

    X, Y, Ux_local, Uy_local, p, w, grad_U_mag, grad_p_x, grad_p_y, f_ducros = \
        rewriteOpenFOAMPoints(P_OpenFOAM, P_Structured, structToFOAM,
                              logicalI, logicalJ, Nx, Ny, Ux, Uy, p, w)

    f_jameson = _jameson_sensor(p)

    helperCurves = loadOrGenerateHelpers(CURVE_CACHE_PATH, generateHelperCurves, NSample)
    tree_cache = {side: build_polyline_tree(np.column_stack((xc, yc)))
                  for side, (xc, yc) in helperCurves.items()}

    stitchBoundaryDict = createStitchBoundaryDict(X, Y)

    w1Base, w2Base = generateWeightFunctions(XI, ETA, clusters=Weight_Function_Cluster)

    _phi_psi_kwargs = dict(
        C2_xi=C2_Xi,
        C2_eta=C2_Eta,
        dxi=dxi, deta=deta,
        smooth_passes=0,
        isotropic=ISOTROPIC,
        percentile_clip=95,
        xi_cutoff=XI_CUTOFF,
        xi_ramp_width=XI_RAMP_WIDTH,
        eta_cutoff=ETA_CUTOFF,
        eta_ramp_width=ETA_RAMP_WIDTH,
        ducros_sensor=f_jameson,
        C_ducros=C_DUCROS,
    )

    phi, psi, w1, w2 = computeThomasMiddlecoff(
        X, Y, grad_p_x, grad_p_y, w1Base, w2Base, **_phi_psi_kwargs)

    global _refine_step
    _save_step_cache(_refine_step, X, Y, f_ducros, f_jameson, phi)
    _refine_step += 1

    phi_psi_updater = None
    if RECOMPUTE_INTERVAL > 0:
        grad_p_pts   = np.column_stack([X.ravel(), Y.ravel()])
        grad_p_field = np.column_stack([grad_p_x.ravel(), grad_p_y.ravel()])
        interp_gx, interp_gy = build_field_interpolator(grad_p_pts, grad_p_field)
        phi_psi_updater = lambda Xc, Yc: reinterpolate_phi_psi(
            Xc, Yc, interp_gx, interp_gy, w1Base, w2Base, **_phi_psi_kwargs)

    X, Y = WinslowSolver(X, Y, dxi, deta, psi, phi,
                         frac=frac,
                         maxIter=N_INNER,
                         helperCurves=helperCurves,
                         neumann=NEUMANN_BOUNDARY,
                         neumann_iters=NEUMANN_ITERATIONS,
                         omega=OMEGA, tol=TOL,
                         stichedBoundaryDict=stitchBoundaryDict,
                         mode=WINSLOW_MODE,
                         phi_psi_updater=phi_psi_updater,
                         recompute_interval=RECOMPUTE_INTERVAL,
                         w1=w1, w2=w2,
                         neumann_bot_fraction=NEUMANN_BOT_FRACTION)

    P0 = np.asarray(generatePoints(X, Y))

    N_foam        = P_OpenFOAM.shape[0] // 2
    P_new_foam_2D = P_OpenFOAM[:N_foam, :2].copy()
    P_new_foam_2D[np.asarray(structToFOAM, dtype=int)] = P0

    deformation = np.zeros_like(P_OpenFOAM)
    deformation[:N_foam, :2] = P_new_foam_2D - P_OpenFOAM[:N_foam, :2]
    deformation[N_foam:, :2] = P_new_foam_2D - P_OpenFOAM[N_foam:, :2]

    write_structured_mesh_files(
        original_points=P0,
        original_to_foam_point=structToFOAM,
        nx=X.shape[1],
        ny=X.shape[0],
        folder="constant/structuredMesh",
        ordering="rowMajor"
    )

    return deformation
