# Libraries
from pathlib import Path

# Generate Mesh
from Geometry.Cases.Horseshoe import *

# Solve Mesh
from TFI.runTFI import TFI
from TTM.TTMSolver import TTMSolver
from TTM.ThomasMiddlecoff import *
from TTM.stitchBoundaryDictionary import createStitchBoundaryDict


# Generate Mesh files for OpenFOAM
from meshGeneration.CellAndPointGeneration import (
    extrudeTo3D, generatePoints,
    generateConnectivity, stitchBoundaryPoints)

from meshGeneration.CreateStructuredMesh import (
    writeStructuredMeshFiles, writeFoamHexMesh)

from meshGeneration.writeTopoSetPointFiles import writeBoundaryPoints

# Plotting
from Plotting.plotWeightFunctions import plotAllWeights
from Plotting.LogicalGridPlotting import plot_logical_and_real


CASE_PATH = "/home/oskarps/MasterOppgave/Simulation/testAerofoil/"
fileName = "symmetricAirfoil"


TEST_MODE =             False # if wanting to test with a test field

MeshResolution =        (121, 75)
OFFSET =                0.1     # Offset in the z direction
N_OUTER =               1       # outer iterations (recompute phi/psi from adapted grid)
N_INNER    =            2000    # max inner TTM iterations per outer step (Phase 2)
NEUMANN_ITERATIONS =    500     # Iterations where neumann boundary condition is applied
TOL  =                  1e-5    # inner convergence tolerance
A_XI       =            3       # weight amplification, ξ-direction
A_ETA      =            3       # weight amplification, η-direction
ISOTROPIC  =            True   # Boolean to tell if the w values should be distinct or magnitude
OMEGA      =            1.8    # Over relexation factor
PLOT_CURVES =           False
PLOT_TFI =              False
PLOT_WEIGHTS =          False   # Plots weights
PLOT_MESH =             False    # Plots mesh
TTM_MODE =              "XY"    # TTM mode solve in X, Y or rgular X and Y



Weight_Function_Cluster = [
    {'side': 'bottom', 'A': 100, 'B': 15}
]

NEUMANN_BOUNDARY =      np.array([True, True, True, True])


def GenerateMesh():
    Nx, Ny = MeshResolution
    infoString = f"Moving mesh with a resolution = ({Nx} x {Ny})\n"
    # Create logical space
    (xi, eta,
     XI, ETA,
     dxi, deta) = logicalSpace(Nx, Ny)

    # Create the curves based on parametrisation
    (botCurve, leftCurve,
     rightCurve, topCurve) = generateGeometry(Nx, Ny, plot=PLOT_CURVES)

    helperCruves = generateHelperCurves(NSample) # Highly dence lines

    infoString += f"Created helper-curves with {NSample} points\n"

    # Create an initial condition with TFI
    X, Y = TFI(botCurve, leftCurve,
               rightCurve, topCurve,
               XI, ETA, plot=PLOT_TFI)

    infoString += f"Generated initial condition using TFI\n"
    # Dictionary describing regions where a curve is stitched together (Annulus)
    stitchBoundaryDict, info = createStitchBoundaryDict(X, Y)


    # Create weight functions
    w1Base, w2Base = generateWeightFunctions(XI, ETA, clusters=Weight_Function_Cluster)

    infoString += "Created weight functions\n"

    infoString += (f"Running loop with outer iterations = {N_OUTER}\n"
                   f"Neumann iterations = {NEUMANN_ITERATIONS}")

    print(infoString)
    for outer in range(N_OUTER):
        phi, psi, w1, w2 = computeGridControlFunctions(X, Y,
                                                       testField,
                                                       w1Base, w2Base,
                                                       A_XI, A_ETA,
                                                       dxi, deta,
                                                       smooth_passes=0,
                                                       isotropic=ISOTROPIC,
                                                       testMode=TEST_MODE)

        # Solve for X and Y using Gauss Seidel iteration with over relaxation factors
        X, Y = TTMSolver(X, Y, dxi, deta,
                         psi, phi,
                         maxIter=N_INNER,
                         helperCurves=helperCruves,
                         neumann=NEUMANN_BOUNDARY,
                         neumann_iters=NEUMANN_ITERATIONS,
                         omega=OMEGA, tol=TOL,
                         stichedBoundaryDict=stitchBoundaryDict,
                         mode=TTM_MODE)


    if PLOT_WEIGHTS:
        infoString += f"Plotting weights\n"
        plotAllWeights(XI, ETA, phi, psi, w1, w2)
    if PLOT_MESH:
        infoString += f"Plotting mesh\n"
        plot_logical_and_real(XI, ETA, X, Y)

    if TEST_MODE:
        return 0

    # Create toposet files for OpenFOAM
    writeBoundaryPoints(X, Y, offset=OFFSET, path=CASE_PATH, stitchDict=stitchBoundaryDict)

    # Create pointfiles and hexes
    P0 = np.asarray(generatePoints(X, Y))
    C  = np.asarray(generateConnectivity(XI))

    # If geometry is stitched the point indices reflect it
    if stitchBoundaryDict:
        P, C, originalToFOAM = stitchBoundaryPoints(P0.copy(), C, X.shape, stitchBoundaryDict)

    else:
        P = P0
        originalToFOAM = np.arange(0, len(P0))

    # Create the structuredMesh files for the openfoam directory
    writeStructuredMeshFiles(
        original_points=P0,
        original_to_foam_point=originalToFOAM,
        nx=X.shape[1],
        ny=X.shape[0],
        folder=Path(CASE_PATH) / "constant" / "structuredMesh.Orig",
        ordering="rowMajor"
    )

    # Make the point and the connectivity 3D for OpenFOAM (2D volumes)
    P, C = extrudeTo3D(P, C, onlyP=False, offset=OFFSET)

    # Create the file for multiple block creator
    writeFoamHexMesh(P, C, CASE_PATH + fileName)


GenerateMesh()
