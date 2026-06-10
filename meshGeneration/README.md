# meshGeneration

Offline Python tool for generating structured hexahedral meshes. It produces the mesh files required by the `hexahedralMeshGenerator` OpenFOAM utility and the metadata consumed by the runtime motion solver.

---

## Entry Point

```
meshGeneration/
└── GenerateMesh.py     ← run this script
```

Run from the `meshGeneration/` directory:

```sh
python GenerateMesh.py
```

Edit the constants at the top of the file before running (see [Parameters](#parameters)).

---

## Pipeline

```
Parametric boundary curves
        │
        ▼
  Transfinite Interpolation (TFI)
        │  initial mesh (algebraic, no orthogonality guarantee)
        ▼
  Thomas–Middlecoff (TTM) elliptic smoother
        │  iteratively redistributes interior points guided by
        │  weight functions and attraction potentials (φ, ψ)
        ▼
  Output files for OpenFOAM
```

### Transfinite Interpolation (`TFI/runTFI.py`)

Blends the four boundary curves to fill the interior algebraically. Provides the initial condition for the TTM solver. Fast but does not guarantee orthogonality.

### Thomas–Middlecoff Solver (`TTM/TTMSolver.py`, `TTM/ThomasMiddlecoff.py`)

Iterates the elliptic grid-generation PDEs using Gauss–Seidel with successive over-relaxation (SOR). The attraction potentials φ and ψ are computed from user-supplied weight functions that can cluster grid lines toward domain boundaries or high-gradient flow regions.

Supported solver modes (`TTM_MODE`):

| Mode | Effect |
|------|--------|
| `"XY"` | Solve both X and Y (default) |
| `"X"` | Solve X only; hold Y fixed |
| `"Y"` | Solve Y only; hold X fixed |

Neumann boundary conditions (ghost-cell substitution) allow boundary points to slide freely along a prescribed curve, useful for open or periodic boundaries.

---

## Geometry Cases

Each case lives in `Geometry/Cases/` and defines the four boundary curves plus helper functions.

| File | Domain |
|------|--------|
| `SymmetricAerofoil.py` | C-topology around a symmetric aerofoil |
| `Horseshoe.py` | Annular domain between two concentric C-shaped curves |
| `ConcentricCylinder.py` | Annulus between two concentric circles |
| `UnitSquare.py` | Simple unit-square test domain |

To add a new geometry, create a new file in `Geometry/Cases/` that defines `generateGeometry(NX, NY)`, `generateHelperCurves(NSample)`, `generateWeightFunctions(XI, ETA)`, and `logicalSpace(NX, NY)`, then import it in `GenerateMesh.py`.

---

## Parameters

Key variables at the top of `GenerateMesh.py`:

| Variable | Description |
|----------|-------------|
| `CASE_PATH` | Path to the target OpenFOAM case directory |
| `MeshResolution` | `(Nx, Ny)` — number of points in ξ and η directions |
| `N_OUTER` | Outer iterations (recompute φ/ψ from the adapted grid) |
| `N_INNER` | Max inner TTM iterations per outer step |
| `NEUMANN_ITERATIONS` | How many early iterations apply the Neumann BC |
| `TOL` | Convergence tolerance (max point displacement) |
| `A_XI` / `A_ETA` | Weight amplification in ξ / η directions |
| `ISOTROPIC` | If `True`, use a scalar weight magnitude instead of separate directional weights |
| `OMEGA` | SOR over-relaxation factor (1.0 = no over-relaxation; typical 1.5–1.8) |
| `TTM_MODE` | `"XY"`, `"X"`, or `"Y"` (see above) |
| `Weight_Function_Cluster` | List of clustering specs. A and B are values in an exponentially decaying function. `{'side', 'A', 'B'}` |
| `NEUMANN_BOUNDARY` | Bool array `[bot, top, left, right]` for Neumann sides |

---

## Output Artifacts

After a successful run the following files are written inside `CASE_PATH`:

| Path | Content |
|------|---------|
| `<name>` (hex geometry file) | Point list + hex connectivity, consumed by `hexahedralMeshGenerator` |
| `constant/structuredMesh.Orig/originalPoints` | 2-D grid coordinates in row-major order |
| `constant/structuredMesh.Orig/originalToFoamPoint` | Index map from structured grid to OpenFOAM point numbering |
| `constant/structuredMesh.Orig/logicalI` / `logicalJ` | Logical grid indices (i, j) for each point |
| `system/topoSetDict` boundary files | Point sets for `topoSet` / `createPatch` |

---

## Module Structure

```
meshGeneration/
├── GenerateMesh.py              # entry point
├── Geometry/
│   ├── ParametrisizedCurves.py  # primitive curve definitions (C-shape, circle, etc.)
│   ├── clusterFunctins.py       # arc-length clustering utilities
│   └── Cases/                   # one file per domain geometry
├── TFI/
│   └── runTFI.py                # transfinite interpolation
├── TTM/
│   ├── TTMSolver.py             # main Gauss–Seidel iteration loop
│   ├── ThomasMiddlecoff.py      # weight functions and φ/ψ computation
│   ├── NeumannSolver.py         # ghost-cell Neumann BC
│   ├── BoundaryProjection.py    # KDTree curve snapping
│   ├── HelperFunctions.py       # metric coefficients, TDMA
│   └── stitchBoundaryDictionary.py  # periodic/stitched boundary handling
├── meshGeneration/
│   ├── CellAndPointGeneration.py    # extrude 2-D grid to 3-D hex cells
│   ├── CreateStructuredMesh.py      # write hex geometry and structured metadata
│   ├── OpenFOAMToStructuredMesh.py  # reverse-read an existing polyMesh
│   └── writeTopoSetPointFiles.py    # write topoSet input files
└── Plotting/                    # optional visualisation helpers
```
