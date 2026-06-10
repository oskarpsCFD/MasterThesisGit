# OpenFOAMCode

Custom OpenFOAM C++ components. There are two independent targets:

| Directory | Type | Purpose |
|-----------|------|---------|
| `hexahedralMeshGenerator/` | Application | Convert a point list + hex connectivity to an OpenFOAM polyMesh |
| `motionSolver/` | Library | Dynamic motion solver that drives mesh adaptation from Python |

---

## Building

Source the OpenFOAM environment, then build each target with `wmake`:

```sh
. /usr/lib/openfoam/openfoam2312/etc/bashrc

cd hexahedralMeshGenerator && wmake
cd ../motionSolver          && wmake
```

Both targets must be compiled before running any simulation.

---

## hexahedralMeshGenerator

### What it does

Reads a custom ASCII geometry file that lists 3-D points and explicit 8-node hexahedral connectivity, then writes a valid `constant/polyMesh` directory using OpenFOAM's `polyMesh` class.

It is the bridge between the Python mesh generator (which outputs raw coordinates and cell indices) and OpenFOAM's mesh representation.

### Input file format

```
cells
<Ncells>
points
<Npoints>
(
  ( x y z )
  ...
)
connectivity
(
  ( i0 i1 i2 i3 i4 i5 i6 i7 )   // 8 vertex indices, 0-based
  ...
)
```

Shared face vertices must share the same index; do not duplicate coincident points with different indices.

### Usage

Run from the OpenFOAM case root after sourcing the environment:

```sh
hexahedralMeshGenerator <geometry-file>
```

All boundary faces land in a single patch called `defaultFaces`. Use `topoSet` + `createPatch` to split them into named boundary patches.

---

## motionSolver (structuredPythonMotionSolver)

### What it does

A custom `motionSolver` subclass that embeds a Python interpreter (pybind11) and calls `RefineMesh.py` at every timestep. It:

1. Reads `constant/structuredMesh/` — the structured grid metadata written by `meshGeneration` and kept up to date by the Python solver.
2. Extracts the current pressure (`p`) and velocity (`U`) fields from the running simulation.
3. Calls `RefineMesh.computeRefinementDisplacement(...)` in Python, passing point coordinates and flow data.
4. Receives a displacement vector (one entry per mesh point) and moves the mesh accordingly.

### Configuration

Add the following to `constant/dynamicMeshDict`:

```
dynamicFvMesh   dynamicMotionSolverFvMesh;

motionSolverLibs ("libstructuredPythonMotionSolver.so");

solver  structuredPythonMotionSolver;

structuredPythonMotionSolverCoeffs {}
```

Refinement is toggled at runtime through `system/rRefinementDict`:

```
customRefinement
{
    refinement      true;      // set to false to freeze the mesh
    fieldName       "p";
    timeCoefficient 1;
}
```

Set `refinement false` to run the solver in frozen-mesh mode without changing any code.

### Structured mesh metadata (`constant/structuredMesh/`)

The motion solver reads four OpenFOAM-format files from this directory at every timestep:

| File | Content |
|------|---------|
| `originalPoints` | 2-D grid coordinates (vectorField, z = 0) |
| `originalToFoamPoint` | Maps structured index → OpenFOAM point index (labelList) |
| `logicalI` / `logicalJ` | Logical grid column / row for each point (labelList) |

These files are updated by `RefineMesh.py` each time the mesh moves, so the motion solver always works from the current grid state.

---

## Source Files

```
hexahedralMeshGenerator/
└── hexahedralMeshGenerator.C   # full application

motionSolver/
├── structuredPythonMotionSolver.H / .C   # motionSolver subclass
├── pythonMotionBridge.H / .C             # calls Python, assembles displacement
├── include/
│   ├── readStructuredMesh.H              # reads constant/structuredMesh/
│   ├── dictionaryHelper.H               # reads rRefinementDict
│   ├── pythonInterface.H                # pybind11 helper wrappers
│   ├── pythonFieldConversions.H         # OpenFOAM ↔ Python array conversions
│   ├── pythonMotionHelper.H             # field extraction utilities
│   └── pythonRefinement.H              # top-level refinement call
└── pythonInterface                      # additional pybind11 glue
```
