# Generative AI has assissted in creating detailed README scripts throughout the repository. The author fully understand the implications of using generative AI and the README 
# files have been thoroughly read to ensure quality in the text and descriptions. 

# Structured Mesh Generation and Adaptive Mesh Refinement for OpenFOAM

This repository implements a fully **structured hexahedral mesh generator** and a **physics-driven adaptive mesh refinement (AMR)** system that couples Python with OpenFOAM through a custom C++ motion solver and pybind11.

---

## Repository Layout

```
.
├── meshGeneration/          # Python: offline structured mesh generator
├── meshMovement/            # Python: online mesh refinement (called at runtime by OpenFOAM)
├── OpenFOAMCode/
│   ├── hexahedralMeshGenerator/   # C++: converts hex geometry file to OpenFOAM polyMesh
│   └── motionSolver/              # C++: pybind11 motion-solver bridge
└── Simulation/
    └── presentedAerofoilCase/     # Complete OpenFOAM case from the article
```

---

## Overall Workflow

### 1 — Generate the mesh (offline)

Run `meshGeneration/GenerateMesh.py` from the `meshGeneration/` directory.

This selects a geometry (e.g. `Geometry/Cases/SymmetricAerofoil.py`), creates an initial mesh with **Transfinite Interpolation (TFI)**, and "smooths" it with the **Thomas–Middlecoff (TTM)** elliptic solver. The script writes:

- `<casePath>/<name>` — hex geometry file consumed by `hexahedralMeshGenerator`
- `<casePath>/constant/structuredMesh.Orig/` — structured mesh metadata used by the motion solver at runtime

Edit the `CASE_PATH` and resolution variables at the top of `GenerateMesh.py` before running.

### 2 — Build the OpenFOAM polyMesh

From the OpenFOAM case root, run:

```sh
hexahedralMeshGenerator <name>     # writes constant/polyMesh
topoSet                            # applies boundary point sets
createPatch -overwrite             # creates named patches
```

### 3 — Prepare the simulation case

Copy the `meshMovement/` directory into the OpenFOAM case root so the motion solver can import `RefineMesh.py` at runtime:

```sh
cp -r meshMovement/ <casePath>/
```

Restore the initial field from the steady-state snapshot:

```sh
cp -r 0.steadyState/ 0
```

### 4 — Run the simulation

```sh
./runMeshMovement
```

The script resets all caches, starts `cycleRefinement.sh`, and runs `overRhoPimpleDyMFoam`. At every timestep the `structuredPythonMotionSolver` calls `RefineMesh.py`, which redistributes mesh points based on the current pressure and velocity fields and returns a displacement vector that is applied to the mesh.

---

## Dependencies

| Component | Requirement |
|-----------|-------------|
| Python    | ≥ 3.10 |
| NumPy     | any recent |
| SciPy     | any recent |
| OpenFOAM  | v2312 (openfoam.com) |
| pybind11  | ≥ 2.10 (motion solver only) |

---

## Building the C++ Components

Source the OpenFOAM environment and build with `wmake`:

```sh
. /usr/lib/openfoam/openfoam2312/etc/bashrc

cd OpenFOAMCode/hexahedralMeshGenerator && wmake
cd ../motionSolver                       && wmake
```

Both targets must be compiled before running any simulation.

---

## Presented Aerofoil Case

The case in `Simulation/presentedAerofoilCase/` is the exact setup reported in the article. Refer to its [README](Simulation/presentedAerofoilCase/README.md) for full setup and run instructions.
