#include "structuredPythonMotionSolver.H"

#include "addToRunTimeSelectionTable.H"
#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "pointFields.H"
#include "Time.H"
#include "mapPolyMesh.H"
#include "IOdictionary.H"
#include "sigFpe.H"

#include "include/dictionaryHelper.H"
#include "include/readStructuredMesh.H"
#include "pythonMotionBridge.H"

namespace Foam
{
    defineTypeNameAndDebug(structuredPythonMotionSolver, 0);

    addToRunTimeSelectionTable
    (
        motionSolver,
        structuredPythonMotionSolver,
        dictionary
    );
}


Foam::structuredPythonMotionSolver::structuredPythonMotionSolver
(
    const polyMesh& mesh,
    const IOdictionary& dict
)
:
    motionSolver(mesh, dict, typeName),
    pythonInterpreter_(new pybind11::scoped_interpreter{}),
    coeffDict_(dict.subDict(typeName + "Coeffs")),
    structuredDataLoaded_(false),
    refinementActive_(false),
    refinementField_("none"),
    timeCoefficient_(1.0),
    haveCachedStructuredMesh_(false),
    newPoints_(mesh.points())
{
    Foam::sigFpe::unset(false);

    std::string casePath = mesh.time().path().c_str();

    pybind11::module_::import("sys").attr("path").attr("insert")
    (
        0,
        casePath
    );

    pybind11::module_::import("RefineMesh");

    Foam::sigFpe::set(false);
}

void Foam::structuredPythonMotionSolver::solve()
{
    const fvMesh& fvm = refCast<const fvMesh>(mesh());

    // Start from current mesh points
    newPoints_ = mesh().points();

    bool refinement = false;
    scalar timeCoefficient = 1.0;

    readRefinementDict(fvm, refinement, timeCoefficient);

    if (!refinement)
    {
        return;  // nothing to do
    }

    const volScalarField& p = fvm.lookupObject<volScalarField>("p");
    const volVectorField& U = fvm.lookupObject<volVectorField>("U");

    vectorField delta =
        computePythonRefinementDisplacement(fvm, p, U, timeCoefficient);

    if (delta.size() != newPoints_.size())
    {
        FatalErrorInFunction
            << "Python displacement size mismatch. Got "
            << delta.size() << " entries for "
            << newPoints_.size() << " mesh points."
            << exit(FatalError);
    }

    forAll(newPoints_, pointI)
    {
        newPoints_[pointI] += delta[pointI];
    }

    Info<< "Applied Python mesh deformation with timeCoefficient = "
        << timeCoefficient << nl;
}

Foam::tmp<Foam::pointField>
Foam::structuredPythonMotionSolver::newPoints() const
{
    return curPoints();
}

Foam::tmp<Foam::pointField>
Foam::structuredPythonMotionSolver::curPoints() const
{
    // Return whatever solve() computed
    return tmp<pointField>(new pointField(newPoints_));
}

void Foam::structuredPythonMotionSolver::movePoints(const pointField&)
{
    // Usually empty unless you cache geometry-dependent data.
}


void Foam::structuredPythonMotionSolver::updateMesh(const mapPolyMesh&)
{
    // Topology changes not expected in your case.
}
