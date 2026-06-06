#include "pythonMotionBridge.H"
#include "pythonMotionHelper.H"

namespace Foam
{

vectorField computePythonRefinementDisplacement
(
    const fvMesh& mesh,
    const volScalarField& p,
    const volVectorField& U,
    const scalar timeCoefficient
)
{
    StructuredMeshData structuredMeshData = readStructuredMeshData(mesh);

    return callPythonForPointDisplacement
    (
        mesh,
        structuredMeshData,
        p,
        U,
        timeCoefficient
    );
}

}
