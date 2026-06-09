#ifndef pythonMotionHelper_H
#define pythonMotionHelper_H

#include "fvMesh.H"
#include "pointField.H"
#include "pointFields.H"
#include "volFields.H"
#include "vectorField.H"
#include "labelList.H"
#include "error.H"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "readStructuredMesh.H"
#include "refinementFieldHelper.H"
#include "pythonRefinement.H"

namespace py = pybind11;

namespace Foam
{

struct StructuredMeshData
{
    pointField structuredPoints;
    labelList foamToOriginal;
    labelList logicalI;
    labelList logicalJ;
};


inline StructuredMeshData readStructuredMeshData(const fvMesh& mesh)
{
    StructuredMeshData data;

    readStructuredMesh
    (
        mesh,
        data.structuredPoints,
        data.foamToOriginal,
        data.logicalI,
        data.logicalJ
    );

    return data;
}


inline py::array_t<double> getValidatedPythonDeformation
(
    const py::object& result,
    const fvMesh& mesh
)
{
    py::array_t<double> deformation = result.cast<py::array_t<double>>();
    py::buffer_info buf = deformation.request();

    if (buf.ndim != 2 || buf.shape[1] != 3)
    {
        FatalErrorInFunction
            << "Python refinement must return array of shape (nPoints, 3)"
            << exit(FatalError);
    }

    if (buf.shape[0] != mesh.nPoints())
    {
        FatalErrorInFunction
            << "Python returned deformation for " << buf.shape[0]
            << " points, but mesh has " << mesh.nPoints()
            << " points"
            << exit(FatalError);
    }

    return deformation;
}


inline vectorField callPythonForPointDisplacement
(
    const fvMesh& mesh,
    const StructuredMeshData& structuredMeshData,
    const volScalarField& p,
    const volVectorField& U,
    const scalar timeCoefficient
)
{
    const Time& runTime = mesh.time();

    tmp<pointVectorField> tUPoint;
    tmp<pointScalarField> tPPoint, tVortMagPoint;
    makeRefinementFields(mesh, runTime, p, U, tUPoint, tPPoint, tVortMagPoint);

    py::object result = callPythonRefinement
    (
        structuredMeshData.structuredPoints,
        mesh,
        structuredMeshData.foamToOriginal,
        tUPoint(),
        tPPoint(),
        tVortMagPoint(),
        structuredMeshData.logicalI,
        structuredMeshData.logicalJ
    );

    py::array_t<double> deformation =
        getValidatedPythonDeformation(result, mesh);

    py::buffer_info buf = deformation.request();
    const double* ptr = static_cast<const double*>(buf.ptr);

    vectorField delta(mesh.nPoints(), vector::zero);

    forAll(delta, pointI)
    {
        delta[pointI] =
            timeCoefficient
           *vector
            (
                ptr[3*pointI + 0],
                ptr[3*pointI + 1],
                ptr[3*pointI + 2]
            );
    }

    return delta;
}

} // End namespace Foam

#endif
