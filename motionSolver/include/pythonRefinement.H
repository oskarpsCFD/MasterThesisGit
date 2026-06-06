#ifndef pythonRefinement_H
#define pythonRefinement_H

#include "fvMesh.H"
#include "pointField.H"
#include "pointFields.H"
#include "labelList.H"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "pythonFieldConversions.H"

namespace py = pybind11;

namespace Foam
{

// Calls RefineMesh.RefineMesh with fields passed as a Python list:
// [UPoint (nx3), pPoint (n), vortMagPoint (n)]
inline py::object callPythonRefinement
(
    const pointField& structuredPoints,
    const fvMesh& mesh,
    const labelList& foamToOriginal,
    const pointVectorField& UPoint,
    const pointScalarField& pPoint,
    const pointScalarField& vortMagPoint,
    const labelList& logicalI,
    const labelList& logicalJ
)
{
    py::module_ post = py::module_::import("RefineMesh");

    py::list fieldList;
    fieldList.append(pointVectorFieldToNumpy(UPoint));
    fieldList.append(pointScalarFieldToNumpy(pPoint));
    fieldList.append(pointScalarFieldToNumpy(vortMagPoint));

    return post.attr("RefineMesh")
    (
        pointsToNumpy(structuredPoints),      // 1) structuredPoints
        pointsToNumpy(mesh.points()),         // 2) OpenFOAM mesh points
        labelListToNumpy(foamToOriginal),     // 3) foamToOriginal
        fieldList,                            // 4) [Umag, p, vortMag] scalar arrays
        labelListToNumpy(logicalI),           // 5) logicalI
        labelListToNumpy(logicalJ)            // 6) logicalJ
    );
}

} // End namespace Foam

#endif
