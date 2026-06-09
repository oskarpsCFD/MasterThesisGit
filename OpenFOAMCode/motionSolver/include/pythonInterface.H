#ifndef pythonInterface_H
#define pythonInterface_H

#include <pybind11/embed.h>
#include "pythonFieldConversions.H"

namespace py = pybind11;

py::object callPythonPostProcessing
(
    py::module_& post,
    const volScalarField& p,
    const fvMesh& mesh
)
{
    return post.attr("process")
    (
        volScalarFieldToNumpy(p),
        pointsToNumpy(mesh.points())
    );
}

py::array_t<double> pointScalarFieldToNumpy(const pointScalarField& fld)
{
    const scalarField& sf = fld.internalField();
    const label n = sf.size();

    py::array_t<double> arr(n);
    auto a = arr.mutable_unchecked<1>();

    for (label i = 0; i < n; ++i)
    {
        a(i) = sf[i];
    }

    return arr;
}


#endif
