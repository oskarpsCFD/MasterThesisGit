#ifndef pythonFieldConversions_H
#define pythonFieldConversions_H

#include "volFields.H"
#include "pointField.H"
#include "pointFields.H"
#include "labelList.H"
#include "vectorField.H"

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace Foam
{

inline py::array_t<double> volScalarFieldToNumpy(const volScalarField& fld)
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


inline py::array_t<double> pointsToNumpy(const pointField& pts)
{
    const label n = pts.size();

    py::array_t<double> arr({n, 3});
    auto a = arr.mutable_unchecked<2>();

    for (label i = 0; i < n; ++i)
    {
        a(i, 0) = pts[i].x();
        a(i, 1) = pts[i].y();
        a(i, 2) = pts[i].z();
    }

    return arr;
}


inline py::array_t<double> pointVectorFieldToNumpy(const pointVectorField& fld)
{
    const vectorField& vf = fld.internalField();
    const label n = vf.size();

    py::array_t<double> arr({n, 3});
    auto a = arr.mutable_unchecked<2>();

    for (label i = 0; i < n; ++i)
    {
        a(i, 0) = vf[i].x();
        a(i, 1) = vf[i].y();
        a(i, 2) = vf[i].z();
    }

    return arr;
}


inline py::array_t<double> pointScalarFieldToNumpy(const pointScalarField& fld)
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


inline py::array_t<int> labelListToNumpy(const labelList& lst)
{
    const label n = lst.size();

    py::array_t<int> arr(n);
    auto a = arr.mutable_unchecked<1>();

    for (label i = 0; i < n; ++i)
    {
        a(i) = static_cast<int>(lst[i]);
    }

    return arr;
}

} // End namespace Foam

#endif
