#ifndef readStructuredMesh_H
#define readStructuredMesh_H

#include "fvMesh.H"
#include "pointField.H"
#include "labelList.H"
#include "pointIOField.H"
#include "labelIOList.H"
#include "IOobject.H"

namespace Foam
{

inline void readStructuredMesh
(
    const fvMesh& mesh,
    pointField& originalPoints,
    labelList& originalToFoamPoint,
    labelList& logicalI,
    labelList& logicalJ
)
{
    pointIOField originalPointsIO
    (
        IOobject
        (
            "originalPoints",
            mesh.time().constant()/"structuredMesh",
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    labelIOList originalToFoamPointIO
    (
        IOobject
        (
            "originalToFoamPoint",
            mesh.time().constant()/"structuredMesh",
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    labelIOList logicalIIO
    (
        IOobject
        (
            "logicalI",
            mesh.time().constant()/"structuredMesh",
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    labelIOList logicalJIO
    (
        IOobject
        (
            "logicalJ",
            mesh.time().constant()/"structuredMesh",
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    originalPoints      = originalPointsIO;
    originalToFoamPoint = originalToFoamPointIO;
    logicalI            = logicalIIO;
    logicalJ            = logicalJIO;
}

} // End namespace Foam

#endif
