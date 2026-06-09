#ifndef pythonMotionBridge_H
#define pythonMotionBridge_H

#include "fvMesh.H"
#include "pointField.H"
#include "volFields.H"
#include "vectorField.H"

namespace Foam
{

// Compute displacement for all mesh points by sending Umag, p, and vorticity
// magnitude to Python as a list of 3 scalar arrays.
vectorField computePythonRefinementDisplacement
(
    const fvMesh& mesh,
    const volScalarField& p,
    const volVectorField& U,
    const scalar timeCoefficient
);

} // End namespace Foam

#endif
