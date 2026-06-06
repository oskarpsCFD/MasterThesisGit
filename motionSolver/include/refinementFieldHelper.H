#ifndef refinementFieldHelper_H
#define refinementFieldHelper_H

#include "fvCFD.H"
#include "volPointInterpolation.H"

namespace
{
    inline void reportScalarFieldStats(const word& name, const scalarField& sf)
    {
        if (sf.empty())
        {
            Info<< name << " is empty" << endl;
            return;
        }

        scalar minVal = sf[0];
        scalar maxVal = sf[0];
        label minI = 0;
        label maxI = 0;

        for (label i = 1; i < sf.size(); ++i)
        {
            if (sf[i] < minVal) { minVal = sf[i]; minI = i; }
            if (sf[i] > maxVal) { maxVal = sf[i]; maxI = i; }
        }

        Info<< name
            << " stats: min = " << minVal << " at index " << minI
            << ", max = " << maxVal << " at index " << maxI
            << endl;
    }


    inline void reportVectorFieldStats(const word& name, const vectorField& vf)
    {
        if (vf.empty())
        {
            Info<< name << " is empty" << endl;
            return;
        }

        scalar minMag = mag(vf[0]);
        scalar maxMag = mag(vf[0]);
        label minI = 0;
        label maxI = 0;

        for (label i = 1; i < vf.size(); ++i)
        {
            const scalar m = mag(vf[i]);
            if (m < minMag) { minMag = m; minI = i; }
            if (m > maxMag) { maxMag = m; maxI = i; }
        }

        Info<< name
            << " stats: min|.| = " << minMag << " at index " << minI
            << ", max|.| = " << maxMag << " at index " << maxI
            << endl;
    }


    inline tmp<pointScalarField> interpolateVolScalarToPoint
    (
        const fvMesh& mesh,
        const Time& runTime,
        const word& name,
        const volScalarField& volFld
    )
    {
        return tmp<pointScalarField>
        (
            new pointScalarField
            (
                IOobject
                (
                    name,
                    runTime.timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                volPointInterpolation::New(mesh).interpolate(volFld)
            )
        );
    }


    inline tmp<pointVectorField> interpolateVolVectorToPoint
    (
        const fvMesh& mesh,
        const Time& runTime,
        const word& name,
        const volVectorField& volFld
    )
    {
        return tmp<pointVectorField>
        (
            new pointVectorField
            (
                IOobject
                (
                    name,
                    runTime.timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                volPointInterpolation::New(mesh).interpolate(volFld)
            )
        );
    }
}


// Compute the 3 refinement fields sent to Python:
//   - U as a full vector field (nx3)
//   - p as a scalar field (n)
//   - vorticity magnitude |curl(U)| as a scalar field (n)
inline void makeRefinementFields
(
    const fvMesh& mesh,
    const Time& runTime,
    const volScalarField& p,
    const volVectorField& U,
    tmp<pointVectorField>& tUPoint,
    tmp<pointScalarField>& tPPoint,
    tmp<pointScalarField>& tVortMagPoint
)
{
    // U: full velocity vector interpolated to points
    tUPoint = interpolateVolVectorToPoint(mesh, runTime, "UPoint", U);
    reportVectorFieldStats("UPoint (point)", tUPoint().internalField());

    // p: raw pressure interpolated to points
    tPPoint = interpolateVolScalarToPoint(mesh, runTime, "pPoint", p);
    reportScalarFieldStats("pPoint (point)", tPPoint().internalField());

    // vorticity magnitude: |curl(U)| interpolated to points
    volScalarField vortMag
    (
        IOobject
        (
            "vortMag",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mag(fvc::curl(U))
    );
    reportScalarFieldStats("vortMag (cell)", vortMag.internalField());
    tVortMagPoint = interpolateVolScalarToPoint(mesh, runTime, "vortMagPoint", vortMag);
    reportScalarFieldStats("vortMagPoint (point)", tVortMagPoint().internalField());
}

#endif
