/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2021 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    structuredPythonMotionSolver

Group
    motionSolver

Description
    Call the python script RefineMesh and provide it with the data from structuredMesh directory the mesh points and flow fields. 
    
    The python script will return a deformation vector to apply to the mesh.  
      
Author 
    X
    
    Developed in accordance with the authors master thesis.
    
    	Unstructured mesh generation is a solved problem 
    	structured mesh generation will never be complete
   
\*---------------------------------------------------------------------------*/

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
