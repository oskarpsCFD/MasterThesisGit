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
    hexahedralMeshGenerator

Group
    grpMeshConversionUtilities

Description
    Convert a point list + explicit hex connectivity to an OpenFOAM mesh.

    Input format:
      cells
      <Ncells>
      points
      <Npoints>
      (
        ( x y z )
        ...
      )
      connectivity
      (
        ( i0 i1 i2 i3 i4 i5 i6 i7 )   // Ncells entries, 8 vertex indices each
        ...
      )

    Notes:
      - Hex-only (8 indices per cell).
      - Indices reference the points list [0..Npoints-1].
      - Reuse indices for shared faces (do NOT duplicate shared points).
      
Author 
    Oskar P. Sagatun NTNU 2026
    
    Developed in accordance with the authors master thesis.
    
    	Unstructured mesh generation is a solved problem 
    	structured mesh generation will never be complete
   
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "polyMesh.H"
#include "IFstream.H"
#include "polyPatch.H"
#include "cellModel.H"
#include "pointField.H"

using namespace Foam;

static void expectChar(IFstream& is, char wanted)
{
    char ch;
    is >> ch;
    if (ch != wanted)
    {
        FatalErrorInFunction
            << "Expected '" << wanted << "' but got '" << ch << "'." << nl
            << exit(FatalError);
    }
}

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Read cells/points/connectivity, assume hexes, write polyMesh"
    );
    argList::noParallel();
    argList::addArgument("geometry file");

    #include "setRootCase.H"
    #include "createTime.H"

    const fileName geoFile = args.get<fileName>(1);
    IFstream is(geoFile);
    if (!is.good())
    {
        FatalErrorInFunction
            << "Cannot open file: " << geoFile << nl
            << exit(FatalError);
    }

    word tok;

    // --- cells ---
    is >> tok;
    if (tok != "cells")
    {
        FatalErrorInFunction
            << "Expected token 'cells', got '" << tok << "'." << nl
            << exit(FatalError);
    }
    label nCells = 0;
    is >> nCells;
    if (nCells <= 0)
    {
        FatalErrorInFunction
            << "Number of cells must be positive. Got " << nCells << nl
            << exit(FatalError);
    }

    // --- points ---
    is >> tok;
    if (tok != "points")
    {
        FatalErrorInFunction
            << "Expected token 'points' after cells, got '" << tok << "'." << nl
            << exit(FatalError);
    }
    label nPoints = 0;
    is >> nPoints;
    if (nPoints <= 0)
    {
        FatalErrorInFunction
            << "Number of points must be positive. Got " << nPoints << nl
            << exit(FatalError);
    }

    Info<< "Reading " << nCells << " hex cells from " << nPoints << " points." << nl;

    // Points list: '('  ( x y z ) ... ')'
    expectChar(is, '(');

    pointField points(nPoints);
    for (label i = 0; i < nPoints; ++i)
    {
        char ch;
        scalar x, y, z;
        is >> ch >> x >> y >> z >> ch;  // ( x y z )
        points[i] = point(x, y, z);
    }

    expectChar(is, ')');

    // --- connectivity ---
    is >> tok;
    if (tok != "connectivity")
    {
        FatalErrorInFunction
            << "Expected token 'connectivity' after points list, got '"
            << tok << "'." << nl
            << "Provide an explicit list of 8 indices per hex, e.g.:\n"
            << "connectivity\n(\n  (0 1 2 3 4 5 6 7)\n)\n"
            << exit(FatalError);
    }

    expectChar(is, '(');

    const cellModel& hexModel = cellModel::ref(cellModel::HEX);
    cellShapeList cells(nCells);

    for (label c = 0; c < nCells; ++c)
    {
        expectChar(is, '(');

        labelList hexPts(8);
        for (label k = 0; k < 8; ++k)
        {
            label idx;
            is >> idx;
            if (idx < 0 || idx >= nPoints)
            {
                FatalErrorInFunction
                    << "Cell " << c << ", vertex " << k
                    << ": point index " << idx
                    << " out of range [0," << (nPoints-1) << "]." << nl
                    << exit(FatalError);
            }
            hexPts[k] = idx;
        }

        expectChar(is, ')');
        cells[c].reset(hexModel, hexPts);
    }

    expectChar(is, ')');

    // --- build/write mesh ---
    polyMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,    // region name
            runTime.constant(),         // constant/polyMesh
            runTime
        ),
        std::move(points),
        cells,
        faceListList(),                 // no explicit boundary faces
        wordList(),                     // no patch names
        wordList(),                     // no patch types
        "defaultFaces",                 // exterior faces -> defaultFaces
        polyPatch::typeName,            // base patch type
        wordList()                      // no patch groups
    );

    // Write with decent precision
    IOstream::defaultPrecision(max(10u, IOstream::defaultPrecision()));

    Info<< "Writing mesh to constant/polyMesh ..." << nl;
    mesh.removeFiles();
    mesh.write();

    Info<< "Done. Points: " << mesh.nPoints()
        << ", Faces: "      << mesh.nFaces()
        << ", Cells: "      << mesh.nCells() << nl;

    Info<< "Reminder: reuse point indices for shared faces; "
           "do not duplicate identical coordinates with new indices." << nl;

    return 0;
}

// ************************************************************************* //
