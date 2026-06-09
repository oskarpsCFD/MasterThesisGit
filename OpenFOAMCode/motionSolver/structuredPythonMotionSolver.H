#ifndef structuredPythonMotionSolver_H
#define structuredPythonMotionSolver_H

#include "motionSolver.H"
#include "pointField.H"
#include "autoPtr.H"
#include "dictionary.H"
#include "word.H"
#include <pybind11/embed.h>
#include <memory>

 
namespace Foam
{

class structuredPythonMotionSolver
:
    public motionSolver
{
    // Private data

	
	std::unique_ptr<pybind11::scoped_interpreter> pythonInterpreter_;
        dictionary coeffDict_;

        mutable bool structuredDataLoaded_;
        mutable bool refinementActive_;
        mutable word refinementField_;
        mutable scalar timeCoefficient_;
        mutable pointField newPoints_;

        // Keep this as a generic holder if your helper defines the type
        // in a header. If StructuredMeshData is defined in readStructuredMesh.H,
        // include it there and use it directly.
        mutable bool haveCachedStructuredMesh_;

public:

    TypeName("structuredPythonMotionSolver");

    // Constructors
    structuredPythonMotionSolver
    (
        const polyMesh& mesh,
        const IOdictionary& dict
    );

    //- Destructor
    virtual ~structuredPythonMotionSolver() = default;

    // Member functions
    virtual tmp<pointField> curPoints() const;

    virtual tmp<pointField> newPoints() const;

    virtual void solve();

    virtual void movePoints(const pointField&);

    virtual void updateMesh(const mapPolyMesh&);
};

} // End namespace Foam

#endif
