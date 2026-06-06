#ifndef dictionaryHelper_H
#define dictionaryHelper_H

#include "fvMesh.H"
#include "IOdictionary.H"
#include "dictionary.H"
#include "word.H"
#include "scalar.H"
#include "IOobject.H"

#include "fvMesh.H"
#include "pointField.H"
#include "pointFields.H"
#include "labelList.H"
#include "error.H"

namespace Foam
{

inline void readRefinementDict
(
    const fvMesh& mesh,
    bool& refinement,
    scalar& timeCoefficient
)
{
    IOdictionary rRefinementDict
    (
        IOobject
        (
            "rRefinementDict",
            mesh.time().system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const dictionary& customRefinement =
        rRefinementDict.subDict("customRefinement");

    customRefinement.lookup("refinement") >> refinement;
    customRefinement.lookup("timeCoefficient") >> timeCoefficient;
}

} // End namespace Foam

#endif
