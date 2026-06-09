import numpy as np
from numpy.typing import NDArray

def locateBoundary(X:NDArray, Y:NDArray, location:str) -> NDArray:
    possible = ["bot", "top", "left", "right"]
    if not location in possible:
        raise ValueError(f"location {location} is not in {possible}")

    if location == "bot":
        return np.column_stack((X[0, :], Y[0,:]))

    elif location == "top":
        return np.column_stack((X[-1, :], Y[-1, :]))

    elif location == "left":
        return np.column_stack((X[:, 0], Y[:, 0]))

    elif location == "right":
        return np.column_stack((X[:, -1], Y[:, -1]))


def confirmStitchRegion(X, Y):
    "return what boundaries need to be studied for stitching"
    boundaries = ["bot", "top", "left", "right"]
    stitchArray = [False, False, False, False]

    for index, boundary in enumerate(boundaries):
        boundaryLine = locateBoundary(X, Y, boundary)
        if np.allclose(boundaryLine[0], boundaryLine[-1]):
            stitchArray[index] = True
    return stitchArray


def locateStitchedIndices(P: NDArray):
    stitchedMap = []
    for index in range(P.shape[0]):
        if np.allclose(P[index], P[-index-1], rtol=1e-3):
            stitchedMap.append([index, -index-1])

        else:
            break
    return np.asarray(stitchedMap)

def createStitchBoundaryDict(X, Y):
    boundary = np.asarray(["bot", "top", "left", "right"])
    stitchMapDictionary = {}
    stitchArray = confirmStitchRegion(X, Y)
    stitchArray = np.asarray(stitchArray)

    boundariesForStitching = boundary[stitchArray]
    for b in boundariesForStitching:
        P = locateBoundary(X, Y, b)
        stitched = locateStitchedIndices(P)
        stitchMapDictionary[b] = stitched

    return stitchMapDictionary