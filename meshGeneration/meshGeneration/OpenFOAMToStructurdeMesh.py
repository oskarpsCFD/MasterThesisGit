import numpy as np
from typing import Tuple
from numpy.typing import NDArray

def pointsToGrid(P: NDArray, Nx: int, Ny: int):
    XY = P.reshape(Ny, Nx, 2)

    X = XY[:, :, 0]
    Y = XY[:, :, 1]

    return X, Y

def processVector(grad:NDArray):
    # gradP.shape = (N, 3)
    grad2d = grad[:, :2]
    return grad2d


def rewriteOpenFOAMPoints(P_OF:NDArray, P_Structured:NDArray, structureToFOAM,
                          Nx, Ny, field_OF: NDArray) -> Tuple[NDArray,NDArray,NDArray,NDArray]:

    # Now on the length of P_OF
    X, Y = pointsToGrid(P_Structured, Nx, Ny)

    if np.shape(P_OF) == np.shape(P_Structured):
        field = processVector(field_OF)

        w1 = field[:, 0]
        w2 = field[:, 1]
        w1 = w1.reshape(Ny, Nx)
        w2 = w2.reshape(Ny, Nx)

        return X, Y, w1[1:-1, 1:-1], w2[1:-1, 1:-1]

    structuredField = np.zeros((P_Structured.shape[0],2))
    for index, value in enumerate(structureToFOAM):
        structuredField[index] = field_OF[value]

    w1 = structuredField[:, 0].reshape(Ny, Nx)
    w2 = structuredField[:, 1].reshape(Ny, Nx)

    return X, Y, w1[1:-1, 1:-1], w2[1:-1, 1:-1]

