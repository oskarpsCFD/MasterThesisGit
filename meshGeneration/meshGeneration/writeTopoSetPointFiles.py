import numpy as np
from numpy.typing import NDArray
import os


def returnBoundary(x:NDArray, y:NDArray, index: int):
    if index == 0:
        return np.column_stack((x[0, :], y[0,:])), "bot"

    if index == 1:
        return np.column_stack((x[-1, :], y[-1, :])), "top"

    if index == 2:
        return np.column_stack((x[:, 0], y[:, 0])), "left"

    if index == 3:
        return np.column_stack((x[:, -1], y[:, -1])), "right"
    else:
        raise RuntimeError(f"index {index} out of range")


def limitPoints(P: NDArray, stitchArray: NDArray | None = None, plot=False):
    if stitchArray is None or len(stitchArray) == 0:
        return P[:-1]
    i0, i1 = stitchArray[-1]
    P_keep = P[i0:i1]

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(P_keep[:, 0], P_keep[:, 1], "ko-")
        plt.axis("equal")
        plt.title("Limited boundary points")
        plt.show()


    return P_keep

def writeBoundaryPoints(X:NDArray, Y:NDArray, offset:float, path:str, stitchDict:dict | None = None):
    """
    frac = (bot, top, left, right)
    """
    system_path = os.path.join(path, "system")
    for index in range(4):
        boundary, key = returnBoundary(X, Y, index)
        if key in list(stitchDict.keys()):
            limitBoundary = limitPoints(boundary, stitchDict[key])
        else:
            limitBoundary = boundary


        fileName = os.path.join(system_path, f"{key}Points")
        with open(fileName, "w") as file:
            for x, y in limitBoundary:
                for z in (0.0, offset):
                    file.write(f"({x} {y} {z})\n")