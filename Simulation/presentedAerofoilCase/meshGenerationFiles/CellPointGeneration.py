import numpy as np
from numpy.typing import NDArray


def generatePoints(X: NDArray, Y: NDArray) -> NDArray:
    return np.stack((X, Y), axis=-1).reshape(-1, 2)
