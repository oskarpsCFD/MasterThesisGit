import numpy as np
from numpy.typing import NDArray


"""
All cluster functions to be used to create special clusterings
"""

def f(s: NDArray, require_odd: bool = False) -> NDArray:
    s = np.asarray(s, dtype=float)
    s = np.sort(s)

    if s.min() < 0.0 or s.max() > 1.0:
        raise ValueError("s must be in [0,1]")

    if require_odd and (len(s) % 2 == 0):
        raise ValueError("len(s) must be odd")
    return s


def fAirfoilCluster(s: NDArray, p: float = 1.0) -> NDArray:
    """
    Your original midpoint-split clustering, but:
    - enforces [0,1] properly
    - enforces odd length (required by this specific construction)
    - raises errors instead of returning 0
    """
    s = np.asarray(s, dtype=float)

    if s.min() < 0.0 or s.max() > 1.0:
        raise ValueError("s must be in [0,1]")

    if (len(s) % 2) == 0:
        raise ValueError("len(s) must be odd for this clustering rule")

    s = np.sort(s)

    splitInd = len(s) // 2  # exact middle index for odd length
    sNew = np.zeros_like(s)

    # left half clusters toward 0
    sNew[:splitInd] = 1.0 - (2.0 * s[:splitInd])**p
    # right half clusters toward 1
    sNew[splitInd:] = 1.0 - (2.0 * (1.0 - s[splitInd:]))**p

    return sNew


def fRadian(s: NDArray) -> NDArray:
    if s[0] < 0 or s[-1] > 1:
        print("s Should be between 0 and 1")
        return 0

    if len(s) % 2 == 0:
        print("s Should be odd number")
        return 0

    s = np.sort(s)

    return s * 2*np.pi