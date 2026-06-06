import numpy as np
from numpy.typing import NDArray
from Plotting.LogicalGridPlotting import plot_logical_and_real

"""
Run Tranfinite interpolation on the parametric curves to ensure a good initial condition
"""

def interpolateCoordinate(i, j, Xi, Eta,
                          Cl, Cr, Ct, Cb):

    eta = Eta[j, i]
    xi = Xi[j, i]

    bot, top, left, right = Cb[i, :], Ct[i, :], Cl[j, :], Cr[j, :]

    botLeft = Cb[0, :]
    botRight = Cb[-1, :]
    topLeft = Ct[0, :]
    topRight = Ct[-1, :]

    part1 = (1-eta) * bot + eta * top + (1-xi) * left + xi * right
    part2 = xi * eta * topRight + xi * (1 - eta) * botRight + eta * (1 - xi) * topLeft + (1-xi) * (1 - eta) * botLeft

    return part1 - part2


def TFI(CB: NDArray, CL: NDArray, CR: NDArray, CT: NDArray,
        Xi: NDArray, Eta: NDArray, plot:bool=False):
    """
    :param Ci: The four sides described by parametric curves
    :param Xi: The logical x coordinate
    :param Eta: The logical y coordinate
    :return: X and Y which is the mapping in the real domain
    """
    if Xi.shape != Eta.shape:
        print("not matching coordinate shapes")
        return 0

    X = np.zeros_like(Xi)
    Y = np.zeros_like(Eta)

    Ny, Nx = Xi.shape

    for j in range(Ny):
        for i in range(Nx):
            if i == 0:
                X[j, i] = CL[j, 0]
                Y[j, i] = CL[j, 1]
                continue

            if i == Nx - 1:
                X[j, i] = CR[j, 0]
                Y[j, i] = CR[j, 1]
                continue

            if j == 0:
                X[j, i] = CB[i, 0]
                Y[j, i] = CB[i, 1]
                continue

            if j == Ny - 1:
                X[j, i] = CT[i, 0]
                Y[j, i] = CT[i, 1]
                continue

            point = interpolateCoordinate(i, j, Xi, Eta, CL, CR, CT, CB)
            X[j, i] = point[0]
            Y[j, i] = point[1]

    if plot:
        plot_logical_and_real(Xi, Eta, X, Y)
    return X,Y
