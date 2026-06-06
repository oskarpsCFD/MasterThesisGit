import matplotlib.pyplot as plt
import numpy as np


def plotAllWeights(XI, ETA, phi, psi, w1, w2):
    """
    2x2 figure over the logical domain:
      [0,0] w1   [0,1] w2
      [1,0] phi  [1,1] psi
    """
    xi_int  = XI[1:-1, 1:-1]
    eta_int = ETA[1:-1, 1:-1]

    rows = [
        [
            (w1,  r"$w_1$",  "viridis"),
            (w2,  r"$w_2$",  "viridis"),
        ],
        [
            (phi, r"$\Phi = \frac{1}{w_1}\frac{\partial w_1}{\partial \xi}$",   "RdBu_r"),
            (psi, r"$\Psi = \frac{1}{w_2}\frac{\partial w_2}{\partial \eta}$",  "RdBu_r"),
        ],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle("Weight functions over logical domain", fontsize=13)

    for r, row in enumerate(rows):
        for c, (field, title, cmap) in enumerate(row):
            ax = axes[r, c]
            coords = (XI, ETA) if field.shape == XI.shape else (xi_int, eta_int)
            if cmap == "RdBu_r":
                absmax = np.max(np.abs(field))
                vmin, vmax = -absmax, absmax
            else:
                vmin, vmax = field.min(), field.max()

            im = ax.pcolormesh(
                coords[0], coords[1], field,
                cmap=cmap, vmin=vmin, vmax=vmax,
                shading="auto",
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(r"$\xi$")
            ax.set_ylabel(r"$\eta$")

    plt.show()


def plotWeightFunctions(Xi, Eta, phi, psi):
    """
    Plot Phi and Psi over the computational domain (1x2, kept for backwards compat).
    """
    if Xi.shape != psi.shape:
        xi_int  = Xi[1:-1, 1:-1]
        eta_int = Eta[1:-1, 1:-1]
    else:
        xi_int  = Xi
        eta_int = Eta

    fields = [
        (phi, r"$\Phi = \frac{1}{w_1}\frac{\partial w_1}{\partial \xi}$"),
        (psi, r"$\Psi = \frac{1}{w_2}\frac{\partial w_2}{\partial \eta}$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle("Weight function fields over logical domain", fontsize=13)

    for ax, (field, title) in zip(axes, fields):
        absmax = np.max(np.abs(field))
        im = ax.pcolormesh(
            xi_int, eta_int, field,
            cmap="RdBu_r",
            vmin=-absmax, vmax=absmax,
            shading="auto",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")

    plt.show()
