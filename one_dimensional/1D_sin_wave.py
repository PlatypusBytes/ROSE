import numpy as np
import matplotlib.pylab as plt
from scipy import sparse
from rose.model.solver import NewmarkSolver
import json
import os


def mass_matrix(H_discretisation, rho):

    discretisation = np.diff(H_discretisation)
    # mass matrix
    aux = np.zeros((2, 2))
    aux[0, 0] = 2
    aux[1, 0] = 1
    aux[0, 1] = 1
    aux[1, 1] = 2
    # mass = np.zeros((H_discretisation.shape[0], H_discretisation.shape[0]))
    mass = sparse.lil_matrix((H_discretisation.shape[0], H_discretisation.shape[0]))
    for i in range(mass.shape[0] - 1):
        mass_aux = aux * rho * discretisation[i] / 6
        mass[i : i + 2, i : i + 2] += mass_aux

    return mass


def stiff_matrix(H_discretisation, kappa):

    discretisation = np.diff(H_discretisation)
    # stiffness matrix
    aux = np.zeros((2, 2))
    aux[0, 0] = 1
    aux[1, 0] = -1
    aux[0, 1] = -1
    aux[1, 1] = 1
    # stiff = np.zeros((H_discretisation.shape[0], H_discretisation.shape[0]))
    stiff = sparse.lil_matrix((H_discretisation.shape[0], H_discretisation.shape[0]))
    for i in range(stiff.shape[0] - 1):
        stiff_aux = aux * kappa / discretisation[i]
        stiff[i : i + 2, i : i + 2] += stiff_aux

    return stiff


def apply_BC(M, C, K, F, H_discre, idx):

    # list of indexes
    indexes = list(range(len(H_discre)))

    # delete indexes that are BC
    del indexes[idx]

    # delete BC from global matrix
    M = M[indexes][:, indexes]
    C = C[indexes][:, indexes]
    K = K[indexes][:, indexes]
    F = F[indexes]

    return M, C, K, F


def main(omega=range(50, 100, 10), out_fold="./"):
    """
    Run 1D wave propagation

    :param omega: frequency range
    :param out_fold: output folder
    :return: None
    """
    # properties
    L = 1  # length of column
    rho = 2000  # density solid
    kappa = 20e6  # bulk modulus solid
    # poisson = 0.3  # poisson ratio
    discretisation = 0.01  # element size
    # settings
    settings_newmark = {"gamma": 0.5, "beta": 0.25}

    # solid discretisation
    H_discre = np.linspace(0, L, int(np.ceil(L / discretisation) + 1))
    # time discretisation
    tim = np.linspace(0, 0.05, 1000)

    # create matrices
    M = mass_matrix(H_discre, rho)
    K = stiff_matrix(H_discre, kappa)
    C = sparse.lil_matrix(K.shape)
    # Force vector
    F = sparse.lil_matrix((len(H_discre), len(tim)))

    # apply boundary conditions
    M, C, K, F = apply_BC(M, C, K, F, H_discre, -1)

    # maximum displacement list
    max_disp = []

    # for each frequency
    for w in omega:
        # add sinusoidal force to the surface (node 0)
        F[0, :] = np.sin(w * tim)
        # newmark solver
        res = NewmarkSolver()
        res.initialise(K.shape[0], tim)
        res.calculate(
            M.tocsc(),
            C.tocsc(),
            K.tocsc(),
            F.tocsc(),
            int(0),
            int(len(tim)-1),
        )
        # add the maximum displacement
        max_disp.append(np.max(res.u))

    # make plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_position([0.08, 0.11, 0.40, 0.8])
    ax[1].set_position([0.55, 0.11, 0.40, 0.8])

    ax[0].plot(omega, max_disp, color="k")
    ax[1].plot(omega, [1 / i for i in max_disp], color="k")  # check if it should be 1 or the sin force
    ax[0].set_xlabel(r"$\omega$")
    ax[1].set_xlabel(r"$\omega$")
    ax[0].set_ylabel("Displacement")
    ax[1].set_ylabel("Impedance")  # ToDo check this term
    ax[0].grid()
    ax[1].grid()
    plt.savefig(os.path.join(out_fold, "./fem.png"))
    plt.close()

    # save data and dump to dump json
    with open(os.path.join(out_fold, "fem.json"), "w") as f:
        json.dump({"omega": [i for i in omega], "displacement": max_disp}, f, indent=2)
    return


if __name__ == "__main__":
    main(omega=range(50, 500, 10))
