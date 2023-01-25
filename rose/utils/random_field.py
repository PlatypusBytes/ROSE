import numpy as np
import matplotlib.pylab as plt
import gstools as gs


def create_rf(mean: float, coefficient_variation: float, len_scale: float, angles: float, nodes: np.ndarray,
              seed: int=14):
    """
    Create a 1D random field with a given mean and coefficient of variation.

    Parameters
    ----------
    mean (float): mean of the random field
    coefficient_variation (float): coefficient of variation of the random field
    len_scale (float): length scale of the random field
    angles (float): angle of the random field
    nodes (np.ndarray): nodes of the random field
    seed (int): seed for the random field

    Returns
    -------
    np.ndarray: 1D random field
    """
    sigma = mean * coefficient_variation
    variance = sigma ** 2
    model = gs.Gaussian(dim=1, var=variance, len_scale=len_scale, angles=angles)
    rf = gs.SRF(model, mean=mean, seed=seed)
    return rf.structured(nodes)


if __name__ == '__main__':
    mean = 132
    cov = 0.25
    nodes = np.linspace(0, 10000, 10000) * 0.6
    rfield = create_rf(mean, cov, 1, 0, nodes)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(nodes, rfield, 'b-', label="RF")
    ax[0].plot(nodes, np.ones(len(nodes)) * mean, 'r-', label="mean")
    ax[0].set_ylim(bottom=0)
    ax[0].grid()
    ax[0].legend()

    ax[1].hist(np.random.normal(mean, mean * cov, 10000), density=True, bins=100, color="b")
    ax[1].hist(rfield, bins=100, density=True, color='r')
    ax[1].grid()
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlim(left=0)
    plt.show()
