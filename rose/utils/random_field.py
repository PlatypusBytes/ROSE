import numpy as np
import matplotlib.pylab as plt
import gstools as gs


def create_rf(mean: float, coefficient_variation: float, len_scale: float, angles: float, nodes: np.ndarray,
              seed: int=14, log_normal: bool=False) -> np.ndarray:
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
    log_normal (bool): if True, the random field will be log-normal

    Returns
    -------
    np.ndarray: 1D random field
    """

    # Compute parameters for the normal / log-normal field
    if log_normal:
        mu_prime = np.log(mean) - 0.5 * np.log(1 + coefficient_variation**2)
        sigma_prime = np.sqrt(np.log(1 + coefficient_variation**2))
    else:
        mu_prime = mean
        sigma_prime = mean * coefficient_variation

    # Define the Gaussian model with the transformed variance
    model = gs.Gaussian(dim=1, var=sigma_prime**2, len_scale=len_scale, angles=angles)

    # Generate the random field
    rf = gs.SRF(model, mean=mu_prime, seed=seed)
    rf.structured(nodes)

    # If log-normal, exponentiate the field
    if log_normal:
        return np.exp(rf.field)
    else:
        return rf.field


if __name__ == '__main__':
    mean = 5
    cov = 0.3
    nodes = np.linspace(0, 10000, 10000) * 0.6
    rfield = create_rf(mean, cov, 1, 0, nodes)

    rfield_lognormal = create_rf(mean, cov, 1, 0, nodes, log_normal=True)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(nodes, rfield, 'b-', label="RF")
    ax[0, 0].plot(nodes, np.ones(len(nodes)) * mean, 'r-', label="mean")
    # ax[0, 0].set_ylim(bottom=0)
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].hist(np.random.normal(mean, mean * cov, 10000), density=True, bins=100, color="b")
    ax[1, 0].hist(rfield, bins=100, density=True, color='r')
    ax[1, 0].grid()
    # ax[1, 0].set_ylim(bottom=0)
    # ax, 0[1].set_xlim(left=0)

    ax[0, 1].plot(nodes, rfield_lognormal, 'b-', label="RF")
    ax[0, 1].plot(nodes, np.ones(len(nodes)) * mean, 'r-', label="mean")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].hist(np.random.lognormal(np.log(mean) - 0.5 * np.log(1 + cov**2), np.sqrt(np.log(1 + cov**2)), 10000), density=True, bins=100, color="b")
    ax[1, 1].hist(rfield_lognormal, bins=100, density=True, color='r')
    ax[1, 1].grid()

    plt.show()
