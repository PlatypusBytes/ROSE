import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.axes as plt_axes
import numpy as np
from typing import Tuple, Union


def plot_2d_geometry(elements):
    fig = plt.figure()
    for element in elements:
        element_coords = [node.coordinates for node in element.nodes]
        x_coords = [coord[0] for coord in element_coords]
        y_coords = [coord[1] for coord in element_coords]
        plt.plot(x_coords, y_coords, color="blue")
        plt.plot(x_coords, y_coords, marker="o", markerfacecolor="blue", color="black")

    return fig


def create_animation(filename, x_data: Union[Tuple, np.array], y_data: Union[Tuple, np.array], format='html',fps=60, fig: plt.Figure = plt.figure(), **kwargs):
    """

    :param filename: name of the animation file
    :param x_data: tuple of multiple x_data np arrays or 1 np.array of x_data
    :param y_data: tuple of multiple y_data np arrays or 1 np.array of y_data
    :param format: video format
    :param fps: frames per second
    :param fig: existing figure with custom axis information or data
    :param kwargs: Line2d properties
    :return:
    """
    ims = []
    ax = fig.gca()
    ax.set_position([0.15, 0.15, 0.8, 0.8])

    # set x_data and y_data as tuple
    if not isinstance(x_data, Tuple):
        x_data = (x_data,)
    if not isinstance(y_data, Tuple):
        y_data = (y_data,)

    # check if y data of each dataset has the same size on axis nr 1
    assert all(item.shape[1] == y_data[0].shape[1] for item in y_data)
    # check if the same amount of datasets are used for x_data and y_data
    assert len(x_data) == len(y_data)

    # set colormap
    cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in np.linspace(0, 1, len(x_data))]

    # loop over data
    for i in range(len(y_data[0][0])):
        plts = []
        # loop over data sets
        for j in range(len(x_data)):
            plts.append(ax.plot(x_data[j], list(y_data[j][:, i]), color=colors[j], **kwargs)[0])

        ims.append(tuple(plts))

    plt.xlabel('distance [m]')
    plt.ylabel('displacement track [m]')
    plt.grid()

    # create animation
    writer = animation.writers[format](fps=fps)
    im_ani = animation.ArtistAnimation(fig, ims,
                                       blit=True)

    # save animation
    im_ani.save(filename, writer=writer)

    # close figure
    fig.clf()


