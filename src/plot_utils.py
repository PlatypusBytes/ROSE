import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def plot_2d_geometry(elements):
    fig = plt.figure()
    for element in elements:
        element_coords = [node.coordinates for node in element.nodes]
        x_coords = [coord[0] for coord in element_coords]
        y_coords = [coord[1] for coord in element_coords]
        plt.plot(x_coords, y_coords, color="blue")
        plt.plot(x_coords, y_coords, marker="o", markerfacecolor="blue", color="black")

    return fig

def create_animation(filename,x_data, y_data: np.array, format='html'):
    """
    Creates animation
    :param filename:
    :param x_data: 
    :param y_data:
    :return:
    """
    fig = plt.figure()

    ims = []
    ax = fig.gca()

    for i in range(len(y_data[0])):
        ims.append((ax.plot(x_data, list(y_data[:, i]), color='black')[0],))

    writer = animation.writers[format](fps=60)
    im_ani = animation.ArtistAnimation(fig, ims,
                                       blit=True)
    im_ani.save(filename, writer=writer)
