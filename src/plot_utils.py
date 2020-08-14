import matplotlib.pyplot as plt


def plot_2d_geometry(elements):
    fig = plt.figure()
    for element in elements:
        element_coords = [node.coordinates for node in element.nodes]
        x_coords = [coord[0] for coord in element_coords]
        y_coords = [coord[1] for coord in element_coords]
        plt.plot(x_coords, y_coords, color="blue")
        plt.plot(x_coords, y_coords, marker="o", markerfacecolor="blue", color="black")

    return fig
    # plt.show()
