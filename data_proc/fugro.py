import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import itertools

from pathlib import Path
from datetime import datetime


def plot_average_height_in_range_vs_date(xlim, ylim, res):
    """
    Plots the average track settlement within a segment for each time step
    :param xlim:
    :param ylim:
    :param res:
    :return:
    """
    m_to_mm = 1000

    fig, ax = plt.subplots()
    heights = []
    for res_at_t in res["data"]:
        coordinates_in_range, heights_in_range = filter_data_within_bounds(xlim, ylim, res_at_t)
        mean_height = np.mean(heights_in_range)
        heights.append(mean_height)

    dates = res["dates"][np.isfinite(heights)]
    heights = np.array(heights)[np.isfinite(heights)]

    if heights.size >0:
        settlement = (np.array(heights) - heights[0]) * m_to_mm

        ax.plot(dates, settlement)
        ax.set_xlabel("Date")
        ax.set_ylabel("Average settlement [mm]")

    return fig, ax

def plot_height_vs_coords_in_range(xlim, ylim, res, projection="3d"):
    """
    Plots height versus coordinates
    :param xlim:
    :param ylim:
    :param res:
    :param projection: Projection of the plot '3d', 'x', or 'y'
    :return:
    """

    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    fig = plt.figure()

    if projection == "3d":
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
    for res_at_t in res["data"]:
        coordinates_in_range, heights_in_range = filter_data_within_bounds(xlim, ylim, res_at_t)
        if coordinates_in_range.size and coordinates_in_range.size:
            # distance = np.append(0,np.cumsum((np.diff(coordinates_in_range[:,0])**2 + np.diff(coordinates_in_range[:,1])**2) **1/2))

            if projection == "3d":
                ax.plot3D(coordinates_in_range[:,0],coordinates_in_range[:,1], heights_in_range, marker=next(marker),markersize=2)

                ax.set_xlabel("x-coord")
                ax.set_ylabel("y-coord")
                ax.set_zlabel("height [m NAP]")

            elif projection == "x":
                ax.plot(coordinates_in_range[:,0], heights_in_range, marker=next(marker),markersize=2)

                ax.set_xlabel("x-coord")
                ax.set_ylabel("height [m NAP]")
            elif projection == "y":
                ax.plot(coordinates_in_range[:,1], heights_in_range, marker=next(marker),markersize=2)

                ax.set_xlabel("y-coord")
                ax.set_ylabel("height [m NAP]")

    return fig, ax


def filter_data_within_bounds(xlim, ylim, res):
    """

    :param xlim: x limit of search area
    :param ylim: y limit of search area
    :param res:
    :return:
    """
    coordinates = res["coordinates"]

    indices_in_range = np.where(np.logical_and(coordinates[:,0]>=xlim[0], coordinates[:,0]<=xlim[1]) & np.logical_and(coordinates[:,1]>=ylim[0], coordinates[:,1]<=ylim[1]))[0]
    coordinates_in_range = coordinates[indices_in_range,:]
    heights_in_range = res["heights"][indices_in_range]

    return coordinates_in_range, heights_in_range


# def get_data_at_location(file_dir, location="Amsterdam_Utrecht"):
#     files = list(Path(file_dir).glob(location + "*"))
#
#     res = {"location": location,
#            "dates": [],
#            "data": []}
#
#     for file in files:
#         date = datetime.strptime(file.stem.split('_')[-1],"%Y%m")
#         res["dates"].append(date)
#         res["data"].append(read_krdz_file(file))
#
#     res["dates"] = np.array(res["dates"])
#     res["data"] = np.array(res["data"])
#
#     sorted_indices = np.argsort(res["dates"])
#
#     res["dates"] = res["dates"][sorted_indices]
#     res["data"] = res["data"][sorted_indices]
#
#     return res

def get_data_at_location(file_dir, location="all"):

    if location == "all":
        files = list(Path(file_dir).glob("*.KRDZ"))
    else:
        files = list(Path(file_dir).glob(location + "*"))


    res = {"location": location,
           "dates": [],
           "data": []}

    for file in files:
        date = datetime.strptime(file.stem.split('_')[-1],"%Y%m")
        res["dates"].append(date)
        res["data"].append(read_krdz_file(file))

    res["dates"] = np.array(res["dates"])
    res["data"] = np.array(res["data"])

    sorted_indices = np.argsort(res["dates"])

    res["dates"] = res["dates"][sorted_indices]
    res["data"] = res["data"][sorted_indices]

    return res


def write_krdz_coordinates_to_csv(filename, coordinates):

    with open(Path(filename + '.csv'), 'w') as f:
        for coord in coordinates:
            f.write(f'{coord[0]};{coord[1]}\n')


def read_krdz_file(filename):
    """
    Reads krdz file, coordinates are arranged from north to south
    :param filename:
    :return:
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    res = {"coordinates": None,
           "height": None}

    coords = []
    heights = []

    for line in lines:
        splitted_line = line.split()
        coords.append([float(splitted_line[1]), float(splitted_line[2])])
        heights.append(float(splitted_line[5]))

    coords = np.array(coords)
    heights = np.array(heights)

    res["coordinates"] = coords
    res["heights"] = heights
    return res

if __name__ == '__main__':

    file_dir = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ"
    # res = get_data_at_location(file_dir, location="Amsterdam_Utrecht")
    res = get_data_at_location(file_dir, location="all")

    xlim = [138108.980, 138131.127]
    ylim = [453435.206, 453476.077]

    xlim = [128326, 128410]
    ylim = [467723, 468058]


    fig, ax = plot_height_vs_coords_in_range(xlim, ylim, res)

    fig.show()
    fig, ax = plot_average_height_in_range_vs_date(xlim, ylim, res)

    fig.show()

    pass

    # heights = []
    # for res_at_t in res["data"]:
    #     coordinates_in_range, heights_in_range = filter_data_within_bounds(xlim, ylim, res_at_t)
    #     mean_height = np.mean(heights_in_range)
    #     heights.append(mean_height)
    #
    # heights = (np.array(heights)  - heights[0]) * 1000
    #
    #
    # # height_matrix = np.array([res_at_t["heights"] for res_at_t in res["data"]])
    #
    # # plt.plot(heights )
    # plt.plot(res["dates"],heights )
    # plt.show()
    # pass

    # fn1 = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ\Amsterdam_Utrecht_201811.KRDZ"
    # fn2 = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ\DenBosch_Eindhoven_201811.KRDZ"
    # fn3 = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ\Utrecht_DenBosch_201811.KRDZ"
    #
    # res1 = read_krdz_file(fn1)
    # res2 = read_krdz_file(fn2)
    # res3 = read_krdz_file(fn3)
    #
    # write_krdz_coordinates_to_csv(Path(fn1).stem, res1["coordinates"])
    # write_krdz_coordinates_to_csv(Path(fn2).stem, res1["coordinates"])
    # write_krdz_coordinates_to_csv(Path(fn3).stem, res1["coordinates"])

