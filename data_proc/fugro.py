import numpy as np
from scipy.interpolate import interp2d, griddata
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pyproj

import itertools

import os
from pathlib import Path
from datetime import datetime
import csv
import pickle
from typing import List, Dict

def plot_settlement_in_range_vs_date(res: Dict, xlim: List, ylim: List, fig=None,position = 111):
    """
    Plots the track settlement within a segment for each time step

    :param res: results dictionary
    :param xlim: x limit
    :param ylim: y limit
    :param fig: optional existing figure
    :param position: position of subplot, default: 111

    :return:
    """
    m_to_mm = 1000

    # initialise figure if it is not an input
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(position)

    # interpolate heights and coordinates on first measurement
    dates, coordinate_data, interpolated_heights = interpolate_coordinates(res, xlim, ylim)

    if interpolated_heights.size > 0:
        # calculate settlement
        settlement = np.subtract(interpolated_heights, interpolated_heights[0,:]) * m_to_mm

        # calculate mean heights per date
        mean_heights = []
        for res_at_t in res["data"]:
            coordinates_in_range, heights_in_range = filter_data_within_bounds(xlim, ylim, res_at_t)
            mean_height = np.mean(heights_in_range)
            mean_heights.append(mean_height)

        mean_heights = np.array(mean_heights)[np.isfinite(mean_heights)]
        mean_settlement = (np.array(mean_heights) - mean_heights[0]) * m_to_mm

        # plot settlement and mean settlement vs dates
        ax.plot(dates, settlement, 'o', color='blue', markersize=0.5)
        ax.plot(dates, mean_settlement, 'o', color='orange')
        ax.set_xlabel("Date")
        ax.set_ylabel("Settlement [mm]")

    return fig, ax


def plot_average_height_in_range_vs_date(xlim, ylim, res, fig=None,position = 111):
    """
    Plots the average track settlement within a segment for each time step

    :param xlim: x limit
    :param ylim: y limit
    :param res: results dictionary
    :param fig: optional existing figure
    :param position: position of subplot, default: 111
    """
    m_to_mm = 1000

    # create figure if it does not exist
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(position)

    # get height data within limits and calculate mean
    heights = []
    for res_at_t in res["data"]:
        coordinates_in_range, heights_in_range = filter_data_within_bounds(xlim, ylim, res_at_t)
        mean_height = np.mean(heights_in_range)
        heights.append(mean_height)

    dates = res["dates"][np.isfinite(heights)]
    heights = np.array(heights)[np.isfinite(heights)]


    if heights.size >0:
        # calculate average settlement
        settlement = (np.array(heights) - heights[0]) * m_to_mm

        # plot average settlement
        ax.plot(dates, settlement)
        ax.set_xlabel("Date")
        ax.set_ylabel("Average settlement [mm]")

    return fig, ax

def plot_height_vs_coords_in_range(xlim, ylim, res, fig=None, projection="3d"):
    """
    Plots height versus coordinates

    :param xlim: x limit
    :param ylim: y limit
    :param res: results dictionary
    :param fig: optional existing figure
    :param projection: Projection of the plot '3d', 'x', or 'y'
    :return:
    """

    marker = itertools.cycle((',', '+', '.', 'o', '*'))

    # create figure if it does not exist
    if fig is None:
        fig = plt.figure()

    if projection == "3d":
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()

    # loop over each date
    for res_at_t in res["data"]:

        #get data within limits
        coordinates_in_range, heights_in_range = filter_data_within_bounds(xlim, ylim, res_at_t)
        if coordinates_in_range.size and coordinates_in_range.size:

            # plot data on a 3d projecten
            if projection == "3d":
                ax.plot3D(coordinates_in_range[:,0],coordinates_in_range[:,1], heights_in_range, marker=next(marker),markersize=2)

                ax.set_xlabel("x-coord")
                ax.set_ylabel("y-coord")
                ax.set_zlabel("height [m NAP]")

            # plot the data on x projection
            elif projection == "x":
                ax.plot(coordinates_in_range[:,0], heights_in_range, marker=next(marker),markersize=2)

                ax.set_xlabel("x-coord")
                ax.set_ylabel("height [m NAP]")

            # plot the data on y projection
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

def merge_data(res):

    unique_dates = np.unique(res["dates"])

    new_res = {'location': 'all',
               'dates': unique_dates,
               'data': []}

    for date in unique_dates:
        duplicate_dates = res["dates"] == date
        # np.vstack(res["data"][duplicate_dates]["coordinates"])
        merged_coordinates = np.vstack([item["coordinates"] for item in res["data"][duplicate_dates]])
        merged_heights = np.concatenate([item["heights"] for item in res["data"][duplicate_dates]])

        new_res["data"].append({"coordinates": merged_coordinates,
                                "heights": merged_heights})



    return new_res

    a=res


def get_data_at_location(file_dir, location: str ="all", filetype: str ='csv') -> Dict:
    """
    :param file_dir: directory where all data files are located
    :param location: location of the data
    :param filetype: 'csv' or 'KRDZ'
    :return:
    """

    # if location is all, get all files in dir which end with the desired extension
    # else get all files based on the location
    if location == "all":
        files = list(Path(file_dir).glob("*." + filetype))
    else:
        files = list(Path(file_dir).glob(location + "*"))

    # initialise results dictionary
    res = {"location": location,
           "dates": [],
           "data": []}

    # add data from all desired files to dictionary
    for file in files:
        date = datetime.strptime(file.stem.split('_')[-1],"%Y%m")
        res["dates"].append(date)
        if filetype == "KRDZ":
            res["data"].append(read_rila_data_from_krdz(file))
        if filetype == "csv":
            res["data"].append(read_rila_data_from_csv(file))

    # convert dates and data to nd arrays
    res["dates"] = np.array(res["dates"])
    res["data"] = np.array(res["data"])

    # sort data based on dates
    sorted_indices = np.argsort(res["dates"])
    res["dates"] = res["dates"][sorted_indices]
    res["data"] = res["data"][sorted_indices]

    return res

def save_fugro_data(data: dict, filename: str) -> None:
    """
    Save data dictionary as pickle file

    :param data: data dictionary
    :param filename: full filename and path to the output json file
    """

    # if path does not exits: creates
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # save as pickle
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_rila_data(filename: str) -> Dict:
    """
    loads processes rila data


    :param filename: input filename
    :return:
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)

    return data

def write_krdz_coordinates_to_csv(filename: str, coordinates: np.ndarray):
    """
    Writes the fugro data coordinates to a csv file
    :param filename: output filename
    :param coordinates: coordinates
    :return:
    """

    with open(Path(filename + '.csv'), 'w') as f:
        for coord in coordinates:
            f.write(f'{coord[0]};{coord[1]}\n')


def read_rila_data_from_krdz(filename) -> Dict:
    """
    Reads rila data from krdz file
    :param filename: krdz file name
    :return:
    """

    # read the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # initialise dictionary
    res = {"coordinates": None,
           "height": None}

    # get coordinates and heights from data
    coords = []
    heights = []
    for line in lines:
        splitted_line = line.split()
        coords.append([float(splitted_line[1]), float(splitted_line[2])])
        heights.append(float(splitted_line[5]))

    # convert coordinates and heights to np array
    coords = np.array(coords)
    heights = np.array(heights)

    res["coordinates"] = coords
    res["heights"] = heights
    return res


def interpolate_coordinates(res: Dict, xlim: List, ylim: List) -> (List, List, np.ndarray):
    """
    interpolate all data of each data on the height data locations of the first measurement date

    :param res: fugro results dictionary
    :param xlim: x limit
    :param ylim: y limit
    :return:
    """

    search_radius = 1 #m

    # initialise lists
    dates = []
    coordinate_data = []
    height_data = []
    interpolated_heights = []
    distances = []


    i = 0
    # loop over dates and data
    for date, res_at_t in zip(res["dates"],res["data"]):

        # get only the data within limits at time t
        coordinates_in_range, heights_in_range = filter_data_within_bounds(xlim, ylim, res_at_t)

        # if coordinates are within limits, add dates, coordinates and height data to list
        if coordinates_in_range.size > 0:

            distance = np.sqrt(coordinates_in_range[:, 0] ** 2 + coordinates_in_range[:, 1] ** 2)
            distance = distance[:,None]
            distances.append(distance)

            # if iteration is the first iteration, save initial position
            if i == 0:
                initial_distance = np.copy(distance)
                interpolated_height = heights_in_range
            else:

                # find nearest distances compared to the initial distance
                tree = KDTree(distance)
                nearest_distances = tree.query(initial_distance,k=5,distance_upper_bound=search_radius)

                # tmp[0]
                valid = nearest_distances[1][:,:]<coordinates_in_range.shape[0]

                interpolated_height = np.zeros(coordinate_data[0].shape[0])
                for i in range(coordinate_data[0].shape[0]):
                    indices = nearest_distances[1][i,valid[i,:]]
                    # coordinates_in_range[i,:]
                    tmp = coordinates_in_range[indices,:] - coordinate_data[0][i,:]
                    tmp2 = np.abs(tmp) < search_radius
                    valid_found_coordinates = tmp2[:, 0] & tmp2[:, 1]

                    heights = heights_in_range[nearest_distances[1][i, valid[i, :]][valid_found_coordinates]]

                    distance_from_point = nearest_distances[0][i, valid[i, :]][valid_found_coordinates]

                    if len(distance_from_point) > 0:
                        if any(distance_from_point < 1e-10):
                            weights = (distance_from_point < 1e-10) * 1
                        else:
                            # inverse distance interpolation
                            weights = 1/distance_from_point
                            weights /= weights.sum()
                        interpolated_heights_at_t = heights.dot(weights)
                        interpolated_height[i] = interpolated_heights_at_t
                    else:
                        interpolated_height[i] = np.NAN

            dates.append(date)
            coordinate_data.append(coordinates_in_range)
            height_data.append(heights_in_range)
            interpolated_heights.append(interpolated_height)

            i += 1

    interpolated_heights = np.array(interpolated_heights)

    return dates, coordinate_data, interpolated_heights


def read_trajectory_qc_data(filename):
    """
    Reads data from trajectory qc file
    :param filename:
    :return:
    """
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = []
        for row in csv_reader:
            rows.append(row)
        return rows

def get_lat_long_from_qc_data(qc_data):
    """
    Gets latitude and longitude coords from qc data
    :param qc_data:
    :return:
    """

    lat = []
    long = []
    for i, row in enumerate(qc_data):
        if i>0:
            lat.append(row[3])
            long.append(row[4])

    return np.array(lat), np.array(long)


def convert_lat_long_to_rd(lat, long):
    """
    Converts latitude and longitude coordinates to rd coordinates
    :param lat:
    :param long:
    :return:
    """

    crs_wgs = pyproj.Proj(init='epsg:4326')
    crs_bng  = pyproj.Proj(init="epsg:28992")

    x, y = pyproj.transform(crs_wgs, crs_bng, long, lat)

    return x,y

def read_rila_data_from_csv(filename) -> Dict:
    """
    Reads rila data from csv file
    :param filename: csv file name
    :return:
    """

    # read all lines from csv file
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        lines = []
        for line in csv_reader:
            lines.append(line)

    # initialise results dictionary
    res = {"coordinates": None,
           "height": None}

    coords = []
    heights = []
    # adds coordinates and heights to list
    for i, line in enumerate(lines):
        if i>0:
            coords.append([float(line[0]), float(line[1])])
            heights.append(float(line[2]))

    # convert coordinates and heights to nd arrays
    coords = np.array(coords)
    heights = np.array(heights)

    res["coordinates"] = coords
    res["heights"] = heights
    return res



if __name__ == '__main__':

    # filename = r"D:\software_development\ROSE\gis_map\qc results\trajectory_qc_data.csv"
    #
    # qc_data = read_trajectory_qc_data(filename)
    # lat, long = get_lat_long_from_qc_data(qc_data)
    #
    # x_coord, ycoords = convert_lat_long_to_rd(lat, long)

    # filename = r"D:\software_development\ROSE\data\Fugro\Amsterdam-Eindhoven TKI Project\01_Amsterdam_Utrecht\Amsterdam_Utrecht_201811.csv"

    # read_rila_data_from_krdz(filename)
    # res = read_rila_data_from_csv(filename)
    # res = get_data_at_location(r"..\data\Fugro\Amsterdam-Eindhoven TKI Project", location="all")
    # save_fugro_data(res, r"..\data\Fugro\rila_data.pickle")
    res = load_rila_data(r"..\data\Fugro\rila_data.pickle")

    merge_data(res)
    a=1+1

    # file_dir = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ"
    # # res = get_data_at_location(file_dir, location="Amsterdam_Utrecht")
    # res = get_data_at_location(file_dir, location="all")
    #
    # xlim = [128326, 128410]
    # ylim = [467723, 468058]
    #
    # dates, coordinate_data, interpolated_heights = interpolate_coordinates(res, xlim, ylim)
    #
    # settlement = np.subtract(interpolated_heights, interpolated_heights[0,:])
    #
    # plt.plot(dates,settlement, 'o',color='black')
    # plt.show()

    #
    # xlim = [138108.980, 138131.127]
    # ylim = [453435.206, 453476.077]
    #
    # xlim = [128326, 128410]
    # ylim = [467723, 468058]
    #
    #
    # fig, ax = plot_height_vs_coords_in_range(xlim, ylim, res)
    #
    # fig.show()
    # fig, ax = plot_average_height_in_range_vs_date(xlim, ylim, res)
    #
    # fig.show()
    #
    # pass

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

