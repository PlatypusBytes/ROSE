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
import xlrd
import re
from typing import List, Dict

from SignalProcessing.signal_tools import Signal

import data_proc.SoS as SoS


def plot_data_colormesh(dates: np.ndarray, chainage: np.ndarray, data: np.ndarray, data_type: str, fig=None,position=111):
    """
    Plots data on a colormesh

    :param dates: nd array of datetime dates
    :param chainage: chainage values in km
    :param data: nd array of data with size [chainage, dates]
    :param data_type: type of data
    :param fig: current figure, default is None
    :param position: position of subplot in figure, default is 111
    :return:
    """
    # initialise figure if it is not an input
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(position)

    ax.pcolormesh(chainage, dates, data,
                        cmap='seismic', shading='auto')

    ax.set_title(data_type)
    ax.set_ylabel("Measurement date")
    ax.set_xlabel("Prorail chainage [km]")
    ax.invert_yaxis()
    ax.grid()

    return fig, ax

def read_rtg_sheet(sheet: xlrd.sheet.Sheet):
    """
    Reads one sheet of the rtg excel.

    :param sheet: one sheet of an rtg excel file
    :return:
    """

    # read prorail and rila chainage and convert to nd array
    prorail_chainage = np.array(sheet.col_values(0))[1:].astype(float)
    rila_chainage = np.array(sheet.col_values(1))[1:].astype(float)

    # reads dates and convert to nd array
    dates = np.array([datetime.strptime(date, '%Y-%m') for date in np.array(sheet.row_values(0))[2:]])

    # read all data values in current excel sheet
    n_cols = len(dates)
    all_data = []
    for col in range(2,n_cols+2):
        data = np.array(sheet.col_values(col))[1:].astype(float)
        all_data.append(data)
    all_data = np.array(all_data)

    return prorail_chainage, rila_chainage, dates, all_data


def read_rtg(rtg_fn,filetype):
    """
    Reads rtg data from excel file or pickle file

    :param rtg_fn: rtg filename
    :param filetype: type of the file: xls or pickle
    :return:
    """

    # reads rtg excel file and dumps data to pickle
    if filetype == "xls":
        wb = xlrd.open_workbook(rtg_fn)
        cant_sheet = wb.sheet_by_index(0)
        h1l_sheet = wb.sheet_by_index(1)
        h1r_sheet = wb.sheet_by_index(2)
        h2l_sheet = wb.sheet_by_index(3)
        h2r_sheet = wb.sheet_by_index(4)

        prorail_chainage, rila_chainage, dates, cant_data = read_rtg_sheet(cant_sheet)
        _, _, _, h1l_data = read_rtg_sheet(h1l_sheet)
        _, _, _, h1r_data = read_rtg_sheet(h1r_sheet)
        _, _, _, h2l_data = read_rtg_sheet(h2l_sheet)
        _, _, _, h2r_data = read_rtg_sheet(h2r_sheet)

        rtg_data = {"prorail_chainage": prorail_chainage,
                    "rila_chainage": rila_chainage,
                    "dates": dates,
                    "cant_data": cant_data,
                    "h1l_data": h1l_data,
                    "h1r_data": h1r_data,
                    "h2l_data": h2l_data,
                    "h2r_data": h2r_data}

        dump_path = (Path(rtg_fn).parents[0]/(Path(rtg_fn).stem + ".pickle"))
        with open(dump_path, "wb") as f:
            pickle.dump(rtg_data, f)

    #loads pickle data
    elif filetype == "pickle":
        with open(rtg_fn, "rb") as f:
            rtg_data = pickle.load(f)

    return rtg_data


def convert_prorail_chainage_to_RD(chainage_fn):
    with open(chainage_fn) as f:

        csv_reader = csv.reader(f, delimiter=';')

        # read header
        headers = next(csv_reader)

        # read all data and convert to float
        chainage_data = np.array([[float(value) for value in row] for row in csv_reader])


def plot_data_summary_on_sos(fugro_dict, sos_dict, dir):

    # loop over sos segments
    Path(dir).mkdir(parents=True, exist_ok=True)
    for name, segment in sos_dict.items():

        # if name == "Segment 1057":
            # initialise figure
            fig = plt.figure(figsize=(20,10))
            plt.tight_layout()

            # get coordinates of current segments
            coordinates = np.array(list(segment.values())[0]['coordinates'])

            # get coordinate limits
            xlim = [min(coordinates[:,0]), max(coordinates[:,0])]
            ylim = [min(coordinates[:,1]), max(coordinates[:,1])]

            # add plot of highlighted sos segments
            _, _ = SoS.ReadSosScenarios.plot_highlighted_sos(sos_dict, name, fig=fig, position=325)
            plt.grid()

            # add plot of settlement within the current segment measured by the fugro rila system
            _, _ = plot_settlement_in_range_vs_date(fugro_dict, xlim, ylim, fig=fig, position=321)
            plt.grid()

            all_d1, all_d2, all_d3, succeeded_dates, all_coords = [], [], [], [],[]
            for date, res_at_t in zip(fugro_dict["dates"], fugro_dict["data"]):

                # get only the data within limits at time t
                try:
                    coordinates_in_range, heights_in_range = get_data_within_bounds(xlim, ylim, res_at_t)
                    d1,d2,d3 = calculate_d_values(heights_in_range, coordinates_in_range)
                    all_d1.append(d1)
                    all_d2.append(d2)
                    all_d3.append(d3)
                    succeeded_dates.append(date)
                    all_coords.append(coordinates_in_range)
                except:
                    print(f"d1, d2, d3 calculation has failed for date {date}")

            _, _  = plot_d_values_vs_date(all_d1, succeeded_dates, "D1", fig=fig, position=322)
            plt.grid()
            _, _ = plot_d_values_vs_date(all_d2, succeeded_dates, "D2", fig=fig, position=324)
            plt.grid()
            _, _ = plot_d_values_vs_date(all_d3, succeeded_dates, "D3", fig=fig, position=326)
            plt.grid()

            fig.suptitle(name)
            fig.savefig(Path(dir, f"{name}"))


def __filter_without_boundary_effects(data, fs, Fpass_high, Fpass_low):
    """
    Filters signal while preventing boundary effects.
    Filters signal twice. Once from front to back, then from back to front. The final filtered signal is half the filtered
    reversed signal and half the filtered original signal

    :param data:
    :param fs:
    :param Fpass_high:
    :param Fpass_low:
    :return:
    """

    signal = Signal(np.zeros(len(data)),data,FS=fs)
    signal.filter(Fpass_high,4, type_filter="highpass")
    signal.filter(Fpass_low, 2, type_filter="lowpass")

    flipped_signal = Signal(np.zeros(len(data)),np.flip(data),FS=fs)
    flipped_signal.filter(Fpass_high,4, type_filter="highpass")
    flipped_signal.filter(Fpass_low, 2, type_filter="lowpass")

    filtered_sig = np.hstack([np.flip(flipped_signal.signal)[:int(len(signal.signal) / 2)],
                    signal.signal[int(len(signal.signal) / 2):]])

    return filtered_sig


def calculate_d_values(heights, coordinates):

    # calculate all distances between follow up measurements in array and return sorted array
    distances = np.linalg.norm(coordinates[1:, :] - coordinates[:-1, :], axis=1)

    # find indices where there is a transition in train track measurement
    transition_indices = np.where(distances > 1.5)[0]
    transition_indices = np.hstack([-1, transition_indices, len(heights)-1])

    # Remove last 10 measurements, such that the returned distances are likely to be on the same train track direction
    if len(distances) > 10:
        distances.sort()
        distances = distances[:-10]

    # calculate Fs
    fs = 1/np.mean(distances)

    # calculate the d1,d2 and d3 while preventing boundary effects
    d1 = np.array([])
    d2 = np.array([])
    d3 = np.array([])
    for i in range(len(transition_indices)-1):
        heights_zone = heights[transition_indices[i] + 1:transition_indices[i+1] + 1]

        d1_part = __filter_without_boundary_effects(heights_zone, fs, 1 / 25, 1/3)
        d1 = np.hstack([d1, d1_part])

        d2_part = __filter_without_boundary_effects(heights_zone, fs, 1 / 70, 1 / 2)
        d2 = np.hstack([d2, d2_part])

        d3_part = __filter_without_boundary_effects(heights_zone, fs, 1 / 150, 1 / 70)
        d3 = np.hstack([d3, d3_part])

    return d1, d2, d3

def plot_d_values_vs_date(d_values,dates, d_type, date_lim=None, fig=None,position = 111):
    m_to_mm = 1000

    # initialise figure if it is not an input
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(position)

    # calculate mean heights and std per date
    mean_d_values, std_d_values = [], []
    for d_ in d_values:
        mean_d_values.append(np.nanmean(d_))
        std_d_values.append(np.nanstd(d_))

    mean_d_values, std_d_values = np.array(mean_d_values), np.array(std_d_values)

    # plot settlement and mean settlement vs dates
    for date, d_ in zip(dates, d_values):
        date_array = np.empty(len(d_)).astype(datetime)
        date_array.fill(date)
        ax.plot(date_array, d_*m_to_mm, 'o', color='blue', markersize=0.5)
    # ax.plot(dates, d_values, 'o', color='blue', markersize=0.5)
    ax.plot(dates, mean_d_values*m_to_mm, 'o', color='orange')
    ax.plot(dates, mean_d_values*m_to_mm + std_d_values*m_to_mm, '_', color='red', markersize=10)
    ax.plot(dates, mean_d_values*m_to_mm - std_d_values*m_to_mm, '_', color='red', markersize=10)
    ax.set_xlabel("Date [y]")
    ax.set_ylabel(f"{d_type} [mm]")

    if date_lim is not None:
        ax.set_xlim(date_lim)

    return fig, ax


def plot_settlement_in_range_vs_date(res: Dict, xlim: List, ylim: List,date_lim=None, fig=None,position = 111):
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

        # calculate mean heights and std per date
        mean_sett = np.nanmean(settlement,axis=1)
        std_sett = np.nanstd(settlement,axis=1)

        # plot settlement and mean settlement vs dates
        ax.plot(dates, settlement, 'o', color='blue', markersize=0.5)
        ax.plot(dates, mean_sett, 'o', color='orange')
        ax.plot(dates, mean_sett + std_sett, '_', color='red', markersize=10)
        ax.plot(dates, mean_sett - std_sett, '_', color='red', markersize=10)
        ax.set_xlabel("Date [y]")
        ax.set_ylabel("Settlement [mm]")

        if date_lim is not None:
            ax.set_xlim(date_lim)

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
        coordinates_in_range, heights_in_range = get_data_within_bounds(xlim, ylim, res_at_t)
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

def plot_height_vs_coords_in_range(xlim, ylim, res, fig=None, projection="3d", position=111):
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
        ax = fig.add_subplot(position)

    # loop over each date
    for res_at_t in res["data"]:

        #get data within limits
        coordinates_in_range, heights_in_range = get_data_within_bounds(xlim, ylim, res_at_t)
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



def plot_date_vs_mileage2(xlim, ylim, res, fig=None):
    """
    # todo clean up
    :param xlim:
    :param ylim:
    :param res:
    :param fig:
    :return:
    """

    m_to_mm = 1e3

    # create figure if it does not exist
    if fig is None:
        fig = plt.figure()

    ax = plt.axes()

    # interpolate heights and coordinates on first measurement
    dates, coordinate_data, interpolated_heights = interpolate_coordinates(res, xlim, ylim)

    for i in range(len(coordinate_data)):

        discont_indices = np.where(np.abs(np.diff(coordinate_data[i][:, 0])) > 1.5)

        coords_1 = coordinate_data[0][:discont_indices[0][0] + 1, :]
        coords_2 = coordinate_data[0][discont_indices[0][0] + 1:, :]

        diff_1 = np.diff(coords_1, axis=0)
        diff_2 = np.diff(coords_2, axis=0)

        distances_1 = np.sqrt(np.sum(np.diff(coords_1,axis=0)**2,axis=1))
        distances_1 = np.cumsum(distances_1)
        distances_1 = np.append(0,distances_1)


        heights_1 = interpolated_heights[0][:discont_indices[0][0] + 1]
        heights_2 = interpolated_heights[0][discont_indices[0][0] + 1:]

        signal = Signal(1/distances_1,heights_1- np.mean(heights_1),FS =distances_1[1] )
        signal.filter(1 / 25, 10, type_filter="highpass")
        signal.filter(1 / 3, 10, type_filter="lowpass")

        signal.fft()
        signal.inv_fft()

        t = signal.time_inv
        u = signal.signal_inv

        plt.plot(t, u + np.mean(heights_1))
        plt.plot(distances_1, heights_1)
        plt.show()


def plot_date_vs_mileage(xlim, ylim, res, fig=None,position=111):
    """
    # todo clean up
    :param xlim:
    :param ylim:
    :param res:
    :param fig:
    :param position:
    :return:
    """

    m_to_mm = 1e3

    # create figure if it does not exist
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(position)

    # interpolate heights and coordinates on first measurement
    dates, coordinate_data, interpolated_heights = interpolate_coordinates(res, xlim, ylim)

    dates = [date.date() for date in dates]

    # get data within limits

    # im_array = np.array[]

    discont_indices = np.where(np.abs(np.diff(coordinate_data[0][:, 0])) > 1.5)

    coords_1 = coordinate_data[0][:discont_indices[0][0] + 1, :]
    coords_2 = coordinate_data[0][discont_indices[0][0] + 1:, :]

    # diff_1 = np.diff(coords_1)
    # diff_2 = np.diff(coords_2)
    #
    diff_1 = np.diff(coords_1, axis=0)
    diff_2 = np.diff(coords_2, axis=0)

    distances_1 = np.sqrt(np.sum(np.diff(coords_1, axis=0) ** 2, axis=1))
    distances_1 = np.cumsum(distances_1)
    distances_1 = np.append(0, distances_1)

    settlement = np.subtract(interpolated_heights[:,:discont_indices[0][0]], interpolated_heights[0, :discont_indices[0][0]]) * m_to_mm

    im = ax.imshow(settlement, aspect='auto', cmap='gray',extent=[0,distances_1[-1],0,13])
    ax.set_ylabel("Column length [m]", fontsize=12)
    ax.set_xlabel("Mileage [m]", fontsize=12)
    # cax = plt.axes([0.55, 0.1, 0.075, 0.8])
    cbar = plt.colorbar(im, fraction=0.1, pad=0.01)
    cbar.set_label("Displacement [mm]", fontsize=10)

    y_ticks = [i for i in range(len(dates))]


    ax.set_yticks(y_ticks)
    ax.set_yticklabels(np.flip(dates))



def get_data_within_bounds(xlim, ylim, res):
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


def merge_data(res: Dict):
    """
    Merges data of equal dates but different sections in one dataset

    :param res: fugro data dictionary
    :return:
    """

    # find all unique dates
    unique_dates = np.unique(res["dates"])

    # initialise new dictionary
    new_res = {'location': 'all',
               'dates': unique_dates,
               'data': []}

    for date in unique_dates:

        # merge coordinates and heights of data on duplicated dates
        duplicate_dates = res["dates"] == date
        merged_coordinates = np.vstack([item["coordinates"] for item in res["data"][duplicate_dates]])
        merged_heights = np.concatenate([item["heights"] for item in res["data"][duplicate_dates]])

        # add merged data to new dictionary
        new_res["data"].append({"coordinates": merged_coordinates,
                                "heights": merged_heights})

    return new_res


def get_data_at_location(file_dir, location: str ="all", filetype: str ='csv') -> Dict:
    """
    #todo add possibility to retrieve data from location, currently all data is retrieved

    :param file_dir: directory where all data files are located
    :param location: location of the data
    :param filetype: 'csv' or 'KRDZ'
    :return:
    """

    # # if location is all, get all files in dir which end with the desired extension
    # # else get all files based on the location
    # if location == "all":
    #     files = list(Path(file_dir).glob("*." + filetype))
    # else:
    #     files = list(Path(file_dir).glob(location + "*"))

    # initialise results dictionary
    res = {"location": location,
           "dates": [],
           "data": []}

    # add all files with desired extension from main directory
    # todo make date match more general
    for path, subdirs, files in os.walk(file_dir):
        for file in files:
            if filetype == "KRDZ" and file.endswith("KRDZ"):
                date_match = re.search(r'\d{4}-\d{2}-\d{2}', file)
                res["dates"].append(datetime.strptime(date_match.group(), "%Y-%m-%d"))
                res["data"].append(read_rila_data_from_krdz(Path(path,file)))
            if filetype == "csv" and file.endswith("csv"):
                date_match = re.search(r'\d{4}-\d{2}-\d{2}', file)
                res["dates"].append(datetime.strptime(date_match.group(), "%Y-%m-%d"))
                res["data"].append(read_rila_data_from_csv(Path(path,file)))

    # # add data from all desired files to dictionary
    # for file in files:
    #     date = datetime.strptime(file.stem.split('_')[-1],"%Y%m")
    #     res["dates"].append(date)
    #     if filetype == "KRDZ":
    #         res["data"].append(read_rila_data_from_krdz(file))
    #     if filetype == "csv":
    #         res["data"].append(read_rila_data_from_csv(file))

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
    loads processes Rila data

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

    # find unique coordinates and preserve original order
    coords = np.array(coords)
    _, unique_indices = np.unique(coords,axis=0,return_index=True)
    unique_indices = np.sort(unique_indices)
    coords = coords[unique_indices]

    # get heights at unique coordinates
    heights = np.array(heights)[unique_indices]

    res["coordinates"] = coords
    res["heights"] = heights
    return res


def interpolate_coordinates(res: Dict, xlim: List, ylim: List, search_radius: float = 1) -> (List, List, np.ndarray):
    """
    Interpolate all data of each dataset on the height data locations of the first measurement date. Note that for this
    algorithm, it is not required to have sorted coordinates. Simple interpolation does not work, as the fugro data is
    recorded on multiple tracks in the same dataset, where the measurement direction varies.

    For each date, first the distance to the [0,0] RD coordinate is calculated. Then the nearest distances compared to
    the first date are calculated. This step is performed to speed up the search. As the nearest distance can be located
    either side of the track, the following step is checking if the x and y coordinate are within the search radius.
    Lastly an inverse distance interpolation is performed between the coordinates at the first date and the other
    coordinates.

    :param res: Fugro results dictionary
    :param xlim: x limit
    :param ylim: y limit
    :param search_radius: radius where should be searched for close coordinates compared to the first data set
    :return:
    """

    # initialise lists
    dates = []
    coordinate_data = []
    interpolated_heights = []
    distances = []

    i = 0
    # loop over dates and data
    for date, res_at_t in zip(res["dates"], res["data"]):

        # get only the data within limits at time t
        coordinates_in_range, heights_in_range = get_data_within_bounds(xlim, ylim, res_at_t)

        # if coordinates are within limits, add dates, coordinates and height data to list
        if coordinates_in_range.size > 0:

            # calculate distance of each coordinate from  [0,0] RD coordinate
            distance = np.sqrt(coordinates_in_range[:, 0] ** 2 + coordinates_in_range[:, 1] ** 2)
            distance = distance[:, None]
            distances.append(distance)

            # if iteration is the first iteration, save initial position
            if i == 0:
                initial_distance = np.copy(distance)
                interpolated_height = heights_in_range
            else:

                # find nearest distances compared to the initial distance. Note that this step is done to limit the
                # search list of close coordinates and speed up the interpolation.
                tree = KDTree(distance)
                nearest_distances = tree.query(initial_distance, k=5, distance_upper_bound=search_radius)

                # Get the valid nearest distances
                valid = nearest_distances[1][:,:]<coordinates_in_range.shape[0]

                # initialise interpolated height
                interpolated_height = np.zeros(coordinate_data[0].shape[0])

                # loop over coordinates of the first data set
                for i in range(coordinate_data[0].shape[0]):

                    # get the indices of the nearest coordinates of compared to the coordinates of the first data set
                    indices = nearest_distances[1][i, valid[i,:]]

                    # calculate difference of x and y coordinates between the closest points in the current dataset
                    # and the first data set.
                    x_y_diff = coordinates_in_range[indices, :] - coordinate_data[0][i,:]

                    # check which coordinates are within the search radius
                    tmp = np.abs(x_y_diff) < search_radius

                    # coordinates are valid if both dx and dy are within the search radius
                    valid_found_coordinates = tmp[:, 0] & tmp[:, 1]

                    # get heights and distances of the closest valid points compared to the first data set
                    heights = heights_in_range[nearest_distances[1][i, valid[i, :]][valid_found_coordinates]]
                    distance_from_point = nearest_distances[0][i, valid[i, :]][valid_found_coordinates]

                    if len(distance_from_point) > 0:
                        # if the distance from the first data set is 0, set weights at 0.
                        if any(distance_from_point < 1e-10):
                            weights = (distance_from_point < 1e-10) * 1
                        else:
                            # else perform an inverse distance interpolation
                            weights = 1/distance_from_point
                            weights /= weights.sum()

                        interpolated_heights_at_t = heights.dot(weights)
                        interpolated_height[i] = interpolated_heights_at_t
                    else:
                        # if no valid coordinates, close to the first data set is found, set the interpolated height at
                        # NAN
                        interpolated_height[i] = np.NAN

            # append the dates, coordinate data, and the interpolated heights to lists
            dates.append(date)
            coordinate_data.append(coordinates_in_range)
            interpolated_heights.append(interpolated_height)

            i += 1

    # convert interpolated heights to a numpy array
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
    crs_bng = pyproj.Proj(init="epsg:28992")

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


def filter_data_at_point_coordinates(res, point_coordinates, search_radius):
    """
    Removes all rila coordinates and data in a range from a list of point coordinates

    :param res: rila results dictionary
    :param point_coordinates: point coordinates to be filtered out
    :param search_radius: radius around point coordinates which are to be filtered out

    :return:
    """

    for data in res["data"]:
        rila_coordinates = data["coordinates"]

        # initialise kd tree
        tree = KDTree(rila_coordinates)

        # initialise mask array
        mask = np.ones(len(rila_coordinates) ).astype(bool)

        # find all rila indices in range around point coordinates
        masked_indices = [j for i in tree.query_ball_point(point_coordinates, search_radius) for j in i]

        # set found indices at false
        mask[masked_indices] = False

        # remove coordinates and heights at found indices from results data
        data["coordinates"] = data["coordinates"][mask,:]
        data["heights"] = data["heights"][mask]

    return res


def filter_data_within_bounds(xbounds: np.ndarray, ybounds: np.ndarray, res: Dict):
    """
    Filters data within x bounds and y bounds

    :param xbounds: x limit of search area
    :param ybounds: y limit of search area
    :param res: RILA results dictionary
    :return:
    """

    for data in res["data"]:

        coordinates = data["coordinates"]

        # initialize mask array as zeros
        mask = np.zeros(coordinates.shape[0])

        # find all coordinates within each x and y limit
        for xlim, ylim in zip(xbounds, ybounds):
            mask += (coordinates[:, 0] >= xlim[0]).astype(int) * \
                    (coordinates[:, 0] <= xlim[1]).astype(int) * \
                    (coordinates[:, 1] >= ylim[0]).astype(int) * \
                    (coordinates[:, 1] <= ylim[1]).astype(int)

        # invert and convert mask array as boolean array
        mask = ~mask.astype(bool)

        # filter coordinates and heights
        data["coordinates"] = coordinates[mask,:]
        data["heights"] = data["heights"][mask]

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
    # dir = r"D:\software_development\rose\data\Fugro\AMS-to-EIN"
    # res = get_data_at_location(dir, location="all", filetype="KRDZ")
    # res = merge_data(res)
    # res = get_data_at_location(r"..\data\Fugro\Amsterdam-Eindhoven TKI Project", location="all")
    # save_fugro_data(res, r"..\data\Fugro\updated_rila_data.pickle")
    res = load_rila_data(r"..\data\Fugro\rila_data.pickle")


    calculate_d_values(res['data'][0]['heights'], res['data'][0]['coordinates'])
    # res = merge_data(res)
    # # point_coordinates = np.array([[122730.096, 487773.31], [138101.172, 453431.389],[0,0]])
    # # filter_data_at_point_coordinates(res, point_coordinates,1)
    #
    # # plot_date_vs_mileage(xlim, ylim, res, fig=None)
    #
    # import data_discontinuities as dd
    #
    # fn = r"D:\software_development\rose\data\data_discontinuities\wissel.json"
    #
    # all_coordinates = dd.get_coordinates_from_json(fn)
    # x_bounds, y_bounds = dd.get_bounds_of_lines(all_coordinates)
    #
    # res = filter_data_within_bounds(x_bounds, y_bounds, res)
    #
    # a=1+1

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

