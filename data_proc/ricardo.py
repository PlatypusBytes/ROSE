import os
import pickle
import json

from typing import List, Dict

from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

import smooth
from rose.utils import signal_proc


settings_filter = {"FS": 250,
                   "cut-off": 30,
                   "n": 10}


def convert_lat_long_to_rd(lat, long):
    """
    Converts latitude and longitude coordinates to rd coordinates
    :param lat:
    :param long:
    :return:
    """

    transformer = Transformer.from_crs("epsg:4326", "epsg:28992")
    x, y = transformer.transform(lat, long)

    return x, y

def load_inframon_data(filename: str) -> Dict:
    """
    loads processes inframon data


    :param filename: input filename
    :return:
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)

    return data


def plot_train_velocity(data: Dict, fig=None, position=111):
    """
    Plots train velocity versus time

    :param data: Ricardo data dictionary
    :param fig: optional existing figure
    :param position: position in subplot
    :return:
    """

    # initialises figure if it does not exists
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(position)

    # plots time series
    if data["time"].size > 0:
        ax.plot(data["time"], data["speed"])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Train velocity [km/u]")

    return fig, ax

def plot_velocity_signal(time, acceleration, fig=None, position=111):
    """
    Integrates acceleration to velocity and plots a time series of the axle velocity time series

    :param time: time array
    :param acceleration:  acceleration array
    :param fig: optional existing figure
    :param position: position in subplot
    :return:
    """

    # integrations acceleration signal to velocity signal
    velocity = signal_proc.int_sig(acceleration, time, hp=True,
                              mov=False, baseline=False, ini_cond=0)

    # initialises figure if it does not exists
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(position)

    # plots time series
    if time.size > 0:
        ax.plot(time, velocity)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Axle velocity [$\mathregular{m/s}$]")

    return fig, ax

def plot_acceleration_signal(time, acceleration, fig=None, position=111):
    """
    Plots a time series of the axle acceleration time series

    :param time: time array
    :param acceleration:  acceleration array
    :param fig: optional existing figure
    :param position: position in subplot
    :return:
    """

    # initialises figure if it does not exists
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(position)

    # plots time series
    if time.size > 0:
        ax.plot(time, acceleration)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Axle acceleration [$\mathregular{m/s^{2}}$]")

    return fig, ax


def plot_fft_acceleration_signal(data, acceleration, smoothing_distance,fig=None,position=111):

    time = data["time"]

    m_to_mm = 1e3
    freq, ampl = signal_proc.fft_sig(np.array(acceleration), 250)
    ampl = smooth_signal_within_bounds_over_wave_length(data, smoothing_distance, ampl)

    # initialises figure if it does not exists
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(position)

    # plots time series
    if time.size > 0:
        ax.plot(freq, ampl * m_to_mm)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Axle acceleration amplitude [$\mathregular{mm/s^{2}/Hz}$]")

    return fig, ax


def plot_fft_velocity_signal(data, acceleration, smoothing_distance, fig=None, position=111):
    m_to_mm = 1e3

    time = data["time"]

    # integrations acceleration signal to velocity signal
    velocity = signal_proc.int_sig(acceleration, time, hp=True,
                              mov=False, baseline=False, ini_cond=0)

    freq, ampl = signal_proc.fft_sig(np.array(velocity), 250)

    ampl = smooth_signal_within_bounds_over_wave_length(data, smoothing_distance, ampl)

    # initialises figure if it does not exists
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(position)

    # plots time series
    if time.size > 0:
        ax.plot(freq, ampl * m_to_mm)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Axle velocity amplitude [$\mathregular{mm/s/Hz}$]")

    return fig, ax

def get_data_within_bounds(data: Dict, xlim: List, ylim: List) -> Dict:
    """
    Gets the data from a ricardo dataset within coordinate boundaries

    :param data: dataset at date
    :param xlim: x limit of the coordinates
    :param ylim: y limit of the coordinates
    :return:
    """

    # find data within coordinate limits
    mask = (data["coordinates"][:, 0] >= xlim[0]) & (data["coordinates"][:, 0] < xlim[1]) & (
                data["coordinates"][:, 1] >= ylim[0]) & (data["coordinates"][:, 1] < ylim[1])

    # create new dict with bounded data
    bounded_data = {"time": data["time"][mask],
                    "coordinates": data["coordinates"][mask,:],
                    "speed": data["speed"][mask],
                    "acc_side_1": data["acc_side_1"][mask],
                    "acc_side_2": data["acc_side_2"][mask],
                    "segment": data["segment"][mask]}

    return bounded_data

def smooth_signal_within_bounds_over_wave_length(data: Dict, wavelength: float, signal: np.ndarray):
    """
    This smooths a time series within bounds. Note that an average velocity is used for the smoothing. Therefore, this
    function is not reliable during acceleration or deceleration.

    :param data: ricardo data within bounds
    :param wavelength: wavelength over which should be smoothed
    :param signal: signal to be smoothed
    :return:
    """
    velocity = np.nanmean(data["speed"]) / 3.6  # velocity in [m/s]
    seconds = wavelength/velocity
    dt = np.mean(np.diff(data["time"]))
    n_points = int(seconds/dt)

    smoothed_signal = smooth.smooth(signal, n_points)
    return smoothed_signal


def read_inframon(file_names: List, output_f: str):

    results = {}
    for fi in file_names:
        with open(fi, "r") as f:
            data = json.load(f)

        # initialise dict
        name = os.path.splitext(os.path.split(fi)[1])[0]
        results.update({name: {"time": [],
                               "coordinates": [],
                               "speed": [],
                               "acc_side_1": [],
                               "acc_side_2": [],
                               "segment": [],
                               }})

        # get lat and lon coordinates
        lat = np.array([data[i]['lat'] for i in range(len(data))]).astype(float)
        lon = np.array([data[i]['lon'] for i in range(len(data))]).astype(float)

        # find invalid indices
        nan_indices_lat = np.argwhere(np.isnan(lat))
        nan_indices_lon = np.argwhere(np.isnan(lon))
        nan_indices = np.unique(np.append(nan_indices_lat,nan_indices_lon))

        # mask invalid indices
        mask = np.ones(lat.shape, bool)
        mask[nan_indices] = False

        # covert data to nd arrays
        time = np.array([data[i]['t'] for i in range(len(data))]).astype(float)[mask]
        coordinates = np.array(convert_lat_long_to_rd(lat[mask], lon[mask])).T
        speed = np.array([data[i]['speed'] for i in range(len(data))]).astype(float)[mask]
        acc_side_1 = np.array([data[i]['acc_side_1'] for i in range(len(data))]).astype(float)[mask]
        acc_side_2 = np.array([data[i]['acc_side_2'] for i in range(len(data))]).astype(float)[mask]
        segment = np.array([data[i]['Segment'] for i in range(len(data))])[mask]

        # add data to res dictionary
        results[name]["time"] = time
        results[name]["coordinates"] = coordinates
        results[name]["speed"] = speed
        results[name]["acc_side_1"] = acc_side_1
        results[name]["acc_side_2"] = acc_side_2
        results[name]["segment"] = segment

    # write results to pickle
    with open(os.path.join(output_f, "inframon.pickle"), "wb") as f:
        pickle.dump(results, f)

    return results


def filter_data_at_point_coordinates(data, point_coordinates, search_radius):
    """
    Removes all ricardo coordinates and data in a range from a list of point coordinates

    :param data: ricardo results dictionary
    :param point_coordinates: point coordinates to be filtered out
    :param search_radius: radius around point coordinates which are to be filtered out

    :return:
    """

    ricardo_coordinates = data["coordinates"]

    # initialise kd tree
    tree = KDTree(ricardo_coordinates)

    # initialise mask array
    mask = np.ones(len(ricardo_coordinates)).astype(bool)

    # find all rila indices in range around point coordinates
    masked_indices = [j for i in tree.query_ball_point(point_coordinates, search_radius) for j in i]

    # set found indices at false
    mask[masked_indices] = False

    # remove coordinates and heights at found indices from results data
    data["coordinates"] = data["coordinates"][mask,:]
    data["time"] = data["time"][mask]
    data["speed"] = data["speed"][mask]
    data["acc_side_1"] = data["acc_side_1"][mask]
    data["acc_side_2"] = data["acc_side_2"][mask]
    data["segment"] = data["segment"][mask]

    return data


def filter_data_within_bounds(xbounds: np.ndarray, ybounds: np.ndarray, data: Dict):
    """
    Filters data within x bounds and y bounds

    :param xbounds: x limit of search area
    :param ybounds: y limit of search area
    :param data: ricardo dataset
    :return:
    """

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
    data["time"] = data["time"][mask]
    data["speed"] = data["speed"][mask]
    data["acc_side_1"] = data["acc_side_1"][mask]
    data["acc_side_2"] = data["acc_side_2"][mask]
    data["segment"] = data["segment"][mask]

    return data


if __name__ == '__main__':
    # filenames = [r"../data/Ricardo/Jan.json",
    #              r"../data/Ricardo/Jun.json",
    #              ]
    # read_inframon(filenames, "./")

    ricardo_data = load_inframon_data("./inframon.pickle")
    # plot_speed(ricardo_data["Jun"])

    xlim = [128326, 128410]
    ylim = [467723, 468058]

    get_data_within_bounds(ricardo_data["Jan"], xlim, ylim)


