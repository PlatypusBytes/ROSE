import fiona
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os
from typing import Dict, List

from datetime import datetime

def get_coordinates(feature):
    return np.array([list(coord) for coord in feature["geometry"]["coordinates"]])

def get_settlement(feature):
    dates = []
    settlements = []
    for k,v in feature["properties"].items():
        if k.startswith("v_"):
            date = datetime.strptime(k.strip("v_"),"%Y%m%d")
            settlement = v
            if date is not None and settlement is not None:
                dates.append(date)
                settlements.append(settlement)

    return dates, settlements

def read_geopackage(filename: str):
    """
    Reads id, coordinates, dates and settlements from Sensar geopackage
    :param filename: filename of the geopackage

    :return:
    """

    data = {}

    with fiona.open(filename) as layer:
        for feature in layer:
            data[feature['id']] = {'coordinates': None,
                                   'dates': None,
                                   'settlements': None}
            # data["id"] = feature['id']
            dates, settlements = get_settlement(feature)
            coordinates = get_coordinates(feature)
            data[feature['id']]["coverage_quality"] = feature["properties"]['coverage_quality']
            data[feature['id']]["coordinates"] = coordinates
            data[feature['id']]["dates"] = dates
            data[feature['id']]["settlements"] = settlements

    return data

def save_sensar_data(data: dict, filename: str) -> None:
    """
    Save data dictionary as pickle file

    :param data: data dictionary
    :param filename: full filename and path to the output json file
    """

    # if path does not exits: creates
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # save as json
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_sensar_data(filename: str):
    """
    Loads Sensar data from pickle

    :param filename:
    :return:
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def get_all_items_within_bounds(data: Dict, xlim, ylim):

    items_within_bounds = []
    for k, v in data.items():
        a=1+1
        if xlim[0] < max(v["coordinates"][:,0]) and xlim[1] > min(v["coordinates"][:,0]):
            if ylim[0] < max(v["coordinates"][:,1]) and ylim[1] > min(v["coordinates"][:,1]):
                items_within_bounds.append(v)

    return items_within_bounds

def get_item_at_coord(data: Dict, coord: List):

    for k, v in data.items():
        if coord[0] > min(v["coordinates"][:,0]) and coord[0] < max(v["coordinates"][:,0]):
            if coord[1] > min(v["coordinates"][:,1]) and coord[1] < max(v["coordinates"][:,1]):
                return v

    return

def plot_date_vs_settlement(item):
    plt.plot(item['dates'], item["settlements"], 'o')
    plt.show()


def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def map_settlement_at_starting_date(all_dates: List, all_settlements: List):
    """
    Maps the first measured settlement of all time series on the expected settlement based on data of time series with a
    longer history

    :param List[np.ndarray[float]]  all_dates: all date timestamps to be checked
    :param List[np.ndarray[float]] all_settlements: all settlements
    :return:
    """

    # find starting dates of each time series
    first_dates = np.array([dates[0] for dates in all_dates])

    # initialize mask
    mask = np.ones(len(first_dates), bool)

    # find all indices for each time serie which starts with the minimal value
    prev_indices = np.where(np.isclose(first_dates, min(first_dates[mask])))[0]

    # mask time series with longest history
    mask[prev_indices] = False

    # get all settlements and dates of all time series with the longest history
    prev_settlements = [all_settlements[idx] for idx in prev_indices]
    prev_dates = [all_dates[idx] for idx in prev_indices]

    # continue looping until all time series are checked
    while any(mask):

        # find all indices for each time serie which starts with the next minmal value
        next_indices = np.where(np.isclose(first_dates, min(first_dates[mask])))[0]
        mask[next_indices] = False

        # get all settlements and dates of all time series with the next longest history
        next_settlements = [all_settlements[idx] for idx in next_indices]
        next_dates = [all_dates[idx] for idx in next_indices]

        # starting date of the time serie with the next longest history
        next_date = next_dates[0][0]

        # concatenate all previous dates and settlements
        concatenated_dates = np.array([date for dates in prev_dates for date in dates])
        concatenated_settlements = np.array([settlement for settlements in prev_settlements for settlement in settlements])

        # create a trend line of all previous dates and settlements
        trend = np.polyfit(concatenated_dates,concatenated_settlements,3)

        # get expected settlement at starting date of next longest time history
        p = np.poly1d(trend)
        current_settlement = p(concatenated_dates[np.argmin(np.abs(concatenated_dates-next_date))])

        # map settlement at next longest time history
        next_settlements = [settlements + current_settlement for settlements in next_settlements]
        for idx, settlement in zip(next_indices,next_settlements):
            all_settlements[idx] = settlement

        # get all indices of already mapped settlements
        prev_indices = np.where(~mask)[0]

        #update previous settlements and dates
        prev_settlements = [all_settlements[idx] for idx in prev_indices]
        prev_dates = [all_dates[idx] for idx in prev_indices]

    return all_dates, all_settlements

def get_all_dates_and_settlement_as_sorted_array(items):
    """
    Gets all the dates and settlements from the sensar dictionary and sets them into a numpy array

    :param items: sensar data item list
    :return:
    """
    if items:
        all_dates = []
        all_settlements = []
        for item in items:
            dates = np.array([d.timestamp() for d in item['dates']])
            settlements = np.array(item['settlements'])
            all_settlements.append(settlements)
            all_dates.append(dates)

        all_dates, all_settlements = map_settlement_at_starting_date(all_dates, all_settlements)

        all_dates = np.array([date for dates in all_dates for date in dates])
        all_settlements = np.array([settlement for settlements in all_settlements for settlement in settlements])

        sorted_indices = np.argsort(all_dates)
        sorted_dates = all_dates[sorted_indices]
        sorted_settlements = all_settlements[sorted_indices]
        return sorted_dates, sorted_settlements
    return None, None

def get_statistical_information(dates_array: np.ndarray, settlements_array: np.ndarray):
    """
    Gets mean and standard deviation at each timestep from sensar settlement data

    :param dates_array: numpy array of the dates
    :param settlements_array: numpy array of the sensar settlement data
    :return:
    """

    # find uniwue time steps
    diff = np.diff(dates_array)
    step_idxs = np.insert(np.where(diff>0),0,0)

    all_means = np.array([])
    new_dates = np.array([])
    all_stds = np.array([])
    for i in range(1,len(step_idxs)):
        new_dates = np.append(new_dates,dates_array[step_idxs[i-1]])
        all_means = np.append(all_means, np.mean(settlements_array[step_idxs[i-1]:step_idxs[i]]))
        all_stds = np.append(all_stds, np.std(settlements_array[step_idxs[i-1]:step_idxs[i]]))

    return new_dates, all_means, all_stds

def plot_settlement_over_time(dates: np.ndarray, settlements: np.ndarray):
    """
    Plots settlement over time

    :param dates:
    :param settlements:
    :return:
    """
    # set timestamp dates into true dates
    if dates.dtype == np.float:
        dates = [datetime.fromtimestamp(int(date)) for date in dates]

    plt.plot(dates, settlements, 'o')

    plt.xlabel('Date [y]')
    plt.ylabel('Settlement [mm]')
    # plt.plot(new_dates2,all_means,'o')

def plot_settlements_from_item_list_over_time(items_within_bounds):
    fig, ax = plt.subplots()
    sorted_dates, sorted_settlements = get_all_dates_and_settlement_as_sorted_array(items_within_bounds)
    new_dates, all_means, all_stds = get_statistical_information(sorted_dates, sorted_settlements)
    plot_settlement_over_time(sorted_dates, sorted_settlements)
    plot_settlement_over_time(new_dates, all_means)

    return fig, ax



if __name__ == '__main__':

    # data = read_geopackage(r"../../data/Sensar/20190047_01_20210308/data/data.gpkg")
    # save_sensar_data(data, "../../data/Sensar/processed/processed_settlements.pickle")

    data = load_sensar_data("../data/Sensar/processed/processed_settlements.pickle")

    # items_within_bounds = get_all_items_within_bounds(data, [144276, 144465], [439011,439301])

    xlim = [128326, 128410]
    ylim = [467723, 468058]

    # items_within_bounds = get_all_items_within_bounds(data, [128162,128868], [467049, 470502])
    items_within_bounds = get_all_items_within_bounds(data, xlim, ylim)

    all_dates = []
    all_settlements = []
    for item in items_within_bounds:
        dates = np.array([d.timestamp() for d in item['dates']])
        settlements = np.array(item['settlements'])
        all_settlements.append(settlements)
        all_dates.append(dates)

    all_dates, all_settlements = map_settlement_at_starting_date(all_dates, all_settlements)

    all_dates2 = np.array([date for dates in all_dates for date in dates])
    all_settlements2 = np.array([settlement for settlements in all_settlements for settlement in settlements])

    all_dates = all_dates2
    all_settlements = all_settlements2

    sorted_indices = np.argsort(all_dates)

    sorted_dates = all_dates[sorted_indices]
    sorted_settlements = all_settlements[sorted_indices]
    # sorted_velocities = all_velocities[sorted_indices]

    diff = np.diff(sorted_dates)

    step_idxs = np.insert(np.where(diff>0),0,0)

    all_means = np.array([])
    new_dates = np.array([])
    all_stds = np.array([])
    for i in range(1,len(step_idxs)):
        new_dates = np.append(new_dates,sorted_dates[step_idxs[i-1]])
        all_means = np.append(all_means, np.mean(sorted_settlements[step_idxs[i-1]:step_idxs[i]]))
        all_stds = np.append(all_stds, np.std(sorted_settlements[step_idxs[i-1]:step_idxs[i]]))

    trend = np.polyfit(sorted_dates,sorted_settlements,1)
    trend_new = np.polyfit(new_dates,all_means,1)
    trend_3 = np.polyfit(new_dates,all_means + 2*all_stds,1)
    trend_4 = np.polyfit(new_dates,all_means - 2*all_stds,1)



    new_dates = [datetime.fromtimestamp(int(date)) for date in new_dates]
    sorted_dates = [datetime.fromtimestamp(int(date)) for date in sorted_dates]
    plt.plot(sorted_dates, sorted_settlements, 'o')
    plt.plot(new_dates,all_means,'o')
    # plt.plot(sorted_dates,np.polyval(trend,sorted_dates))
    # plt.plot(new_dates,np.polyval(trend_new,new_dates))
    # plt.plot(new_dates,np.polyval(trend_3,new_dates))
    # plt.plot(new_dates,np.polyval(trend_4,new_dates))

    plt.show()
