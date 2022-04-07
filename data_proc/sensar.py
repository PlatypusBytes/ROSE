import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pickle
import os
from typing import Dict, List
from datetime import datetime

import rose.utils.Kalman_Filter as KF


def filter_dataset(data: Dict) -> Dict:
    """
    Filter all sensar timeseries with kalman filter.

    :param data: Sensar data dictionary
    :return:
    """

    # loop over each time series
    for key, value in data.items():

        # filter signal from back to front
        flipped = filter_signal(data[key], is_flip=True)

        # filter signal from front to back
        non_flipped = filter_signal(data[key], is_flip=False)

        # combine flipped and non flipped filtered signals such that start and end of timeseries are filtered more
        # accurately
        if flipped.size >0 and non_flipped.size>0:
            combined = np.append(flipped[:int(len(flipped) / 2)],
                                 non_flipped[int(len(flipped) / 2):]
                                 - non_flipped[int(len(flipped) / 2)] + flipped[int(len(flipped) / 2)])
        else:
            combined = np.array([])

        # replace time series with filtered time series
        data[key]["settlements"] = combined

        # plt.plot(data[key]["dates"], flipped, 'x')
        # plt.plot(data[key]["dates"], non_flipped, 'v')
        # plt.plot(data[key]["dates"], combined, '^')
        #
        # plt.show()

    return data


def filter_signal(feature, is_flip=False):
    """
    Filters one time series with the Kalman filter

    :param feature: time series data
    :param is_flip: If true, time series is filtered from back to front
    :return:
    """

    dates = feature["dates"]

    # flip time series if required
    if is_flip:
        settlements = np.flip(feature["settlements"])
        # if settlements are not present, return
        if settlements.size == 0:
            return settlements
        settlements = settlements-settlements[0]
    else:
        settlements = np.array(feature["settlements"])
        # if settlements are not present, return
        if settlements.size == 0:
            return settlements

    # get dates in seconds
    timestamps = np.array([date.timestamp() for date in dates])
    # calculate timestamps relative to the first time stamp
    timestamps = timestamps - timestamps[0]

    # calculate timesteps in years
    dt = np.diff(timestamps) / 3600 / 24 / 365

    # calculate settlement differences at each time step
    sett_diff = np.diff(settlements)
    # calculate velocities
    velocities = np.append(0, sett_diff / dt)

    # set observations array
    observations = np.array([settlements, velocities]).T

    # calculate standard deviation of settlements and calculated velocities
    disp_std = np.std(sett_diff)
    velocity_std = np.std(velocities)

    # initialise kalman filter
    kf = KF.KalmanFilter([0, 0], 2, [disp_std, velocity_std], dt[0], independent=True)

    kf.initialise_control_matrices(dt)
    kf.error_covariance_measures(disp_std, velocity_std)

    # loop over each observation and perform iterative kalman filter operations
    for i in range(1, len(settlements)):
        # kf.update_control_matrices(dt[i - 1])
        kf.update_control_matrices_by_index(i-1)
        kf.predict_process_cov_matrix()
        kf.kalman_gain()
        kf.new_observation(observations[i])
        kf.predicted_state()
        kf.update_process_covariance_matrix()

    # assign Kalman filter results to updated settlements array
    if is_flip:
        updated_settlements = np.flip(np.array(kf.updated_x)[:, 0])
    else:
        updated_settlements = np.array(kf.updated_x)[:, 0]

    updated_settlements = updated_settlements - updated_settlements[0]

    return updated_settlements


def read_geopackage(filename: str) -> Dict:
    """
    Reads the geopackage of sensar data

    @param filename: File name of geopackage
    @return: dictionary with data
    """
    # connect to database
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()
    # queries all database
    query = 'SELECT * FROM "20190047SegmentedRailway"'
    data = cursor.execute(query)
    data_full = cursor.fetchall()

    # collect index dates
    idx_dates = [i for i, d in enumerate(data.description) if d[0].startswith("v_")]

    # collect index coverage
    idx_coverage = [i for i, d in enumerate(data.description) if d[0] == "coverage_quality"][0]

    # make dates list
    dates = []
    for i in idx_dates:
        aux = data.description[i][0].split("v_")[1]
        dates.append(datetime(int(aux[:4]), int(aux[4:6]), int(aux[6:8])))

    data_sensar = {}
    for i in range(len(data_full)):
        # get ID
        id = data_full[i][0]
        # query coordinates from index
        query = f'SELECT * FROM rtree_20190047SegmentedRailway_geom where ID= {id}'
        cursor.execute(query)
        coords = cursor.fetchall()[0]

        # add data to data_sensar
        data_sensar.update({f"{i + 1}": {"coordinates": np.array([[coords[1], coords[3]],
                                                                  [coords[2], coords[4]]]),
                                         "dates": dates,
                                         "settlements": [data_full[i][j] for j in idx_dates],
                                         "coverage_quality": data_full[i][idx_coverage]},
                            })

    # close connection
    conn.close()
    return data_sensar


def save_sensar_data(data: dict, filename: str) -> None:
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

def load_sensar_data(filename: str) -> Dict:
    """
    Loads Sensar data from pickle

    :param filename:
    :return:
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def get_all_items_within_bounds(data: Dict, xlim: List, ylim: List) -> List:
    """
    Gets all items from sensar data dictionary within the x limit and y limit
    :param data: sensar data dictionary
    :param xlim: x limit
    :param ylim: y limit
    :return:
    """

    items_within_bounds = []
    # loop over items in dictionary
    for k, v in data.items():

        # append item to list of coordinates are within bounds
        if xlim[0] < max(v["coordinates"][:,0]) and xlim[1] > min(v["coordinates"][:,0]):
            if ylim[0] < max(v["coordinates"][:,1]) and ylim[1] > min(v["coordinates"][:,1]):
                items_within_bounds.append(v)

    return items_within_bounds

def get_item_at_coord(data: Dict, coord: List):
    """
    Gets the item located on a coordinate
    :param data: sensar data dictionary
    :param coord: coordinate
    :return:
    """

    for k, v in data.items():
        if coord[0] > min(v["coordinates"][:,0]) and coord[0] < max(v["coordinates"][:,0]):
            if coord[1] > min(v["coordinates"][:,1]) and coord[1] < max(v["coordinates"][:,1]):
                return v

    return None

def plot_date_vs_settlement(item):
    """
    Plots date vs settlement
    :param item:
    :return:
    """
    plt.plot(item['dates'], item["settlements"], 'o')
    plt.show()


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
        # current_settlement = p(concatenated_dates[np.argmin(np.abs(concatenated_dates-next_date))])
        current_settlement = p(next_date)

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
    # check if items are available
    if items:
        all_dates = []
        all_settlements = []

        # gather all settlements and dates
        for item in items:
            # convert dates to timestamps
            dates = np.array([d.timestamp() for d in item['dates']])
            settlements = np.array(item['settlements'])

            if settlements.size>0 and dates.size>0:
                all_settlements.append(settlements)
                all_dates.append(dates)

        # maps all settlements at starting date
        all_dates, all_settlements = map_settlement_at_starting_date(all_dates, all_settlements)

        # convert dates and settlements to nd arrays
        all_dates = np.array([date for dates in all_dates for date in dates])
        all_settlements = np.array([settlement for settlements in all_settlements for settlement in settlements])

        # sort dates and settlements based on dates
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

    # find unique time steps
    diff = np.append(0,np.diff(dates_array))
    step_idxs = np.insert(np.where(diff>0),0,0)
    # step_idxs = np.append(step_idxs, len(dates_array)-1)
    all_means = np.array([])
    new_dates = np.array([])
    all_stds = np.array([])
    for i in range(1,len(step_idxs)):
        new_dates = np.append(new_dates,dates_array[step_idxs[i-1]])
        all_means = np.append(all_means, np.mean(settlements_array[step_idxs[i-1]:step_idxs[i]]))
        all_stds = np.append(all_stds, np.std(settlements_array[step_idxs[i-1]:step_idxs[i]]))

    return new_dates, all_means, all_stds


def plot_settlement_over_time(dates: np.ndarray, settlements: np.ndarray, **kwargs):
    """
    Plots settlement over time

    :param dates:
    :param settlements:
    :return:
    """
    # set timestamp dates into true dates
    if dates.dtype == np.float:
        dates = [datetime.fromtimestamp(int(date)) for date in dates]

    plt.plot(dates, settlements, **kwargs)

    plt.xlabel('Date [y]')
    plt.ylabel('Settlement [mm]')


def plot_settlements_from_item_list_over_time(items_within_bounds,date_lim=None, fig=None, position=111):
    """

    :param items_within_bounds:
    :param fig:
    :param position:
    :return:
    """

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(position)

    sorted_dates, sorted_settlements = get_all_dates_and_settlement_as_sorted_array(items_within_bounds)
    new_dates, all_means, all_stds = get_statistical_information(sorted_dates, sorted_settlements)
    plot_settlement_over_time(sorted_dates, sorted_settlements, marker="o", color='blue', linewidth=0, markersize=4)
    plot_settlement_over_time(new_dates, all_means, marker="o", color='red', linewidth=0, markersize=4)
    plot_settlement_over_time(new_dates, all_means + all_stds, marker="_", color="red", linewidth=0)
    plot_settlement_over_time(new_dates, all_means - all_stds, marker="_", color="red", linewidth=0)

    if date_lim is not None:
        ax.set_xlim(date_lim)

    return fig, ax


def calculate_centroids(data: Dict):
    """
    Calculates centroids of all values in sensar data dict
    :param data: sensar data dictionary
    :return:
    """

    # get all coordinates in data dict
    coordinates = [value["coordinates"] for value in list(data.values())]

    # calculate and return centroids
    return np.array([[np.mean(coordinate[:,0]), np.mean(coordinate[:,1])] for coordinate in coordinates])


def filter_data_at_point_coordinates(res, point_coordinates):
    """
    Checks if point coordinates lay within sensar data lines, if so, the corresponding sensar data is filtered

    :param res: sensar results dictionary
    :param point_coordinates: point coordinates to be filtered out

    :return:
    """

    # gets coordinates of the sensar data
    coordinates = [value["coordinates"] for value in list(res.values())]

    # get the end coordinates of each Sensar data line
    min_coordinates = np.array([[np.min(coordinate[:, 0]), np.min(coordinate[:, 1])] for coordinate in coordinates])
    max_coordinates = np.array([[np.max(coordinate[:, 0]), np.max(coordinate[:, 1])] for coordinate in coordinates])

    # initialize mask array as zeros
    mask = np.zeros(min_coordinates.shape[0])

    # find all points which are located at sensar data and adds to mask array
    for coord in point_coordinates:

        mask += (min_coordinates[:,0] <= coord[0]).astype(int) * \
                (max_coordinates[:, 0] >= coord[0]).astype(int) * \
                (min_coordinates[:, 1] <= coord[1]).astype(int) * \
                (max_coordinates[:, 1] >= coord[1]).astype(int)

    # invert and convert mask array as boolean array
    mask = ~mask.astype(bool)

    # convert dictionary to np array
    np_data = np.array(list(res.items()))

    # filter data
    filtered_data = np_data[mask,:]

    # convert filtered data to dictionary
    filtered_data = dict(filtered_data)

    return filtered_data


def filter_data_within_bounds(xbounds: np.ndarray, ybounds: np.ndarray, data: Dict):
    """
    Filters data which lies partly or completely within x bounds and y bounds

    :param xbounds: x limit of search area
    :param ybounds: y limit of search area
    :param data: sensar data dictionary
    :return:
    """

    # gets coordinates of the sensar data
    coordinates = [value["coordinates"] for value in list(data.values())]

    # get the end coordinates of each Sensar data line
    min_coordinates = np.array([[np.min(coordinate[:, 0]), np.min(coordinate[:, 1])] for coordinate in coordinates])
    max_coordinates = np.array([[np.max(coordinate[:, 0]), np.max(coordinate[:, 1])] for coordinate in coordinates])

    # initialize mask array as zeros
    mask = np.zeros(min_coordinates.shape[0])

    # find all lines which are partly or completely within each x and y limit
    for xlim, ylim in zip(xbounds, ybounds):
        mask += (min_coordinates[:, 0] >= xlim[0]).astype(int) * \
                (min_coordinates[:, 0] <= xlim[1]).astype(int) * \
                (min_coordinates[:, 1] >= ylim[0]).astype(int) * \
                (min_coordinates[:, 1] <= ylim[1]).astype(int) + \
                (max_coordinates[:, 0] >= xlim[0]).astype(int) * \
                (max_coordinates[:, 0] <= xlim[1]).astype(int) * \
                (max_coordinates[:, 1] >= ylim[0]).astype(int) * \
                (max_coordinates[:, 1] <= ylim[1]).astype(int)

    # invert and convert mask array as boolean array
    mask = ~mask.astype(bool)

    # convert dictionary to np array
    np_data = np.array(list(data.items()))

    # filter data
    filtered_data = np_data[mask,:]

    # convert filtered data to dictionary
    filtered_data = dict(filtered_data)

    return filtered_data


def plot_old_and_new_dataset(old_sensar, new_sensar,sos_dict):
    import SoS
    from pathlib import Path


    sensar_dates = list(old_sensar.values())[0]["dates"]

    min_date = min(sensar_dates)
    max_date = max(sensar_dates)

    date_lim = [min_date, max_date]

    # loop over sos segments
    for name, segment in sos_dict.items():

        # if name == "Segment 1072":
        #
        # if name == "Segment 1003":
        # initialise figure
            fig = plt.figure(figsize=(20, 10))
            plt.tight_layout()

            # get coordinates of current segments
            coordinates = np.array(list(segment.values())[0]['coordinates'])

            # get coordinate limits
            xlim = [min(coordinates[:, 0]), max(coordinates[:, 0])]
            ylim = [min(coordinates[:, 1]), max(coordinates[:, 1])]

            # fugro.plot_date_vs_mileage(xlim, ylim, fugro_dict)

            # add plot of highlighted sos segments
            _, _ = SoS.ReadSosScenarios.plot_highlighted_sos(sos_data, name, fig=fig, position=221)
            plt.grid()


            # add plot of Sensar settlement measurements within the current segment
            sensar_items_within_bounds1 = get_all_items_within_bounds(old_sensar, xlim, ylim)
            if sensar_items_within_bounds1:
                _, ax = plot_settlements_from_item_list_over_time(sensar_items_within_bounds1, date_lim=date_lim,
                                                                        fig=fig, position=223)
                plt.grid()

                ax.title.set_text('Old dataset')

                ylim1 = plt.subplot(223).get_ylim()


            sensar_items_within_bounds2 = get_all_items_within_bounds(new_sensar, xlim, ylim)
            if sensar_items_within_bounds2:
                _, ax = plot_settlements_from_item_list_over_time(sensar_items_within_bounds2, date_lim=date_lim,
                                                                        fig=fig, position=224)
                plt.grid()

                ax.title.set_text('New dataset')
                ylim2 = plt.subplot(224).get_ylim()

            if sensar_items_within_bounds1 and sensar_items_within_bounds2:

                plt.subplot(223).set_ylim(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
                plt.subplot(224).set_ylim(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))


            fig.suptitle(name)
            fig.savefig(Path("sensar_old_vs_new", name))

            plt.close(fig)


if __name__ == '__main__':

    import json
    data = read_geopackage(r"../data/Sensar/20190047_02_20210630\data/data.gpkg")
    save_sensar_data(data, "../data/Sensar/processed/processed_settlements_2.pickle")
    #
    old_data = load_sensar_data("../data/Sensar/processed/processed_settlements.pickle")
    new_data = load_sensar_data("../data/Sensar/processed/processed_settlements_2.pickle")

    # filter_dataset(data)
    #
    # save_sensar_data(data, "../data/Sensar/processed/filtered_processed_settlements_combined2.pickle")


    sos_fn = "../data_proc/SOS.json"
    with open(sos_fn, 'r') as f:
        sos_data = json.load(f)

    plot_old_and_new_dataset(old_data, new_data, sos_data)


    #
    # import data_discontinuities as dd
    #
    # fn = r"D:\software_development\rose\data\data_discontinuities\wissel.json"
    #
    # all_coordinates = dd.get_coordinates_from_json(fn)
    # x_bounds, y_bounds = dd.get_bounds_of_lines(all_coordinates)
    #
    #
    # point_coordinates = np.array([[122730.096, 487773.31], [138101.172, 453431.389],[0,0]])
    # # filter_data_at_point_coordinates(res, point_coordinates,1)
    #
    # filter_data_at_point_coordinates(data, point_coordinates)
    #
    # # filtered_data = filter_data_within_bounds(x_bounds, y_bounds, data)
    # # centroids = calculate_centroids(data)
    #
    # # items_within_bounds = get_all_items_within_bounds(data, [144276, 144465], [439011,439301])
    #
    # xlim = [128326, 128410]
    # ylim = [467723, 468058]
    #
    # # items_within_bounds = get_all_items_within_bounds(data, [128162,128868], [467049, 470502])
    # items_within_bounds = get_all_items_within_bounds(data, xlim, ylim)
    #
    # all_dates = []
    # all_settlements = []
    # for item in items_within_bounds:
    #     dates = np.array([d.timestamp() for d in item['dates']])
    #     settlements = np.array(item['settlements'])
    #     all_settlements.append(settlements)
    #     all_dates.append(dates)
    #
    # all_dates, all_settlements = map_settlement_at_starting_date(all_dates, all_settlements)
    #
    # all_dates2 = np.array([date for dates in all_dates for date in dates])
    # all_settlements2 = np.array([settlement for settlements in all_settlements for settlement in settlements])
    #
    # all_dates = all_dates2
    # all_settlements = all_settlements2
    #
    # sorted_indices = np.argsort(all_dates)
    #
    # sorted_dates = all_dates[sorted_indices]
    # sorted_settlements = all_settlements[sorted_indices]
    # # sorted_velocities = all_velocities[sorted_indices]
    #
    # diff = np.diff(sorted_dates)
    #
    # step_idxs = np.insert(np.where(diff>0),0,0)
    #
    # all_means = np.array([])
    # new_dates = np.array([])
    # all_stds = np.array([])
    # for i in range(1,len(step_idxs)):
    #     new_dates = np.append(new_dates,sorted_dates[step_idxs[i-1]])
    #     all_means = np.append(all_means, np.mean(sorted_settlements[step_idxs[i-1]:step_idxs[i]]))
    #     all_stds = np.append(all_stds, np.std(sorted_settlements[step_idxs[i-1]:step_idxs[i]]))
    #
    # trend = np.polyfit(sorted_dates,sorted_settlements,1)
    # trend_new = np.polyfit(new_dates,all_means,1)
    # trend_3 = np.polyfit(new_dates,all_means + 2*all_stds,1)
    # trend_4 = np.polyfit(new_dates,all_means - 2*all_stds,1)
    #
    #
    #
    # new_dates = [datetime.fromtimestamp(int(date)) for date in new_dates]
    # sorted_dates = [datetime.fromtimestamp(int(date)) for date in sorted_dates]
    # plt.plot(sorted_dates, sorted_settlements, 'o')
    # plt.plot(new_dates,all_means,'o')
    # # plt.plot(sorted_dates,np.polyval(trend,sorted_dates))
    # # plt.plot(new_dates,np.polyval(trend_new,new_dates))
    # # plt.plot(new_dates,np.polyval(trend_3,new_dates))
    # # plt.plot(new_dates,np.polyval(trend_4,new_dates))
    #
    # plt.show()
