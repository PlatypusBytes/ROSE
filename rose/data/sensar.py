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
            # if coord[1] > min(v["coordinates"][:,1]) and coord[1] < max(v["coordinates"][:,1]):
            #     return v



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


if __name__ == '__main__':

    # data = read_geopackage(r"../../data/Sensar/20190047_01_20210308/data/data.gpkg")
    # save_sensar_data(data, "../../data/Sensar/processed/processed_settlements.pickle")

    data = load_sensar_data("../../data/Sensar/processed/processed_settlements.pickle")
    #

    items_within_bounds = get_all_items_within_bounds(data, [144276, 144465], [439011,439301])
    # items_within_bounds = get_all_items_within_bounds(data, [139361.8, 144281.8], [439303.0,450982.1])

    # item1 = get_item_at_coord(data, [144281.8, 439303.0])
    #
    # #
    # item2 = get_item_at_coord(data, [144287.5, 439293.8])
    # item3 = get_item_at_coord(data, [144313.9, 439249])
    # item4 = get_item_at_coord(data, [139361.8, 450982.1])
    # #
    # # # plot_date_vs_settlement(item3)
    # plot_date_vs_settlement(item1)
    # plot_date_vs_settlement(item2)
    # plot_date_vs_settlement(item3)
    # plot_date_vs_settlement(item4)
    #
    # plot_date_vs_settlement(item3)

    all_dates = np.array([])
    all_settlements = np.array([])
    for item in items_within_bounds:
        dates = np.array([d.timestamp() for d in item['dates']])
        all_dates = np.append(all_dates,dates)
        settlements = np.array(item['settlements'])
        all_settlements = np.append(all_settlements,settlements)

        # np.polyfit(dates,items_within_bounds[0]['settlements'])

        # trend, V = np.polyfit(dates,items_within_bounds[0]['settlements'],3, cov='unscaled')
        # # std =
        # trendpoly = np.poly1d(trend)
        # std_poly = np.poly1d(np.sqrt(np.diagonal(V)))
        # plt.plot(dates,trendpoly(dates))
        # plt.plot(dates,std_poly(dates))
        # plt.plot(dates, item['settlements'], 'o')
        # plt.show()

    sorted_indices = np.argsort(all_dates)

    sorted_dates = all_dates[sorted_indices]
    sorted_settlements = all_settlements[sorted_indices]

    diff = np.diff(sorted_dates)

    step_idxs = np.insert(np.where(diff>0),0,0)

    all_means = np.array([])
    new_dates = np.array([])
    all_stds = np.array([])
    for i in range(1,len(step_idxs)):
        new_dates = np.append(new_dates,sorted_dates[step_idxs[i-1]])
        all_means = np.append(all_means, np.mean(sorted_settlements[step_idxs[i-1]:step_idxs[i]]))
        all_stds = np.append(all_stds, np.std(sorted_settlements[step_idxs[i-1]:step_idxs[i]]))

    trend, V = np.polyfit(sorted_dates,sorted_settlements,3, cov=True)
    trendpoly = np.poly1d(trend)

    trend_new, V2 = np.polyfit(new_dates,all_means,3, cov=True)
    trendpoly_new = np.poly1d(trend_new)

    trend_3, V3 = np.polyfit(new_dates,all_means + all_stds,3, cov=True)
    trendpoly_3 = np.poly1d(trend_3)

    trend_4, V3 = np.polyfit(new_dates,all_means - all_stds,3, cov=True)
    trendpoly_3 = np.poly1d(trend_3)
    # np.std(rolling_window(sorted_settlements, 1e8))

    std = np.sqrt(np.diagonal(V))


    plt.plot(sorted_dates, sorted_settlements, 'o')
    plt.plot(new_dates,all_means,'o')
    # plt.plot(new_dates,all_means+all_stds,'o')
    # plt.plot(new_dates,all_means-all_stds,'o')
    plt.plot(sorted_dates,np.polyval(trend,sorted_dates))
    plt.plot(new_dates,np.polyval(trend_new,new_dates))
    plt.plot(new_dates,np.polyval(trend_3,new_dates))
    plt.plot(new_dates,np.polyval(trend_4,new_dates))
    # plt.plot(sorted_dates,trendpoly(sorted_dates))

    plt.show()
    #
    # test = np.polyfit(items_within_bounds[0]['dates'],items_within_bounds[0]['settlements'])

    pass