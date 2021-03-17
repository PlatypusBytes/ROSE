import fiona
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

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

def get_item_at_coord(data, coord):

    for k, v in data.items():
        if coord[0] > min(v["coordinates"][:,0]) and coord[0] < max(v["coordinates"][:,0]):
            if coord[1] > min(v["coordinates"][:,1]) and coord[1] < max(v["coordinates"][:,1]):
                return v

    return

def plot_date_vs_settlement(item):
    plt.plot(item['dates'], item["settlements"], 'o')
    plt.show()



if __name__ == '__main__':

    # data = read_geopackage(r"../../data/Sensar/20190047_01_20210308/data/data.gpkg")
    # save_sensar_data(data, "../../data/Sensar/processed/processed_settlements.pickle")

    data = load_sensar_data("../../data/Sensar/processed/processed_settlements.pickle")

    item1 = get_item_at_coord(data, [144281.8, 439303.8])

    item2 = get_item_at_coord(data, [144322.7, 439234.9])
    item3 = get_item_at_coord(data, [144298.6, 439274.9])

    plot_date_vs_settlement(item3)
    # plot_date_vs_settlement(item2)
    pass