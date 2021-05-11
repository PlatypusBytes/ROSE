import os
import pickle
import json

from typing import List, Dict

from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt

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
        coordinates = convert_lat_long_to_rd(lat[mask], lon[mask])
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

    #     # results[name]["acc_side_1"] = signal_proc.filter_sig(results[name]["acc_side_1"],
    #     #                                                      settings_filter["FS"], settings_filter["n"], settings_filter["cut-off"]).tolist()
    #     # results[name]["acc_side_2"] = signal_proc.filter_sig(results[name]["acc_side_2"],
    #     #                                                      settings_filter["FS"], settings_filter["n"], settings_filter["cut-off"]).tolist()
    #
    #     tmp = signal_proc.filter_sig(results[name]["acc_side_1"],
    #                                                          settings_filter["FS"], settings_filter["n"],
    #                                                          settings_filter["cut-off"]).tolist()
    #     tmp2 = signal_proc.filter_sig(results[name]["acc_side_2"],
    #                                                          settings_filter["FS"], settings_filter["n"],
    #                                                          settings_filter["cut-off"]).tolist()
    #
    #     f1, a1, _, _ = signal_proc.fft_sig(results[name]["acc_side_2"], settings_filter["FS"])
    #     f2, a2, _, _ = signal_proc.fft_sig(np.array(tmp2), settings_filter["FS"])
    #
    #
    # plt.plot(f1, a1)
    # plt.plot(f2, a2)
    # plt.xlim(0,  settings_filter["FS"]/2)
    # plt.show()
    #
    # plt.plot(results[name]["acc_side_2"])
    # plt.plot(tmp2)
    # plt.show()

    # write results to pickle
    with open(os.path.join(output_f, "inframon.pickle"), "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':
    filenames = [r"../data/Ricardo/Jan.json",
                 r"../data/Ricardo/Jun.json",
                 ]
    read_inframon(filenames, "./")
