import os
import pickle
import json
from pyproj import Transformer
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


def read_inframon(file_names, output_f):

    results = {}
    for fi in file_names:
        with open(fi, "r") as f:
            data = json.load(f)

        name = os.path.splitext(os.path.split(fi)[1])[0]
        results.update({name: {"time": [],
                               "coordinates": [],
                               "speed": [],
                               "acc_side_1": [],
                               "acc_side_2": [],
                               "segment": [],
                               }})

        for i in range(len(data)):
            results[name]["time"].append(data[i]['t'])
            results[name]["coordinates"].append(convert_lat_long_to_rd(data[i]['lat'], data[i]['lon']))
            results[name]["speed"].append(data[i]['speed'])
            results[name]["acc_side_1"].append(data[i]['acc_side_1'])
            results[name]["acc_side_2"].append(data[i]['acc_side_2'])
            results[name]["segment"].append(data[i]['Segment'])

    results[name]["acc_side_1"] = signal_proc.filter_sig(results[name]["acc_side_1"],
                                                         settings_filter["FS"], settings_filter["n"], settings_filter["cut-off"]).tolist()
    results[name]["acc_side_2"] = signal_proc.filter_sig(results[name]["acc_side_2"],
                                                         settings_filter["FS"], settings_filter["n"], settings_filter["cut-off"]).tolist()

    with open(os.path.join(output_f, "inframon.pickle"), "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':
    filenames = [r"../data/Ricardo/Jan.json",
                 r"../data/Ricardo/Jun.json",
                 ]
    read_inframon(filenames, "./")
