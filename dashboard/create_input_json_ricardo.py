
import os
import json
from typing import List

import numpy as np

from data_proc.ricardo import convert_lat_long_to_rd, get_data_within_bounds

def read_sos_data():
    """
    Reads sos json file
    :return:
    """
    path_sos_json = r"..\data_proc\SOS.json"
    with open(path_sos_json,'r') as f:
        sos_data = json.load(f)
    return sos_data

def read_inframon(file_names: List):
    results = {}
    for fi in file_names:
        with open(fi, "r") as f:
            data = json.load(f)

        # initialise dict
        name = os.path.splitext(os.path.split(fi)[1])[0]
        results = {"name": name,
                   "time": [],
                   "coordinates": [],
                   "speed": [],
                   "acc_side_1": [],
                   "acc_side_2": [],
                   "segment": []
                   }

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
        results["time"] = time
        results["coordinates"] = coordinates
        results["speed"] = speed
        results["acc_side_1"] = acc_side_1
        results["acc_side_2"] = acc_side_2
        results["segment"] = segment

    return results

def create_inframon_input_json():

    # read sos data
    sos_data = read_sos_data()

    # read inframon data
    inframon_data = read_inframon([r"..\data\Ricardo\Jan.json"])
    input_dict = {"project_name": "proj1",
                  "data": {}}

    # loop over sos segments
    for name, segment in sos_data.items():
        # get coordinates of current segments
        coordinates = np.array(list(segment.values())[0]['coordinates'])

        # get coordinate limits
        xlim = [min(coordinates[:, 0]), max(coordinates[:, 0])]
        ylim = [min(coordinates[:, 1]), max(coordinates[:, 1])]

        # get inframon data within current segment
        ricardo_data_within_bounds = get_data_within_bounds(inframon_data, xlim, ylim)
        input_dict["data"][name] = {"coordinates": ricardo_data_within_bounds["coordinates"].tolist(),
                            "speed": ricardo_data_within_bounds["speed"].tolist(),
                            "axle_acc": ricardo_data_within_bounds["acc_side_1"].tolist()}

    # write example input json
    with open('example_ricardo_input.json', 'w') as json_file:
        json.dump(input_dict, json_file, indent=2)

if __name__ == '__main__':
    create_inframon_input_json()