from data_proc.fugro import get_data_at_location, merge_data, interpolate_coordinates

import json
import numpy as np

def read_sos_data():
    """
    Reads sos json file
    :return:
    """
    path_sos_json = r"..\data_proc\SOS.json"
    with open(path_sos_json,'r') as f:
        sos_data = json.load(f)
    return sos_data


def create_rila_input_json(input_dir: str):
    # read sos data
    sos_data = read_sos_data()

    # read Rila data and which are generated on the same day
    fugro_data = get_data_at_location(input_dir, location="all",filetype="KRDZ")
    fugro_data = merge_data(fugro_data)

    # initialise dictionary
    input_dict = {"project_name": "proj1",
                  "data": {}}

    # loop over sos segments
    for name, segment in sos_data.items():

        # get coordinates of current segments
        coordinates = np.array(list(segment.values())[0]['coordinates'])

        # get coordinate limits
        xlim = [min(coordinates[:, 0]), max(coordinates[:, 0])]
        ylim = [min(coordinates[:, 1]), max(coordinates[:, 1])]

        # interpolate rila height data on the locations of the measurements in the first dataset
        dates, coordinate_data, interpolated_heights = interpolate_coordinates(fugro_data, xlim, ylim)

        # add coordinates, interpolated heights and dates to input dictionary
        if coordinate_data:
            input_dict["data"][name] = {"coordinates": coordinate_data[0].tolist(),
                                        "heights": interpolated_heights.tolist(),
                                        "dates": dates}
        else:
            input_dict["data"][name] = {"coordinates": [],
                                        "heights": [],
                                        "dates": []}

    # write example input json
    with open('example_rila_input.json', 'w') as json_file:
        json.dump(input_dict, json_file, indent=2, default=str)


if __name__ == '__main__':
    dir = r"..\data\Fugro\AMS-to-EIN"
    create_rila_input_json(dir)

