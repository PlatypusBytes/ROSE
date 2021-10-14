from data_proc.sensar import read_geopackage, filter_dataset, get_all_items_within_bounds,save_sensar_data, load_sensar_data, map_settlement_at_starting_date

import json
import numpy as np
from pathlib import Path

def read_sos_data():
    """
    Reads sos json file
    :return:
    """
    path_sos_json = r"..\data_proc\SOS.json"
    with open(path_sos_json,'r') as f:
        sos_data = json.load(f)
    return sos_data


def create_sensar_input_json(input_file: str):
    # read sos data
    sos_data = read_sos_data()
    file_path, file_name = Path(input_file).parents[0], Path(input_file).stem

    # open unprocessed geopackage and process
    if input_file.endswith(".gpkg"):
        sensar_data = read_geopackage(input_file)
        filtered_sensar_data = filter_dataset(sensar_data)
        save_sensar_data(filtered_sensar_data, str(Path(file_path/file_name)) + ".pickle")
    # open processed pickle file
    elif input_file.endswith(".pickle"):
        filtered_sensar_data = load_sensar_data(input_file)
    else:
        raise Exception("input file is not a .gpkg or .pickle file")

    # initialise dictionary
    input_dict={"project_name": "proj1",
                  "data": {}}
    # loop over sos segments
    for name, segment in sos_data.items():
        # get coordinates of current segments
        coordinates = np.array(list(segment.values())[0]['coordinates'])

        # get coordinate limits of sos segment
        xlim = [min(coordinates[:, 0]), max(coordinates[:, 0])]
        ylim = [min(coordinates[:, 1]), max(coordinates[:, 1])]

        # get sensar data within sos segment
        data_within_bounds = get_all_items_within_bounds(filtered_sensar_data, xlim, ylim)

        all_dates = []
        all_settlements = []

        # gather all settlements and dates
        if data_within_bounds:
            for item in data_within_bounds:
                # convert dates to timestamps
                dates = np.array([d.timestamp() for d in item['dates']])
                settlements = np.array(item['settlements'])

                if settlements.size > 0 and dates.size > 0:
                    all_settlements.append(settlements)
                    all_dates.append(dates)

            # maps all settlements at starting date
            all_dates, all_settlements = map_settlement_at_starting_date(all_dates, all_settlements)

            # add mapped settlements to dictionary
            for item, settlements in zip(data_within_bounds, all_settlements):
                item["settlements"] = settlements

        # add sos-segment sensar data to example input dictionary
        input_dict["data"][name] = dict(zip(range(len(data_within_bounds)), data_within_bounds))

    # convert arrays to lists
    for segment in input_dict["data"].values():
        if len(segment)>0:
            for item in segment.values():
                if isinstance(item["coordinates"],np.ndarray):
                    item["coordinates"] = item["coordinates"].tolist()
                if isinstance(item["settlements"], np.ndarray):
                    item["settlements"] = item["settlements"].tolist()

    # write example input json
    with open('example_sensar_input.json', 'w') as json_file:
        json.dump(input_dict, json_file, indent=2, default=str)


if __name__ == '__main__':
    processed_input_file = r"test_data/filtered_data.pickle"
    unprocessed_input_file = r"../data/Sensar/20190047_02_20210630/data/data.gpkg"
    create_sensar_input_json(processed_input_file)

# import cProfile
# cProfile.run("create_sensar_input_json(processed_input_file)","profiler4")
