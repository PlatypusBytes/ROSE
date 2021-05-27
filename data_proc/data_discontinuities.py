from typing import Dict

import fiona
import json

def get_coordinates_from_json(filename):
    """
    Gets coordinates from discontinuity json file

    :param filename: json file name
    :return:
    """
    with open(filename) as json_file:
        data = json.load(json_file)

        all_coordinates = []
        for feature in data["features"]:
            all_coordinates.append(feature["geometry"]["coordinates"])

        return all_coordinates




if __name__ == '__main__':

    fn = r"D:\software_development\rose\data\data_discontinuities\wissel.json"

    get_coordinates_from_json(fn)
    # with open(fn) as json_file:
    #     data = json.load(json_file)
    #
    #     a=1+1
    #
    # # json.load()
    #
    # # fn  = r"D:\software_development\rose\data\data_discontinuities\overweg.gpkg"
    # fn = r"D:\software_development\rose\data\data_discontinuities\kruising_4.gpkg"
    # read_kruising_geopackage(fn)
    # pass