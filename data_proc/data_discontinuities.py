import numpy as np
import matplotlib.pyplot as plt
import json

def get_bounds_of_lines(all_coordinates):
    """
    gets x-bounds and y-bounds of all entries

    :param all_coordinates: all coordinates in json file structure
    :return:
    """

    # initialize  bounds
    x_bounds = np.zeros((len(all_coordinates),2))
    y_bounds = np.zeros((len(all_coordinates), 2))

    # loop over each entry
    for i, coordinates in enumerate(all_coordinates):

        # initialize bounds of line in entry
        line_x_bounds = np.zeros((len(coordinates), 2))
        line_y_bounds = np.zeros((len(coordinates), 2))

        # loop over each line
        for j, line in enumerate(coordinates):

            # get x and y bounds of each line
            line_np = np.array(line)
            line_x_bounds[j,:] = [line_np[:,0].min(), line_np[:,0].max()]
            line_y_bounds[j,:] = [line_np[:,1].min(), line_np[:,1].max()]


        # get x and y bounds of each entry
        x_bounds[i, :] = [line_x_bounds.min(), line_x_bounds.max()]
        y_bounds[i, :] = [line_y_bounds.min(), line_y_bounds.max()]

    return x_bounds, y_bounds

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

    all_coordinates = get_coordinates_from_json(fn)
    x_bounds, y_bounds = get_bounds_of_lines(all_coordinates)

    i = 100

    plt.plot(np.array(all_coordinates[i][0])[:, 0], np.array(all_coordinates[i][0])[:, 1], marker='o')
    plt.plot(np.array(all_coordinates[i][1])[:, 0], np.array(all_coordinates[i][1])[:, 1], marker='x')
    plt.plot(np.array(all_coordinates[i][2])[:, 0], np.array(all_coordinates[i][2])[:, 1], marker='v')

    plt.show()
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