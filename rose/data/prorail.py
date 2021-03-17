import os
import json
from zipfile import ZipFile
from pyproj import Transformer
from tqdm import tqdm
from datetime import datetime
import copy

import numpy as np

def sort_track(data, key):

    track = []
    for k,v in data.items():
        if v['track'] == key:
            track.append(v)
    return sorted(track, key=lambda x: x['km'], reverse=False)

def calculate_settlement(data: dict):
    sorted_gh = sort_track(data, 'Track GH')
    sorted_gt = sort_track(data, 'Track GT')


    date_time = datetime.strptime(sorted_gh[0]['settlement']['time'][0],'%Y-%m-%d %H:%M:%S')


    dx_gh = np.diff([x['km'] for x in sorted_gh])
    slope_gh = np.array([x['settlement']['value'] for x in sorted_gh]).astype(float)
    # gh_date_time = np.array([datetime.strptime(sorted_gh[0]['settlement']['time'][0],'%Y-%m-%d %H:%M:%S'])

    heights_gh_tmp = np.cumsum(slope_gh[1:,:].T* dx_gh,axis=1)

    heights_gh = heights_gh_tmp - heights_gh_tmp[0,:]

    import matplotlib.pyplot as plt

    plt.plot(heights_gh[:,2])
    plt.plot(heights_gh[:,10])

    plt.plot(heights_gh[:,15])

    plt.legend(['2','10','15'])
    plt.show()
    settlement_gh = 0



def read_csv(file_name: str, key: list, header: int = 0) -> dict:
    """
    Reads from a generic CSV the columns (keys)

    :param file_name: CSV file name
    :param key: columns name
    :param header: number of lines of header
    :return: Dictionary with the CSV information. Columns as keys.
    """
    with open(file_name, 'r') as f:
        data = f.read().splitlines()

    data = [i.split(',') for i in data[header:]]
    head = data[header]

    idx = [data[0].index(k) for k in key]

    data_dic = {}
    for i in idx:
        aux = [j[i] for j in data if j[0] != "" and j[0] != "name"]
        data_dic.update({head[i]: aux})

    return data_dic


def read_csv_traces(file_name: str, key: list, header: int = 0) -> dict:
    """
    Reads the traces CSV

    :param file_name: CSV file name
    :param key: columns name
    :param header: number of lines of header
    :return: Dictionary with the CSV information. Columns as keys.
    """
    with open(file_name, 'r') as f:
        data = f.read().splitlines()

    data = [i.split(',') for i in data[header:]]
    head = data[header]

    data_dic = dict(zip(key, (dict() for _ in key)))

    # for each key
    for k in key:
        # find index of type
        i1 = data[0].index("type")
        # find index of name
        i2 = data[0].index("name")

        # index containing key
        idx = [i for i, val in enumerate(data) if val[i1] == k]

        # determine number of tracks
        nb_tracks = list(set([data[i][i2] for i in idx]))

        # create auxiliar dict for each track
        aux_dic = dict(zip(nb_tracks, (dict([("time", []),
                                             ("position", []),
                                             ("value", [])]) for _ in nb_tracks)))

        # update results dictionary
        data_dic[k].update(aux_dic)

        # new file format
        if len(head) > 5:
            for typ in nb_tracks:
                # indices containing the key and name
                id = [i for i, val in enumerate(data) if val[i1] == k and val[i2] == typ]

                # find remaining indexes
                i3 = data[0].index("time")

                for j in id:
                    # convert time
                    datetime_object = datetime.strptime(data[j][i3], "%Y-%m-%d %H:%M:%S")
                    data_dic[k][typ]["position"].extend(data[0][4:])
                    # data_dic[k][typ]["time"].extend([(datetime_object - datetime(1900, 1, 1)).total_seconds()] * len(data[0][4:]))
                    data_dic[k][typ]["time"].extend([str(datetime_object)] * len(data[0][4:]))
                    data_dic[k][typ]["value"].extend(data[j][4:])

        # old file format
        else:
            for typ in nb_tracks:
                # indices containing the key and name
                id = [i for i, val in enumerate(data) if val[i1] == k and val[i2] == typ]

                # find remaining indexes
                i3 = data[0].index("time")
                i4 = data[0].index("position")
                i5 = data[0].index("value")

                # add data to results dict
                for i in id:
                    # convert time
                    datetime_object = datetime.strptime(data[i][i3].split(".")[0].replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                    # data_dic[k][data[i][i2]]["time"].append((datetime_object - datetime(1900, 1, 1)).total_seconds())
                    data_dic[k][data[i][i2]]["time"].append(str(datetime_object))
                    data_dic[k][data[i][i2]]["position"].append(data[i][i4])
                    data_dic[k][data[i][i2]]["value"].append(data[i][i5])

    return data_dic


def convert_to_rd(coord: list) -> list:
    """
    Converts coordinates from Long-Lat to RD coordinates

    :param coord: list with coordinate [long, lat]
    :return: list with coordinate [RDx, RDy]
    """
    # transform coordinates
    transformer = Transformer.from_crs("epsg:4326", "epsg:28992")
    x, y = transformer.transform(coord[0], coord[1])
    return [x, y]


def sensor_location(file_name: str) -> dict:
    """
    Reads the sensor CSV metadata

    :param file_name: CSV file name of the sensor data
    :return: Dictionary with sensor metadata
    """
    loc = read_csv(file_name, ["track", "KM", "sensor", "GPS_Lat", "GPS_Long", "Hoogte", "sensor_type", "Name"])

    sensor = dict()

    # for every sensor name: collect data
    for i, val in enumerate(loc["sensor"]):
        # parse loc into new structure
        sensor.update({val: {"km": float(loc["KM"][i]),
                             "coordinates": convert_to_rd([float(loc["GPS_Lat"][i]), float(loc["GPS_Long"][i])]),
                             "depth": float(loc["Hoogte"][i]),
                             "sensor-type": loc["sensor_type"][i],
                             "code-name": loc["Name"][i],
                             "track": loc["track"][i],
                             }})

    return sensor


def collect_files_moisture(sensor: dict, folder: str, key: list, key_name: str) -> dict:
    """
    Reads all the moisture csv data files and combines the data with the sensor information

    :param sensor: dictionary with the sensor metadata
    :param folder: root folder where all the csv files are
    :param key: list with the headers of the csv file to be read
    :param key_name: str containing the key that has the sensor name
    :return: dictionary with all the sensor data and metadata
    """
    # collect all files in folder
    csv_data = dict(zip(key, ([] for _ in key)))

    # go through all the csv files in folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                # csv path
                csv_file = os.path.join(root, file)
                # read csv
                aux = read_csv(csv_file, key)
                # add to csv dict
                for k in key:
                    csv_data[k].extend(aux[k])

    # check list of set
    unique_names = list(set(csv_data[key_name]))

    # remove the key name to update dictionary
    key.remove(key_name)

    results = dict()
    # for each unique sensor name: collect all information, and couple it with sensor
    for nam in unique_names:
        # collect all indexes
        idx = [i for i, val in enumerate(csv_data[key_name]) if val == nam]

        # update the dictionary with sensor name
        results.update({nam: {}})

        # add time and value of the data
        for k in key:
            results[nam].update({k: [csv_data[k][i] for i in idx]})

        # add additional sensor information. if the name does not exist in information does not add
        if nam in sensor.keys():
            results[nam].update(sensor[nam])

    return results


def collect_files_traces(sensor: dict, folder: str, key: list, track: list, keys_data: list, key_name: str) -> dict:
    """
    Reads all the traces csv data files and combines the data with the sensor information

    :param sensor: dictionary with the sensor metadata
    :param folder: root folder where all the csv files are
    :param key: list with the headers of the csv file to be read
    :param track: list with track names
    :param keys_data: list with csv data names
    :param key_name: str containing the key that has the sensor name
    :return: dictionary with all the sensor data and metadata
    """

    # collect all files in folder
    aux_dic = dict(zip(keys_data, ([] for _ in keys_data)))
    aux_dic = dict(zip(track, (copy.deepcopy(aux_dic) for _ in track)))

    csv_data = dict(zip(key, (copy.deepcopy(aux_dic) for _ in key)))

    # go through all the csv files in folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                # csv path
                csv_file = os.path.join(root, file)

                # read csv
                aux = read_csv_traces(csv_file, key)

                # add to csv dict
                for k in key:
                    for t in track:
                        for d in keys_data:
                            try:
                                csv_data[k][t][d].extend(aux[k][t][d])
                            except KeyError:
                                continue

    # check list of unique positions
    position = []
    for k in key:
        for t in track:
            position.extend(list(map(float, csv_data[k][t][key_name])))

    # unique position
    position = list(set(position))

    # get name for every position
    aux = [[s, sensor[s]["track"], sensor[s]["km"]] for i, s in enumerate(sensor)]
    names = []
    for p in position:
        idx = [i for i, val in enumerate(aux) if val[2] == float(p)]
        names.extend([aux[i][0] for i in idx if aux[i][1]])

    names = list(set(names))

    # update keys_data
    keys_data.remove(key_name)

    results = dict()
    # for each unique sensor name: collect all information, and couple it with sensor
    for nam in names:
        results.update({nam: dict(zip(key, ({} for _ in key)))})

        # get position coordinate
        idx = [i for i, val in enumerate(aux) if val[0] == nam][0]
        track = aux[idx][1]
        coord = aux[idx][2]
        # collect data for each key
        for k in key:
            results[nam][k].update(dict(zip(keys_data, ([] for _ in keys_data))))
            idx = [i for i, val in enumerate(csv_data[k][track][key_name]) if float(val) == coord]
            for d in keys_data:
                results[nam][k][d].extend([csv_data[k][track][d][i] for i in idx])

        # update the dictionary with sensor name
        results[nam].update(sensor[nam])

    return results


def save_data(data: dict, filename: str) -> None:
    """
    Save data dictionary as json file

    :param data: data dictionary
    :param filename: full filename and path to the output json file
    """

    # if path does not exits: creates
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # save as json
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return


def unzip_file(file_name: str, output_location: str) -> None:
    """
    Unzip a file name into a specific location showing progress bar

    :param file_name: Zip filename
    :param output_location: path for the extraction
    """

    # if output location does not exist: create
    if not os.path.isdir(output_location):
        os.makedirs(output_location)

    print("Extracting zip file...")

    # Open your .zip file
    with ZipFile(file=file_name) as zip_file:
        # Loop over each file
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            # Extract each file to another directory
            zip_file.extract(member=file, path=output_location)

    print("Zip extraction completed.")

    return


if __name__ == "__main__":
    # # unzip_file("../../data/ProRail/Culemborg.zip", "../../data/ProRail/Culemborg")
    # sens = sensor_location(r"../../data/ProRail/Culemborg_sensor_locaties.csv")
    #
    # res = collect_files_moisture(sens, r"../../data/ProRail/Culemborg/measurements", ["name", "time", "value"], "name")
    # save_data(res, "../../data/ProRail/processed/processed_moisture.json")
    #
    # res = collect_files_traces(sens, r"../../data/ProRail/Culemborg/traces",
    #                            ["temperature", "cant", "settlement"],
    #                            ["Track GT", "Track GH"],
    #                            ["time", "position", "value"],
    #                            "position")
    # save_data(res, "../../data/ProRail/processed/processed_geometry.json")

    with open("../../data/ProRail/processed/processed_geometry.json") as f:
        data = json.load(f)

    calculate_settlement(data)
