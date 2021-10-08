import numpy as np
import json

ALL_RESULTS = {}

TEMPLATE_SETTLEMENT_FEATURE = {"geometry": {"type": "LineString",
                                            "coordinates": None},
                               "properties": {"segmentId": None,
                                              "value": None}}

def parse_dynamic_stiffness_data(train_type, value_type):
    features = []
    for segment_id, segment_data in ALL_RESULTS.items():

        train_index = segment_data["properties"]["train_names"].index(train_type)

        feature = {"geometry": {"type": "LineString",
                                "coordinates": segment_data["geometry"]["coordinates"]},
                   "properties": {"segmentId": segment_id,
                                  "value": segment_data["properties"][value_type][train_index]}}

        features.append(feature)

    geojson = {"type": "FeatureCollection",
               "features": features}
    return geojson

def parse_cumulative_settlement_data(time_index, value_type):
    features = []
    for segment_id, segment_data in ALL_RESULTS.items():
        feature = {"geometry": {"type": "LineString",
                                "coordinates": segment_data["geometry"]["coordinates"]},
                   "properties": {"segmentId": segment_id,
                                  "value": segment_data["properties"][value_type][time_index]}}

        features.append(feature)

    geojson = {"type": "FeatureCollection",
               "features": features}

    return geojson

def parse_graph_data(geojson_template, segment_id, value_type):

    geojson_template["time"] = ALL_RESULTS[segment_id]["properties"]["time"]
    geojson_template[value_type] = ALL_RESULTS[segment_id]["properties"]["value_type"]
    return geojson_template

def write_all_results(features_dict, filename: str):
    """
    Creates and writes geojson dict to json file
    :param features: all features in the results
    :param filename: output filename
    :return:
    """

    # write results dict
    with open(filename, "w") as json_file:
        json.dump(features_dict, json_file, indent=2)