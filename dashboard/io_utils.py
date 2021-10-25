import numpy as np
import json

# ALL_RESULTS = {}

TEMPLATE_SETTLEMENT_FEATURE = {"geometry": {"type": "LineString",
                                            "coordinates": None},
                               "properties": {"segmentId": None,
                                              "value": None}}

def parse_dynamic_stiffness_data(ALL_RESULTS, train_type, value_type):
    features = []
    for segment_id, segment_data in ALL_RESULTS.items():

        train_index = segment_data["properties"]["train_names"].index(train_type)

        feature = {"type": "Feature",
                   "geometry": {"type": "LineString",
                                "coordinates": segment_data["geometry"]["coordinates"]},
                   "properties": {"segmentId": segment_id,
                                  "value": segment_data["properties"][value_type][train_index]}}

        features.append(feature)

    geojson = {"type": "FeatureCollection",
               "features": features}
    return geojson

def parse_cumulative_settlement_data(ALL_RESULTS, time_index, value_type):
    features = []
    for segment_id, segment_data in ALL_RESULTS.items():
        feature = {"type": "Feature",
                   "geometry": {"type": "LineString",
                                "coordinates": segment_data["geometry"]["coordinates"]},
                   "properties": {"segmentId": segment_id,
                                  "value": segment_data["properties"][value_type][time_index]}}

        features.append(feature)

    geojson = {"type": "FeatureCollection",
               "features": features}

    return geojson

def parse_graph_data(ALL_RESULTS, segment_id):

    graph_json = {"time": ALL_RESULTS[segment_id]["properties"]["time"],
                  "cumulative_settlement_mean": ALL_RESULTS[segment_id]["properties"]["cumulative_settlement_mean"],
                  "cumulative_settlement_std": ALL_RESULTS[segment_id]["properties"]["cumulative_settlement_std"]}

    return graph_json

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