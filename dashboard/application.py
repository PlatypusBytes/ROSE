from flask import Flask, render_template, Response, request
from flask_cors import CORS
import os
import json
import numpy as np

# ROSE packages
from dashboard import app_utils
from dashboard import validate_input
from dashboard import hashing

# import app_utils
from rose.model import Varandas

app = Flask(
    __name__, static_url_path="", static_folder="templates", template_folder="templates"
)
CORS(app)

# poth for the local calculations
CALCS_PATH = "../dash_calculations"
CALCS_JSON = "calculations.json"
if not os.path.isdir(CALCS_PATH):
    os.makedirs(CALCS_PATH)
    # create json
    with open(os.path.join(CALCS_PATH, CALCS_JSON), "w") as f:
        json.dump({}, f)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def write_geo_json(features, filename):

    all_train_types = set()
    min_sett, max_sett = 1e10, -1e10
    min_stiff, max_stiff = 1e10, -1e10
    for feature in features:
        min_sett = min(
            min_sett, min(feature["properties"]["cumulative_settlement_mean"])
        )
        max_sett = max(
            max_sett, max(feature["properties"]["cumulative_settlement_mean"])
        )
        min_stiff = min(
            min_stiff, min(min(feature["properties"]["mean_dyn_stiffness"]))
        )
        max_stiff = max(
            max_stiff, max(max(feature["properties"]["mean_dyn_stiffness"]))
        )
        all_train_types.update(feature["properties"]["train_names"])

    all_train_types = list(all_train_types)
    cumulative_settlement_limits = np.linspace(min_sett, max_sett, 5)
    dyn_stiffness_limits = np.linspace(min_stiff, max_stiff, 5)

    geo_json = {
        "colours": ["#691aff", "#b81010", "#ffdb1a", "#6d1046", "#000066"],
        "cumulative_sett_limits": np.around(
            cumulative_settlement_limits, decimals=2
        ).tolist(),
        "dyn_stiff_limits": np.around(dyn_stiffness_limits, decimals=2).tolist(),
        "all_train_types": all_train_types,
        "geojson": {"type": "FeatureCollection", "features": features},
    }

    with open(filename, "w") as json_file:
        json.dump(geo_json, json_file, indent=2)


def add_feature_to_geo_json(
    coordinates,
    time,
    mean_dyn_stiffness,
    std_dyn_stiffness,
    train_names,
    mean_cum_settlement,
    std_cum_settlement,
):

    # convert coordinates to lat lon coordinates
    np_coords = np.array(coordinates)
    lat, lon = app_utils.transform_rd_to_lat_lon(np_coords[:, 0], np_coords[:, 1])
    new_coordinates = np.array([lat, lon]).T.tolist()

    # create feature dict
    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": new_coordinates},
        "properties": {
            "mean_dyn_stiffness": np.around(mean_dyn_stiffness, decimals=2).tolist(),
            "std_dyn_stiffness": np.around(std_dyn_stiffness, decimals=2).tolist(),
            "train_names": train_names,
            "time": np.around(time, decimals=2).tolist(),
            "cumulative_settlement_mean": np.around(
                mean_cum_settlement, decimals=2
            ).tolist(),
            "cumulative_settlement_std": np.around(
                std_cum_settlement, decimals=2
            ).tolist(),
        },
    }
    return feature


@app.route("/runner", methods=["POST"])
def run():

    input_json = request.get_json()
    # check input json & runs calculation
    message = calculation(input_json)
    return message


def calculation(input_json):
    r"""
    Reads and validates the input json file

    @param input_json: input json file
    """

    # validates json input
    status = validate_input.check_json(input_json)

    # ToDo: if file is not valid renders input not valid
    if not status:
        return render_template("message.html", message="Input file not valid")

    # hash file, check if file exists & has results
    hash = hashing.Hash()
    hash.hash_dict(input)

    # load calculations.json
    with open(os.path.join(CALCS_PATH, CALCS_JSON), "r") as f:
        calcs = json.load(f)

    # if hash exists visualise results
    if hash.hash_value in calcs.keys():
        # ToDo: load results
        return render_template(
            "message.html", message="Calculation exists. Load results"
        )

    # ToDo: run calculation
    # runner(input_json)
    # add hash to calculations.json
    calcs.update({str(hash.hash_value): "path_to_geojson"})
    with open(os.path.join(CALCS_PATH, CALCS_JSON), "w") as fo:
        json.dump(calcs, fo, indent=2)
    return render_template("message.html", message="Calculation will run")


def runner(json_input):

    from run_rose.run_wolf import create_layering_for_wolf, run_wolf_on_layering

    # retrieve data from input file
    with open(json_input, "r") as f:
        input_data = json.load(f)

    sos_data = input_data["sos_data"]
    traffic_data = input_data["traffic_data"]

    # set embankment data
    E = 100e6
    v = 0.2
    emb = ["embankment", E / (2 * (1 + v)), v, 2000, 0.05, 1]

    features = []
    # loop over segments
    for k, v in sos_data.items():
        # loop over trains
        mean_forces = []
        std_forces = []
        mean_dynamic_stiffnesses = []
        std_dynamic_stiffnesses = []

        train_types = []

        train_dicts = {}
        for train in traffic_data:
            train_types.append(train["type"])
            vertical_force_soil_segment = []
            dyn_stiffnesses = []
            # loop over scenarios
            for k2, v2 in v["scenarios"].items():

                # get wolf data
                layering = create_layering_for_wolf(v2["soil_layers"], emb)
                omega = 2 * np.pi * train["velocity"] / train["cart_length"]
                wolf_data = run_wolf_on_layering(layering, np.array([omega]))

                # determine dynamic soil stiffness and damping
                sleeper_dist = input_data["track_info"]["geometry"]["sleeper_distance"]
                dyn_stiffness = np.real(wolf_data.K_dyn) * sleeper_dist
                damping = np.imag(wolf_data.K_dyn) / omega * sleeper_dist
                soil = {"stiffness": dyn_stiffness, "damping": damping}

                dyn_stiffnesses.append(dyn_stiffness)

                # assign data to coupled model
                coupled_model = app_utils.assign_data_to_coupled_model(
                    train,
                    input_data["track_info"],
                    input_data["time_integration"],
                    soil,
                )

                # run coupled model
                coupled_model.main()

                # get results from coupled model
                (
                    time,
                    vertical_force_soil_scenario,
                ) = app_utils.get_results_coupled_model(coupled_model, 10)
                vertical_force_soil_segment.append(vertical_force_soil_scenario)

            train_dicts[train["type"]] = train["traffic"]

            # todo change with cumulative settlement

            # calculate mean and std of force of current train
            vertical_force_soil_segment = np.array(vertical_force_soil_segment)

            mean_force = np.mean(vertical_force_soil_segment, axis=0)
            std_force = np.std(vertical_force_soil_segment, axis=0)
            mean_forces.append(mean_force)
            std_forces.append(std_force)

            train_dicts[train["type"]]["forces"] = mean_force[None, :]

            mean_dynamic_stiffnesses.append(list(np.mean(dyn_stiffnesses, axis=0)))
            std_dynamic_stiffnesses.append(list(np.std(dyn_stiffnesses, axis=0)))
            break

        sett = Varandas.AccumulationModel()
        sett.read_traffic(train_dicts, 50)
        sett.settlement(idx=[0])

        # calculate output step size (1 outout value per day + last value
        n_steps = len(sett.results["time"])
        n_days = sett.results["time"][-1] - sett.results["time"][0]
        step_size = n_steps / n_days

        # get output indices
        indices = [int(day * step_size) for day in range(int(n_days))]
        # add last index
        indices.append(n_steps - 1)

        cumulative_time = np.array(sett.results["time"])[indices]
        cumulative_settlement_mean = (
            np.array(sett.results["settlement"]["0"])[indices] * 1000
        )  # in mm
        cumulative_settlement_std = cumulative_settlement_mean / 10

        # add feature to geo json
        feature = add_feature_to_geo_json(
            v["coordinates"],
            cumulative_time,
            mean_dynamic_stiffnesses,
            std_dynamic_stiffnesses,
            train_types,
            cumulative_settlement_mean,
            cumulative_settlement_std,
        )

        features.append(feature)
        break

    write_geo_json(features, "geojson_example.json")


if __name__ == "__main__":
    app.run("127.0.0.1")
