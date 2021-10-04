from flask import Flask, render_template, Response
import os
import json
import numpy as np
# ROSE packages
from dashboard import app_utils
from dashboard import validate_input
from dashboard import hashing

# app
app = Flask(__name__)

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


@app.route("/runner", methods=["POST"])
def run():

    # ToDo: parse input json from Front End
    input_json = "../run_rose/example_rose_input.json"

    # check input json & runs calculation
    message = calculation(input_json)

    return message


def calculation(input_json):
    r"""
    Reads and validates the input json file

    @param input_json: input json file
    """

    # read json file
    with open(input_json, "r") as fi:
        input = json.load(fi)

    # validates json input
    status = validate_input.check_json(input)

    # ToDo: if file is not valid renders input not valid
    if not status:
        return render_template("message.html",  message="Input file not valid")

    # hash file, check if file exists & has results
    hash = hashing.Hash()
    hash.hash_dict(input)

    # load calculations.json
    with open(os.path.join(CALCS_PATH, CALCS_JSON), "r") as f:
        calcs = json.load(f)

    # if hash exists visualise results
    if hash.hash_value in calcs.keys():
        #ToDo: load results
        return render_template("message.html", message="Calculation exists. Load results")

    # ToDo: run calculation
    # runner(input_json)
    # add hash to calculations.json
    calcs.update({str(hash.hash_value): "path_to_geojson"})
    with open(os.path.join(CALCS_PATH, CALCS_JSON), "w") as fo:
        json.dump(calcs, fo, indent=2)
    return render_template("message.html", message="Calculation will run")


def write_geo_json():
    geo_json = {"type": "FeatureCollection",
                "features": [
                    {"type": "Feature",
                     "geometry": {
                         "type": "LineString",
                         "coordinates": [np.nan]
                     },
                     "properties":{
                         "dynamic_stiffness_mean": [np.nan],
                         "dynamic_stiffness_std": [np.nan],
                         "cumulative_settlement_mean":[np.nan],
                         "cumulative_settlement_std":[np.nan]
                     }}
                ]}


def add_to_geo_json(coordinates, data: np.ndarray):
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)

    pass

def runner(json_input):

    from run_rose.run_wolf import create_layering_for_wolf, run_wolf_on_layering

    with open(json_input,'r') as f:
        input_data = json.load(f)

    sos_data = input_data["sos_data"]
    traffic_data =input_data["traffic_data"]

    E = 100e6
    v = 0.2
    emb = ["embankment", E / (2 * (1 + v)), v, 2000, 0.05, 1]

    for train in traffic_data:
        for k, v in sos_data.items():
            vertical_force_soil_segment = []
            for k2, v2 in v["scenarios"].items():
                layering = create_layering_for_wolf(v2["soil_layers"], emb)
                omega = 2*np.pi * train["velocity"]/ train["cart_length"]
                wolf_data = run_wolf_on_layering(layering,np.array([omega]))

                sleeper_dist = input_data["track_info"]["geometry"]["sleeper_distance"]
                dyn_stiffness = np.real(wolf_data.K_dyn) * sleeper_dist
                damping = np.imag(wolf_data.K_dyn) / omega * sleeper_dist
                soil = {"stiffness": dyn_stiffness,
                        "damping": damping}

                coupled_model = app_utils.assign_data_to_coupled_model(train,input_data["track_info"], input_data["time_integration"], soil)

                coupled_model.main()

                _, vertical_force_soil_scenario, _ = app_utils.get_results_coupled_model(coupled_model, 10)
                vertical_force_soil_segment.append(vertical_force_soil_scenario)
                a=1+1

            vertical_force_soil_segment = np.array(vertical_force_soil_segment)
            add_to_geo_json(vertical_force_soil_segment)


if __name__ == "__main__":
    app.run("127.0.0.1")
