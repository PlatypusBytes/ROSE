from flask import Flask, render_template, Response, request
import os
import json
import numpy as np
# ROSE packages
from dashboard import app_utils
from dashboard import validate_json
from dashboard import hashing

# app
app = Flask(__name__)

# path for the local calculations
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

    calc = {"valid": False,
            "exist": False,
            "sessionID": "",
            "data": {},  # input json file Aron
            }

    # check input json
    status = validate_input(input_json)
    calc["valid"] = status

    # run calculation
    status, dat = calculation(input_json)
    calc["exist"] = status
    calc["data"] = dat

    return calc

@app.route("/dynamic_stiffness")
def dynamic_stiffness():

    train_type = request.args.get('train_type')
    value_type = request.args.get('value_type')  # mean or std

    geojson = parse_dyn_data(geojson_template, train_type, value_type)

    return geojson


@app.route("/settlement")
def settlement():

    time = request.args.get('time')
    value_type = request.args.get('value_type')  # mean or std

    geojson = parse_set_data(geojson_template,  time, value_type)

    return geojson


@app.route("/graph_values")
def graph_values():
    segment_id = request.args.get('segment_id')

    geojson = parse_graph_data(geojson_template, segment_id)

    return geojson



def validate_input(input_json):
    r"""
    Validates the input json file

    @param input_json: input json file
    """

    # read json file
    with open(input_json, "r") as fi:
        input = json.load(fi)

    # validates json input
    status = validate_json.check_json(input)

    return status


def calculation(input_json):
    r"""
    Runs the input json file

    @param input_json: input json file
    """

    # read json file
    with open(input_json, "r") as fi:
        input = json.load(fi)

    # hash file, check if file exists & has results
    hash = hashing.Hash()
    hash.hash_dict(input)

    # load calculations.json
    with open(os.path.join(CALCS_PATH, CALCS_JSON), "r") as f:
        calcs = json.load(f)

    # if hash exists visualise results
    if hash.hash_value in calcs.keys():
        #ToDo: load results
        return True, {}  #geo_json  # needs to return input_json

    # ToDo: run calculation
    app_utils.runner(input_json)
    # add hash to calculations.json
    calcs.update({str(hash.hash_value): "path_to_geojson"})
    with open(os.path.join(CALCS_PATH, CALCS_JSON), "w") as fo:
        json.dump(calcs, fo, indent=2)
    return True, {}


if __name__ == "__main__":
    app.run("127.0.0.1")
