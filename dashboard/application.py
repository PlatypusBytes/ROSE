from flask import Flask, render_template, session, request
from flask_session import Session
import os
import json
import numpy as np
# ROSE packages
from dashboard import app_utils
from dashboard import validate_json
from dashboard import hashing
from dashboard import io_utils

# app
app = Flask(__name__)

# session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


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
    # create session data
    if "data" not in session:
        session["data"] = []
    return render_template("index.html")


@app.route("/runner", methods=["GET", "POST"])
def run():

    # ToDo: parse input json from Front End
    input_json = "../run_rose/example_rose_input.json"

    calc = {"valid": False,
            "exist": False,
            "sessionID": "",
            "data": {},  # input json file Aron first call
            }

    # check input json
    status = validate_input(input_json)
    calc["valid"] = status

    # run calculation
    status, initial_json, loc = calculation(input_json)
    calc["exist"] = status
    calc["data"] = initial_json

    # assigns the json data to session
    with open(os.path.join(CALCS_PATH, loc, "data.json"), "r") as f:
        session["data"] = json.load(f)

    return calc


@app.route("/dynamic_stiffness")
def dynamic_stiffness():

    train_type = request.args.get('train_type')
    value_type = request.args.get('value_type')  # mean or std

    geojson = io_utils.parse_dynamic_stiffness_data(session.get("data"), train_type, value_type)

    return geojson


@app.route("/settlement")
def settlement():

    time = request.args.get('time_index')
    value_type = request.args.get('value_type')  # mean or std

    geojson = io_utils.parse_cumulative_settlement_data(session.get("data"), time, value_type)

    return geojson


@app.route("/graph_values")
def graph_values():
    segment_id = request.args.get('segment_id')

    geojson = io_utils.parse_graph_data(session.get("data"), segment_id)

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

    # if hash exists load results
    if hash.hash_value in calcs.keys():
        # location output json
        location = calcs[hash.hash_value]
        status = True
        # open json results and return
        with open(os.path.join(CALCS_PATH, location, "settings.json")) as f:
            data = json.load(f)

    else:
        # run calculation
        status, data = app_utils.runner(input_json, CALCS_PATH)

        # add hash to calculations.json
        calcs.update({str(hash.hash_value): input["project_name"]})
        # location of output json
        location = input["project_name"]
        with open(os.path.join(CALCS_PATH, CALCS_JSON), "w") as fo:
            json.dump(calcs, fo, indent=2)

    return True, data, location


if __name__ == "__main__":
    app.run("127.0.0.1")
