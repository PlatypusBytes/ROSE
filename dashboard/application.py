from flask import Flask, render_template, session, request
from flask_cors import CORS

from flask_session import Session
import os
import json
import numpy as np

# ROSE packages
from dashboard import app_utils
from dashboard import validate_json
from dashboard import hashing
from dashboard import io_utils

app = Flask(__name__,
            static_url_path="", static_folder="templates", template_folder="templates"
            )
CORS(app)

# session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


# path for the local calculations
CALCS_PATH = "../dash_calculations"
CALCS_JSON = "calculations.json"
RUNNING_JSON = "tmp.json"

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
    input_json = request.get_json()

    # # structure of input json:
    # {'SOS_Segment_Input': {},
    #  'Sensar_Input': {},
    #  'Rila_Input': {},
    #  'InfraMon_Input': {},
    # }

    # check if calculation running
    status = is_running(input_json["SOS_Segment_Input"])
    if status:
        return "Calculation is already running."

    # calculation dictionary
    calc = {
        "valid": False,
        "exist": False,
        "data": {},  # input json file
    }

    # check input json
    status = validate_input(input_json["SOS_Segment_Input"])
    calc["valid"] = status

    # run calculation
    status, initial_json, loc = calculation(input_json["SOS_Segment_Input"])
    calc["exist"] = status
    calc["data"] = initial_json

    # assigns the json data to session
    with open(os.path.join(CALCS_PATH, loc, "data.json"), "r") as f:
        session["data"] = json.load(f)

    return calc


@app.route("/dynamic_stiffness")
def dynamic_stiffness():

    train_type = request.args.get("train_type")
    value_type = request.args.get("value_type")  # mean or std

    geojson = io_utils.parse_dynamic_stiffness_data(
        session.get("data"), train_type, value_type
    )

    return geojson


@app.route("/settlement")
def settlement():

    time = request.args.get("time_index")
    value_type = request.args.get("value_type")  # mean or std

    geojson = io_utils.parse_cumulative_settlement_data(
        session.get("data"), time, value_type
    )

    return geojson


@app.route("/graph_values")
def graph_values():
    segment_id = request.args.get("segment_id")

    geojson = io_utils.parse_graph_data(session.get("data"), segment_id)

    return geojson


def is_running(input_json):
    r"""
    Checks if the calculation is running

    @param input_json: input json file
    @return True or False
    """

    # hash file
    hash = hashing.Hash()
    hash.hash_dict(input_json)

    # if running (tmp) file exists
    if os.path.isfile(os.path.join(CALCS_PATH, RUNNING_JSON)):
        # open tmp file
        with open(os.path.join(CALCS_PATH, RUNNING_JSON), "r") as fi:
            running = json.load(fi)
        # check if hash is in tmp file
        if hash.hash_value in running.keys():
            return True
    else:
        # create empty json file
        with open(os.path.join(CALCS_PATH, RUNNING_JSON), "w") as fi:
            json.dump({}, fi)

    return False


def validate_input(input_json):
    r"""
    Validates the input json file

    @param input_json: input json file
    """

    # validates json input
    status = validate_json.check_json(input_json)

    return status


def calculation(input_json):
    r"""
    Runs the input json file

    @param input_json_file: input json file
    """

    # hash file, check if file exists & has results
    hash = hashing.Hash()
    hash.hash_dict(input_json)

    # add calculation to tmp file (as running)
    with open(os.path.join(CALCS_PATH, RUNNING_JSON), "w") as fi:
        json.dump({hash.hash_value: "Running"}, fi, indent=2)

    # load calculations.json
    with open(os.path.join(CALCS_PATH, CALCS_JSON), "r") as fi:
        calcs = json.load(fi)

    # if hash exists load results
    if hash.hash_value in calcs.keys():
        # location output json
        location = calcs[hash.hash_value]
        status = True
        # open json results and return
        with open(os.path.join(CALCS_PATH, location, "settings.json")) as fi:
            data = json.load(fi)
    else:
        # run calculation
        status, data = app_utils.runner(input_json, CALCS_PATH)

        # add hash to calculations.json
        calcs.update({str(hash.hash_value): input_json["project_name"]})
        # location of output json
        location = input_json["project_name"]
        with open(os.path.join(CALCS_PATH, CALCS_JSON), "w") as fo:
            json.dump(calcs, fo, indent=2)

    # clear hash from tmp file
    with open(os.path.join(CALCS_PATH, RUNNING_JSON), "r") as fi:
        running_tmp = json.load(fi)
    running_tmp.pop(hash.hash_value, None)
    # if after deleting has file is empty -> delete file
    if not running_tmp:
        os.remove(os.path.join(CALCS_PATH, RUNNING_JSON))
    # if the file is not empty is still computing -> delete local entry
    else:
        with open(os.path.join(CALCS_PATH, RUNNING_JSON), "w") as fi:
            json.dump(running_tmp, fi, indent=2)

    return True, data, location


if __name__ == "__main__":
    app.run("127.0.0.1")
