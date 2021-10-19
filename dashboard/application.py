from flask import Flask, render_template, session, request
from flask_cors import CORS

from flask_session import Session
import os
import json
from threading import Thread

# ROSE packages
from dashboard import app_utils
from dashboard import validate_json, validate_ricardo_json
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

    # input json from Front End
    input_json = request.get_json()

    # # structure of input json:
    # {'SOS_Segment_Input': {},
    #  'Sensar_Input': {},
    #  'Rila_Input': {},
    #  'InfraMon_Input': {},
    # }

    # calculation dictionary
    calc = {
        "valid": False,
        "exist": False,
        "running": False,
        "data": {},  # input json file
        "message": ""
    }

    # check if calculation running
    status = is_running(input_json["SOS_Segment_Input"])
    if status:
        calc["running"] = True
        calc["message"] = "Calculation running"
        return calc

    # check input json
    status_sos_segment_input = validate_input(input_json["SOS_Segment_Input"])
    if not status_sos_segment_input:
        calc["message"] = "Input file not valid"

    # validate inframon input if it is given
    if input_json["InfraMon_Input"] is not None:
        status_inframon = validate_inframon(input_json["InfraMon_Input"])
        if not status_inframon:
            message = "InfraMon input is not valid"

            # add message to calc message
            if calc["message"] == "":
                calc["message"] = message
            else:
                calc["message"] = calc["message"] + "; " + message
    else:
        status_inframon = True

    #todo add validation of rila and sensar
    status = all([status_sos_segment_input, status_inframon])

    if not status:
        calc["valid"] = status
        return calc

    # check if calculation exists
    status, initial_json, loc = calculation_exist(input_json["SOS_Segment_Input"])
    if status:
        calc["message"] = "Calculation already exists"
        calc["exist"] = status
        calc["data"] = initial_json

        # assigns the json data to session
        with open(os.path.join(CALCS_PATH, loc, "data.json"), "r") as fi:
            session["data"] = json.load(fi)
        return calc

    # run calculation
    status, initial_json, loc = calculation_basic(input_json["SOS_Segment_Input"])
    calc["running"] = status
    calc["data"] = initial_json

    if not calc["running"]:
        # assigns the json data to session
        with open(os.path.join(CALCS_PATH, loc, "data.json"), "r") as fi:
            session["data"] = json.load(fi)
        return calc
    else:
        calc["message"] = "Calculation running"
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

    time = int(request.args.get("time_index"))
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

def validate_inframon(inframon_json):
    r"""
    Validates the inframon input json file

    @param inframon_json: inframon input json file
    """
    status = validate_ricardo_json.check_json(inframon_json)
    return status


def calculation_exist(input_json):
    r"""
    Checks if the input json file has previously been calculated

    @param input_json: input json file
    """

    # hash file, check if file exists & has results
    hash = hashing.Hash()
    hash.hash_dict(input_json)

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
        status = False
        data = {}
        location = []

    return status, data, location


def calculation_basic(input_json):
    r"""
    Caller for the calculation threading

    @param input_json: input json file
    @return: calculation running, data, location
    """
    Thread(target=calculation, args=(input_json,)).start()
    return True, {}, ""


def calculation(input_json):
    r"""
    Runs the input json file

    @param input_json: input json file
    @param location: location of the settings file
    @return: calculation running, data, location
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
    app.run("127.0.0.1", port=8080)
    # app.run("0.0.0.0")
