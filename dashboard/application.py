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
    app_utils.runner(input_json)
    # add hash to calculations.json
    calcs.update({str(hash.hash_value): "path_to_geojson"})
    with open(os.path.join(CALCS_PATH, CALCS_JSON), "w") as fo:
        json.dump(calcs, fo, indent=2)
    return render_template("message.html", message="Calculation will run")


if __name__ == "__main__":
    app.run("127.0.0.1")
