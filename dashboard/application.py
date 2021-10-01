from flask import Flask, render_template, request
import json

import numpy as np

import app_utils

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

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
    # app.run("127.0.0.1")
    runner(r'D:\software_development\rose\run_rose\example_rose_input.json')