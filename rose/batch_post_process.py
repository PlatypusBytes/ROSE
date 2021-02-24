import pickle
import json
import os

import matplotlib.pyplot as plt
import numpy as np

def write_gis_csv(res_dict):
    header = f"x-coordinate; y-coordinate; segment; max_disp; stiffness; dyn_stiffness; cum_settlement\n"
    lines = [header]
    for k,v in res_dict.items():
        for coordinate in v["coordinates"]:
            line = f"{coordinate[0]}; {coordinate[1]}; {k}; {v['w_disp']}; {v['w_stiffness']}; {v['dyn_stiffness']}; {v['cum_settlement']}\n "
            lines.append(line)

    with open('sos_disp_res.csv', 'w') as f:
        f.writelines(lines)

def calculate_weighted_values(res_dict, key, res_key):
    """
    Calculates weighted values from segment and scenario dictionary

    :param res_dict: results dictonary
    :param key: value key
    :param res_key: result key
    :return:
    """

    for k,v in res_dict.items():
        v[res_key] = 0
        for k2, v2 in v['scenarios'].items():
            v[res_key] += v2[key] * v2['probability']

def plot_cumulative_results(file_name):
    """
    Plots vertical displacement over time after using a cumulative model
    :param file_name: cumulative result file
    :return:
    """

    with open(file_name, 'r') as f:
        sett = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(sett['time'], sett['settlement']['100'])
    ax.grid()
    ax.set_xlabel("Time [d]")
    ax.set_ylabel("Vertical displacement [m]")
    plt.show()

def calculate_dynamic_stiffness(dynamic_displacement, dynamic_force):
    return np.array(dynamic_force)/np.array(dynamic_displacement)

def get_batch_dynamic_stiffnesses(res_dir, sos_fn, node_nr, train_type):
    """
    Gets dynamic stiffnesses per SOS scenario and collects them in a dictionary
    :param res_dir: dynamic stiffness directory
    :param sos_fn: SOS filename
    :param node_nr: node number for the results
    :return:
    """
    with open(sos_fn, 'r') as f:
        sos_data = json.load(f)

    res_dict = {"time": [0],
                "coordinates": {},
                "data": {}}

    prev_segment_id = ""

    for file in os.listdir(res_dir):
        if file.endswith(train_type+".pickle"):

            with open(os.path.join(res_dir, file), 'rb') as f:
                res_numerical = pickle.load(f)


            max_dyn_stiffness = calculate_dynamic_stiffness(np.max(np.abs(res_numerical['vertical_displacements_soil'][node_nr])),
                                                                      np.max(np.abs(res_numerical['vertical_force_soil'][node_nr])))

            _, segment_id, scenario_id = res_numerical["name"].split("_")
            probability = sos_data[segment_id][scenario_id]['probability']/100

            if prev_segment_id != segment_id:
                res_dict["data"][segment_id] = {}
                res_dict["weighted_data"][segment_id] = np.array([0.0],)

            res_dict["data"][segment_id][scenario_id] = {"data": [max_dyn_stiffness],
                                                         "probability": probability}

            res_dict["weighted_data"][segment_id] += np.array(res_dict["data"][segment_id][scenario_id]["data"])*probability

            prev_segment_id = segment_id

    for segment,v in sos_data.items():
        if segment in res_dict["data"]:
            res_dict["coordinates"][segment] = list(v.values())[0]["coordinates"]

    return res_dict

def get_segment_and_scenario_from_fn(file_name):
    """
    Gets segment and scenario id from the file name
    :param file_name:
    :return:
    """
    str_parts = file_name.split("_")
    segment_id = scenario_id = ""

    for part in str_parts:
        if "segment" in part.lower():
            segment_id = part
        if "scenario" in part.lower():
            scenario_id = part
    return segment_id, scenario_id

def write_gis_csv_2(res_dict, data_type):
    header = f"x-coordinate; y-coordinate; segment; {data_type}\n"
    lines = [header]

    for (coord_seg_k, coord_seg_v), (data_seg_k, data_seg_v) in zip(res_dict["coordinates"].items(), res_dict["weighted_data"].items()):
        if coord_seg_k == data_seg_k:
            for coordinate in coord_seg_v:
                line = f"{coordinate[0]}; {coordinate[1]}; {coord_seg_k}; {data_seg_v[-1]}\n "
                lines.append(line)

    with open(f"{data_type}_res.csv", 'w') as f:
        f.writelines(lines)

def get_batch_cumulative_settlement(res_dir, sos_fn, node_nr, time_step=-1):
    """
    Gets cumulative settlement from batch output and stores it in a dictionary

    :param res_dir: cumulative settlement directory
    :param sos_fn: SOS file name
    :param node_nr: node number of the result
    :param time_step: time step index of result, default is the last step
    :return:
    """
    with open(sos_fn, 'r') as f:
        sos_data = json.load(f)

    res_dict = {"time": [time_step],
                "coordinates": {},
                "data": {},
                "weighted_data": {}}

    prev_segment_id = ""

    for file in os.listdir(res_dir):
        if file.endswith(".json"):

            with open(os.path.join(res_dir, file)) as f:
                res_numerical = json.load(f)

            settlement = res_numerical["settlement"][str(node_nr)][time_step]

            segment_id, scenario_id = get_segment_and_scenario_from_fn(file)

            probability = sos_data[segment_id][scenario_id]['probability']/100

            if prev_segment_id != segment_id:
                res_dict["data"][segment_id] = {}
                res_dict["weighted_data"][segment_id] = np.array([0.0],)

            res_dict["data"][segment_id][scenario_id] = {"data": [settlement],
                                                         "probability": probability}

            res_dict["weighted_data"][segment_id] += np.array(res_dict["data"][segment_id][scenario_id]["data"])*probability

            prev_segment_id = segment_id

    for segment,v in sos_data.items():
        if segment in res_dict["data"]:
            res_dict["coordinates"][segment] = list(v.values())[0]["coordinates"]

    return res_dict

def plot_max_disp_vs_velocity(res_dir, start_time_idx):

    max_disps = []
    velocities = []
    for file in os.listdir(res_dir):
        if file.endswith(".pickle"):

            with open(os.path.join(res_dir, file), 'rb') as f:
                res_numerical = pickle.load(f)

            disp = np.array(res_numerical['vertical_displacements_soil'])[:,start_time_idx:]

            # max_disps.append(max(np.abs(res_numerical['vertical_displacements_soil'][node_nr])))
            max_disps.append(np.max(np.abs(disp))*1000)
            velocities.append(res_numerical['velocity'][-1]*3.6)

    plt.plot(np.array(velocities)[:],np.array(max_disps)[:], 'o')

    plt.xlabel('Velocity [km/u]')
    plt.ylabel('Displacement [mm]')
    plt.show()



def get_results(res_dir, sos_dir, sos_fn, wolf_dir, cum_dir, node_nr):

    with open(os.path.join(sos_dir, sos_fn), 'r') as f:
        sos_data = json.load(f)

    prev_segment_id = ""

    res_dict = {}
    for file in os.listdir(res_dir):
        if file.endswith(".pickle"):

            with open(os.path.join(res_dir, file), 'rb') as f:
                res_numerical = pickle.load(f)

            min_disp = min(res_numerical['vertical_displacements_soil'][node_nr])
            _, segment_id, scenario_id = res_numerical["name"].split("_")
            probability = sos_data[segment_id][scenario_id]['probability']/100

            with open(os.path.join(wolf_dir, res_numerical["name"]+'.json'), 'r') as f:
                wolf_data = json.load(f)
                stiffness = wolf_data["stiffness"][0]

            if prev_segment_id != segment_id:
                res_dict[segment_id] = {}
                res_dict[segment_id]['coordinates'] = sos_data[segment_id][scenario_id]['coordinates']
                res_dict[segment_id]['w_disp'] = 0
                res_dict[segment_id]['w_stiffness'] = 0
                res_dict[segment_id]['scenarios'] = {}

            res_dict[segment_id]['scenarios'][scenario_id] = {}
            res_dict[segment_id]['scenarios'][scenario_id]['min_disp'] = min_disp
            res_dict[segment_id]['scenarios'][scenario_id]['probability'] = probability
            res_dict[segment_id]['scenarios'][scenario_id]['stiffness'] = stiffness

            prev_segment_id = segment_id

    return res_dict


if __name__ == "__main__":

    res_dir = "batch_results/intercity"
    cum_dir = "batch_results/varandas"
    # res_dir = r"D:\software_development\ROSE\rose\batch_results\velocity_no_damping"
    sos_dir = "SOS"
    sos_fn = "SOS.json"

    wolf_dir = r"wolf/dyn_stiffness"

    node_nr = 100
    start_time_idx = 1000

    data_type = "dynamic_stiffness_soil"

    # res_dict = get_results(res_dir, sos_dir, sos_fn, wolf_dir,cum_dir,node_nr)

    # for k,v in res_dict.items():
    #     for k2,v2 in v['scenarios'].items():
    #         v['w_disp'] += v2['min_disp'] * v2['probability']
    #         v['w_stiffness'] += v2['stiffness'] * v2['probability']

    # calculate_weighted_values(res_dict,'min_disp', 'w_disp')
    # calculate_weighted_values(res_dict,'stiffness', 'w_stiffness')
    # # write_gis_csv(res_dict)
    #
    # plot_max_disp_vs_velocity(res_dir, start_time_idx)

    res_dict = get_batch_cumulative_settlement(cum_dir, os.path.join(sos_dir,sos_fn), node_nr)
    # write_gis_csv_cum_settlement(res_dict)
    write_gis_csv_2(res_dict, "cum_settlement")
    #
    # # res_dict = get_batch_dynamic_stiffnesses(res_dir, os.path.join(sos_dir,sos_fn), node_nr, "intercity")
    #
    # with open(r"batch_results/intercity/cumulative_settlement_profile.json", "w") as f:
    #     json.dump(res_dict, f)

    # fn = r'D:\software_development\ROSE\rose\batch_results\varandas\s_Kdyn_Segment 1001_scenario 2__incl_cargo_100d.json'
    #
    # plot_cumulative_results(fn)
    #
    # fn2 = r'D:\software_development\ROSE\rose\batch_results\varandas\s_Kdyn_Segment 1001_scenario 2__100d.json'
    #
    # plot_cumulative_results(fn2)