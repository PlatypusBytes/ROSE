import pickle
import json
import os

def write_gis_csv(res_dict):
    header = f"x-coordinate; y-coordinate; segment; max_disp; stiffness\n"
    lines = [header]
    for k,v in res_dict.items():
        for coordinate in v["coordinates"]:
            line = f"{coordinate[0]}; {coordinate[1]}; {k}; {v['w_disp']}; {v['w_stiffness']}\n "
            lines.append(line)

    with open('sos_disp_res.csv', 'w') as f:
        f.writelines(lines)

def calculate_weighted_disp(res_dict):

    for k,v in res_dict.items():
        for k2,v2 in v['scenarios'].items():
            v['w_disp'] += v2['min_disp'] * v2['probability']
            v['w_stiffness'] += v2['stiffness'] * v2['probability']


def get_results(res_dir, sos_dir, sos_fn, wolf_dir):

    with open(os.path.join(sos_dir, sos_fn), 'r') as f:
        sos_data = json.load(f)

    prev_segment_id = ""

    res_dict = {}
    for file in os.listdir(res_dir):
        if file.endswith(".pickle"):

            with open(os.path.join(res_dir, file), 'rb') as f:
                res_numerical = pickle.load(f)

            min_disp = min(res_numerical['vertical_displacements_soil'][200])
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

    res_dir = "batch_results"
    sos_dir = "SOS"
    sos_fn = "SOS.json"

    wolf_dir = r"wolf/dyn_stiffness"

    res_dict = get_results(res_dir, sos_dir, sos_fn, wolf_dir)
    calculate_weighted_disp(res_dict)
    write_gis_csv(res_dict)
