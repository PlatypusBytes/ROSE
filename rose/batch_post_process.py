import pickle
import json
import os

def write_gis_csv(res_dict):
    header = f"x-coordinate; y-coordinate; segment; max_disp\n"
    lines = [header]
    for k,v in res_dict.items():
        for coordinate in v["coordinates"]:
            line = f"{coordinate[0]}; {coordinate[1]}; {k}; {v['w_disp']}\n "
            lines.append(line)

    with open('sos_disp_res.csv', 'w') as f:
        f.writelines(lines)

def calculate_weighted_disp(res_dict):

    for k,v in res_dict.items():
        for k2,v2 in v['scenarios'].items():
            v['w_disp'] += v2['min_disp'] * v2['probability']


def get_results(res_dir, sos_dir, sos_fn):

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

            if prev_segment_id != segment_id:
                res_dict[segment_id] = {}
                res_dict[segment_id]['coordinates'] = sos_data[segment_id][scenario_id]['coordinates']
                res_dict[segment_id]['w_disp'] = 0
                res_dict[segment_id]['scenarios']={}

            res_dict[segment_id]['scenarios'][scenario_id] = {}
            res_dict[segment_id]['scenarios'][scenario_id]['min_disp'] = min_disp
            res_dict[segment_id]['scenarios'][scenario_id]['probability'] = probability

            prev_segment_id = segment_id

    return res_dict


if __name__ == "__main__":

    res_dir = "batch_results"
    sos_dir = "SOS"
    sos_fn = "SOS.json"

    res_dict = get_results(res_dir, sos_dir, sos_fn)
    calculate_weighted_disp(res_dict)
    write_gis_csv(res_dict)
