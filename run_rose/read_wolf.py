import os
import json

def read_wolf(file_names):
    wolf_dicts = []
    for idx,file_name in enumerate(file_names):
        base = os.path.basename(file_name)
        base_name = os.path.splitext(base)[0]

        if os.path.splitext(base)[1] == '.json':

            with open(file_name) as f:
                data = json.load(f)

            wolf_dict = {"name": base_name,
                         "omega": data["omega"][0],
                         "stiffness": data["stiffness"][0],
                         "damping": data["damping"][0]}
            wolf_dicts.append(wolf_dict)
    return wolf_dicts

if __name__ == "__main__":
    from os.path import isfile, join
    wolf_res_path = r'../rose/utils/dyn_stiffness'
    wolf_files = [os.path.join(wolf_res_path, f) for f in os.listdir(wolf_res_path) if isfile(join(wolf_res_path, f))]
    results = read_wolf(wolf_files)

    stiffnesses = [res['stiffness'] for res in results]
    dampings = [res['damping'] for res in results]
    max_stiffness, min_stiffness = max(stiffnesses), min(stiffnesses)
    max_damping, min_damping = max(dampings), min(dampings)
    pass

