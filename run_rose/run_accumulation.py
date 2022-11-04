import os
import json
import pickle
import numpy as np
from copy import deepcopy
from rose.model.accumulation_model import Varandas, LiSelig

base_path = "./results_TZ"
output_folder = "./results_TZ/passengers"


# load transition zones results for all trains
with open(os.path.join(base_path, f"DOUBLEDEKKER.pickle"), "rb") as f:
    doubledekker = pickle.load(f)
# with open(os.path.join(base_path, f"SPRINTER_SLT.pickle"), "rb") as f:
#     sprinter_stl = pickle.load(f)
# with open(os.path.join(base_path, f"SPRINTER_SGM.pickle"), "rb") as f:
#     sprinter_sgm = pickle.load(f)
# with open(os.path.join(base_path, f"CARGO_TAPPS.pickle"), "rb") as f:
#     tapps = pickle.load(f)
# with open(os.path.join(base_path, f"TRAXX.pickle"), "rb") as f:
#     traxx = pickle.load(f)
# with open(os.path.join(base_path, f"BR189.pickle"), "rb") as f:
#     BR189 = pickle.load(f)


total_time = [5, 10, 15]  # days
# total_time = [365]  # days
idx = range(500-333, 500+333)  # indexes to collect data

# train info
train_info = {"dubbeldekker": {"forces": np.array(doubledekker['vertical_force_soil']),
                               "nb-per-hour": 6,
                               "nb-hours": 16,
                               "nb-axles": 16},
              # "sprinter_stl": {"forces": np.array(sprinter_stl['vertical_force_soil']),
              #                  "nb-per-hour": 3,
              #                  "nb-hours": 16,
              #                  "nb-axles": 16},
              # "sprinter_sgm": {"forces": np.array(sprinter_sgm['vertical_force_soil']),
              #                  "nb-per-hour": 1,
              #                  "nb-hours": 6,
              #                  "nb-axles": 16},
              # "tapps": {"forces": np.array(tapps['vertical_force_soil']),
              #           "nb-per-hour": 1,
              #           "nb-hours": 1,
              #           "nb-axles": 24},
              # "traxx": {"forces": np.array(traxx['vertical_force_soil']),
              #           "nb-per-hour": 1,
              #           "nb-hours": 2,
              #           "nb-axles": 24},
              # "BR189": {"forces": np.array(BR189['vertical_force_soil']),
              #           "nb-per-hour": 3,
              #           "nb-hours": 1,
              #           "nb-axles": 24},
              }

sleeper_width = 0.25
sleeper_length = 3.5

with open(r"../data_proc/SOS.json", "r") as f:
    dat = json.load(f)

soil1 = dat["Segment 1077"]["scenario 2"]  # stiff
soil2 = dat["Segment 1079"]["scenario 4"]  # soft
soil2["soil_layers"]["top_level"][0] = 0

steps = 10
reload_s, reload_v = False, False
for t in total_time:
    sellig = LiSelig(t_ini=50, steps=steps, reload=reload_s)
    sellig.read_traffic(train_info, t)
    sellig.read_SoS([soil1, soil2], doubledekker["soil_ID"])  # stiff to soft
    sellig.calculate(sleeper_width, sleeper_length, idx=idx)
    sellig.dump(os.path.join(output_folder, f"./LiSelig_time_{t}.pickle"))
    reload_s = deepcopy(sellig)

    sett = Varandas(steps=steps, reload=reload_v)
    sett.read_traffic(train_info, t)
    sett.settlement(idx=idx)
    sett.dump(os.path.join(output_folder, f"./Varandas_time_{t}.pickle"))
    reload_v = deepcopy(sett)
