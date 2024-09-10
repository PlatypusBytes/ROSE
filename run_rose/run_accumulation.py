import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rose.model.accumulation_model import Varandas, LiSelig, AccumulationModel

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


total_time = [10, 25]  # days
idx = range(500-100, 500+100)  # indexes to collect data

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

with open(r"./run_rose/soil_layers.json", "r") as f:
    dat = json.load(f)

soil1 = dat["soil1"]  # stiff
soil2 = dat["soil2"]  # soft
soil2["soil_layers"]["top_level"][0] = 0

steps = 10
reload_s, reload_v = False, False


# varandas model
# start_time = 0
# reload_v = False
model = Varandas()
set_varandas = AccumulationModel(accumulation_model=model)

# sellig model
sellig = LiSelig([soil1, soil2], doubledekker["soil_ID"], sleeper_width, sleeper_length, t_ini=50)
set_sellig = AccumulationModel(accumulation_model=sellig)


start_time = 0
for t in total_time:
    set_sellig.read_traffic(train_info, start_time=start_time, end_time=t)
    set_sellig.calculate_settlement(idx=idx, reload=reload_s)
    set_sellig.write_results(os.path.join(output_folder, f"./LiSelig_time_{t}.pickle"))
    reload_s = True

    set_varandas.read_traffic(train_info, t, start_time=start_time)
    set_varandas.calculate_settlement(idx=idx, reload=reload_v)
    set_varandas.write_results(os.path.join(output_folder, f"./Varandas_time_{t}.pickle"))
    reload_v = True
    start_time = t

with open(os.path.join(output_folder, f"LiSelig_time_25.pickle"), "rb") as f:
    sellig = pickle.load(f)

with open(os.path.join(output_folder, f"Varandas_time_25.pickle"), "rb") as f:
    varandas = pickle.load(f)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(sellig["time"], sellig["displacement"][50], label="LiSelig")
ax[0].plot(varandas["time"], varandas["displacement"][50], label="Varandas")
ax[0].plot(varandas["time"], np.array(sellig["displacement"][50]) +
           np.array(varandas["displacement"][50]), label="Total")

ax[1].plot(sellig["time"], np.array(sellig["displacement"][150])*40, label="LiSelig")
ax[1].plot(varandas["time"], varandas["displacement"][150], label="Varandas")
ax[1].plot(varandas["time"], np.array(sellig["displacement"][150])*40 +
           np.array(varandas["displacement"][150]), label="Total")

ax[0].grid()
ax[0].set_xlabel("Time [days]")
ax[0].set_ylabel("Settlement [m]")
ax[0].legend()

ax[1].grid()
ax[1].set_xlabel("Time [days]")
ax[1].set_ylabel("Settlement [m]")
ax[1].legend()

plt.show()