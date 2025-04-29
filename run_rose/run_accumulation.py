import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rose.model.accumulation_model import AccumulationModel, Varandas, LiSelig, Nasrollahi


base_path = "erju/cargo"
output_folder = "erju/cargo/accumulation"

with open(os.path.join(base_path, f"erju.pickle"), "rb") as f:
    doubledekker = pickle.load(f)


total_time = [50, 365]  # days
idx = range(100-25, 100+25)  # indexes to collect data

# train info
train_info = {"dubbeldekker": {"forces": np.array(doubledekker['vertical_force_soil']),
                               "nb-per-hour": 16,
                               "nb-hours": 16,
                               "nb-axles": 1},
              }

sleeper_width = 0.25
sleeper_length = 3.5

with open(r"./run_rose/soil_layers.json", "r") as f:
    dat = json.load(f)

soil1 = dat["soil1"]  # stiff
soil2 = dat["soil2"]  # soft
soil2["soil_layers"]["top_level"][0] = 0

steps = 10
reload_s, reload_v, reload_k = False, False, False


# varandas model
# start_time = 0
# reload_v = False
model = Varandas()
set_varandas = AccumulationModel(accumulation_model=model)

# # sellig model
# sellig = LiSelig([soil1, soil2], doubledekker["soil_ID"], sleeper_width, sleeper_length, t_ini=0)
# set_sellig = AccumulationModel(accumulation_model=sellig)

# kourosh model
model = Nasrollahi(0.02, 2.8, 0.11)
set_kourosh = AccumulationModel(accumulation_model=model)



start_time = 0
for t in total_time:
    # set_sellig.read_traffic(train_info, start_time=start_time, end_time=t)
    # set_sellig.calculate_settlement(idx=idx, reload=reload_s)
    # set_sellig.write_results(os.path.join(output_folder, f"./LiSelig_time_{t}.pickle"))
    # reload_s = True

    # set_varandas.read_traffic(train_info, start_time=start_time, end_time=t)
    # set_varandas.calculate_settlement(idx=idx, reload=reload_v)
    # set_varandas.write_results(os.path.join(output_folder, f"./Varandas_time_{t}.pickle"))
    # reload_v = True

    set_kourosh.read_traffic(train_info,  start_time=start_time, end_time=t)
    set_kourosh.calculate_settlement(idx=idx, reload=reload_k)
    set_kourosh.write_results(os.path.join(output_folder, f"./Kourosh_time_{t}.pickle"))
    reload_k = True
    start_time = t



# with open(os.path.join(output_folder, f"LiSelig_time_50.pickle"), "rb") as f:
#     sellig = pickle.load(f)

with open(os.path.join(output_folder, f"Varandas_time_365.pickle"), "rb") as f:
    varandas = pickle.load(f)

with open(os.path.join(output_folder, f"Kourosh_time_365.pickle"), "rb") as f:
    kourosh = pickle.load(f)


fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# ax[0].plot(sellig["time"], sellig["displacement"][5], label="LiSelig")
ax[0].plot(varandas["time"], varandas["displacement"][5], label="Varandas")
ax[0].plot(kourosh["time"], kourosh["displacement"][5], label="Kourosh")

# ax[1].plot(sellig["time"], np.array(sellig["displacement"][15])*40, label="LiSelig")
ax[1].plot(varandas["time"], varandas["displacement"][15], label="Varandas")
ax[1].plot(kourosh["time"], kourosh["displacement"][15], label="Kourosh")

ax[0].grid()
ax[0].set_xlabel("Time [days]")
ax[0].set_ylabel("Settlement [m]")
ax[0].legend()

ax[1].grid()
ax[1].set_xlabel("Time [days]")
ax[1].set_ylabel("Settlement [m]")
ax[1].legend()

plt.savefig(os.path.join(output_folder, "aaa.png"))
plt.close()