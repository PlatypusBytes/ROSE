import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rose.model.accumulation_model import AccumulationModel, Varandas, LiSelig, Nasrollahi, Sato, Shenton

base_path = "results_TZ"
output_folder = "./comparison_models"

with open(os.path.join(base_path, f"DOUBLEDEKKER.pickle"), "rb") as f:
    doubledekker = pickle.load(f)


total_time = [50, 365]  # days
idx = range(100-25, 100+25)  # indexes to collect data

# train info
train_info = {"dubbeldekker": {"forces": np.array(doubledekker['vertical_force_soil'])/2,
                               "nb-per-hour": 16,
                               "nb-hours": 16,
                               "nb-axles": 1},
            "dubbeldekker2": {"forces": np.array(doubledekker['vertical_force_soil']),
                               "nb-per-hour": 8,
                               "nb-hours": 8,
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
reload = False


# varandas model
model = Varandas()
set_varandas = AccumulationModel(accumulation_model=model, steps=steps)

# sellig model
sellig = LiSelig([soil1, soil2], doubledekker["soil_ID"], sleeper_width, sleeper_length, t_ini=1)
set_sellig = AccumulationModel(accumulation_model=sellig, steps=steps)

# kourosh model
model = Nasrollahi(0.02, 2.8, 0.11)
set_kourosh = AccumulationModel(accumulation_model=model, steps=steps)

# Sato model
model = Sato(1e-4, 1e-8, 0.005)
set_sato = AccumulationModel(accumulation_model=model, steps=steps)

# Shenton model
model = Shenton(1e-4, 1e-8)
set_shenton = AccumulationModel(accumulation_model=model, steps=steps)

start_time = 0
for t in total_time:
    set_sellig.read_traffic(train_info, start_time=start_time, end_time=t)
    set_sellig.calculate_settlement(idx=idx, reload=reload)
    set_sellig.write_results(os.path.join(output_folder, f"./LiSelig_time_{t}.pickle"))

    set_varandas.read_traffic(train_info, start_time=start_time, end_time=t)
    set_varandas.calculate_settlement(idx=idx, reload=reload)
    set_varandas.write_results(os.path.join(output_folder, f"./Varandas_time_{t}.pickle"))

    set_sato.read_traffic(train_info, start_time=start_time, end_time=t)
    set_sato.calculate_settlement(idx=idx, reload=reload)
    set_sato.write_results(os.path.join(output_folder, f"./Sato_time_{t}.pickle"))

    set_shenton.read_traffic(train_info, start_time=start_time, end_time=t)
    set_shenton.calculate_settlement(idx=idx, reload=reload)
    set_shenton.write_results(os.path.join(output_folder, f"./Shenton_time_{t}.pickle"))


    # set_kourosh.read_traffic(train_info,  start_time=start_time, end_time=t)
    # set_kourosh.calculate_settlement(idx=idx, reload=reload)
    # set_kourosh.write_results(os.path.join(output_folder, f"./Kourosh_time_{t}.pickle"))

    reload = True
    start_time = t


with open(os.path.join(output_folder, f"LiSelig_time_365.pickle"), "rb") as f:
    sellig = pickle.load(f)

with open(os.path.join(output_folder, f"Varandas_time_365.pickle"), "rb") as f:
    varandas = pickle.load(f)

# with open(os.path.join(output_folder, f"Kourosh_time_365.pickle"), "rb") as f:
#     kourosh = pickle.load(f)

with open(os.path.join(output_folder, f"Sato_time_365.pickle"), "rb") as f:
    sato = pickle.load(f)

with open(os.path.join(output_folder, f"Shenton_time_365.pickle"), "rb") as f:
    shenton = pickle.load(f)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
ax[0].plot(sellig["time"], np.array(sellig["displacement"][5])*1000, label="LiSelig")
ax[0].plot(varandas["time"], np.array(varandas["displacement"][5])*1000, label="Varandas")
ax[0].plot(sato["time"], np.array(sato["displacement"][5])*1000, label="Sato")
ax[0].plot(shenton["time"], np.array(shenton["displacement"][5])*1000, label="Shenton")
# ax[0].plot(kourosh["time"], kourosh["displacement"][5], label="Kourosh")

ax[1].plot(sellig["time"], np.array(sellig["displacement"][15])*1000, label="LiSelig")
ax[1].plot(varandas["time"], np.array(varandas["displacement"][15])*1000, label="Varandas")
ax[1].plot(sato["time"], np.array(sato["displacement"][15])*1000, label="Sato")
ax[1].plot(shenton["time"], np.array(shenton["displacement"][15])*1000, label="Shenton")
# ax[1].plot(kourosh["time"], kourosh["displacement"][15], label="Kourosh")

ax[0].grid()
ax[0].set_xlabel("Time [days]")
ax[0].set_ylabel("Settlement [mm]")
ax[0].legend()
ax[0].set_xlim(0, 365)
ax[0].set_ylim(0, 10)

ax[1].grid()
ax[1].set_xlabel("Time [days]")
ax[1].set_ylabel("Settlement [m]")
ax[1].legend()

plt.savefig(os.path.join(output_folder, "example.png"))
plt.close()