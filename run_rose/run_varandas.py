import json
from rose.model import accumulation_model
import numpy as np
import matplotlib.pylab as plt
import pickle
from time import time

if __name__ == "__main__":

    t = time()
    with open(r"../tests/test_data/res_KDyn_Segment 1001_scenario 1_damping_70.pickle", "rb") as f:
        data = pickle.load(f)

    train_info = {"dubbeldekker": {"forces": np.array(data['vertical_force_soil']),
                                   "nb-per-hour": 6,
                                   "nb-hours": 6,
                                   "nb-axles": 16},
                  "sprinter": {"forces": np.array(data['vertical_force_soil']) / 2,
                               "nb-per-hour": 6,
                               "nb-hours": 16,
                               "nb-axles": 16},
                  }

    sett = accumulation_model.AccumulationModel()
    sett.read_traffic(train_info, 365)
    sett.settlement(idx=[100])
    print(time()-t)
    sett.dump("./res/settlement.pickle")

    print(time()-t)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(sett.cumulative_time, sett.displacement[0, :])
    ax.grid()
    ax.set_xlabel("Time [d]")
    ax.set_ylabel("Vertical displacement [m]")
    plt.show()
