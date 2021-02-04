from cumulative_models import Varandas
import numpy as np
import matplotlib.pylab as plt
import pickle


if __name__ == "__main__":

    with open(r"./tests/test_data/res_KDyn_Segment 1001_scenario 1_damping_70.pickle", "rb") as f:
        data = pickle.load(f)

    train_info = {"dubbeldekker": {"forces": data['vertical_force_soil'],
                                   "nb-per-hour": 6,
                                   "nb-hours": 5,
                                   "nb-axles": 4},
                  "sprinter": {"forces": np.array(data['vertical_force_soil']) / 2,
                               "nb-per-hour": 4,
                               "nb-hours": 12,
                               "nb-axles": 4},
                  }

    sett = Varandas.AccumulationModel()
    sett.read_traffic(train_info, 20)
    sett.settlement(idx=[100])
    sett.dump("./settlement.json")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(sett.cumulative_time, sett.displacement[0, :])
    ax.grid()
    ax.set_xlabel("Time [d]")
    ax.set_ylabel("Vertical displacement [m]")
    plt.show()
