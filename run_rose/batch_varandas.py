from cumulative_models import Varandas
import numpy as np
import matplotlib.pylab as plt
import pickle
from pathlib import Path

import os
from os.path import isfile, join
from time import time

def main():

    cd = os.getcwd()
    inp_dir = os.path.join(cd, "batch_results","tmp")
    out_dir = os.path.join(cd, "batch_results","varandas")


    intercity_files = [os.path.join(inp_dir, f) for f in os.listdir(inp_dir) if isfile(join(inp_dir, f))
                       and f.endswith("intercity.pickle")]

    sprinter_files = [os.path.join(inp_dir, f) for f in os.listdir(inp_dir) if isfile(join(inp_dir, f))
                      and f.endswith("sprinter.pickle")]

    cargo_files = [os.path.join(inp_dir, f) for f in os.listdir(inp_dir) if isfile(join(inp_dir, f))
                      and f.endswith("cargo.pickle")]

    assert len(intercity_files) == len(sprinter_files)

    for intercity_file, sprinter_file, cargo_file in zip(intercity_files, sprinter_files, cargo_files):
        with open(intercity_file, "rb") as f:
            intercity_data = pickle.load(f)

        with open(sprinter_file, "rb") as f:
            sprinter_data = pickle.load(f)

        with open(cargo_file, "rb") as f:
            cargo_data = pickle.load(f)

    # for intercity_file in intercity_files:
    #     with open(intercity_file, "rb") as f:
    #         intercity_data = pickle.load(f)



        train_info = {"dubbeldekker": {"forces": intercity_data['vertical_force_soil'],
                                       "nb-per-hour": 6,
                                       "nb-hours": 6,
                                       "nb-axles": 16},
                      "sprinter": {"forces": sprinter_data['vertical_force_soil'],
                                   "nb-per-hour": 6,
                                   "nb-hours": 16,
                                   "nb-axles": 16},
                      "cargo": {"forces": cargo_data['vertical_force_soil'],
                                   "nb-per-hour": 27,
                                   "nb-hours": 1,
                                   "nb-axles": 10*4},
                      }

        try:
            t = time()
            out_name = Path(intercity_file).name.strip("intercity.pickle")

            sett = Varandas.AccumulationModel()
            sett.read_traffic(train_info, 100)
            sett.settlement(idx=[100])
            sett.dump(Path(out_dir,out_name + "_incl_cargo_100d.json"))

            # fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            print(time()-t)
            a=1+1
            # ax.plot(sett.cumulative_time, sett.displacement[0, :])
            # ax.grid()
            # ax.set_xlabel("Time [d]")
            # ax.set_ylabel("Vertical displacement [m]")
            # plt.save(Path(out_dir,out_name + "_incl_cargo_100d.png"))
            # plt.close()
        except:
            pass

if __name__ == "__main__":

    main()
    # with open(r"./tests/test_data/res_KDyn_Segment 1001_scenario 1_damping_70.pickle", "rb") as f:
    #     data = pickle.load(f)
    #
    # train_info = {"dubbeldekker": {"forces": data['vertical_force_soil'],
    #                                "nb-per-hour": 6,
    #                                "nb-hours": 6,
    #                                "nb-axles": 16},
    #               "sprinter": {"forces": np.array(data['vertical_force_soil']) / 2,
    #                            "nb-per-hour": 6,
    #                            "nb-hours": 16,
    #                            "nb-axles": 16},
    #               }
    #
    # sett = Varandas.AccumulationModel()
    # sett.read_traffic(train_info, 20)
    # sett.settlement(idx=[100])
    # sett.dump("./settlement.json")
    #
    # fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # ax.plot(sett.cumulative_time, sett.displacement[0, :])
    # ax.grid()
    # ax.set_xlabel("Time [d]")
    # ax.set_ylabel("Vertical displacement [m]")
    # plt.show()