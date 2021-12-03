import pathlib
import pickle
import json
import pytest
from data_proc import fugro

import matplotlib.pyplot as plt
import numpy as np

class TestFugro:

    def test_calculate_d_values(self):
        with open("test_data/rila_data.pickle", "rb") as f:
            all_rila_data = pickle.load(f)

        rila_data = all_rila_data["data"][0]

        d1, d2,d3 = fugro.calculate_d_values(rila_data["heights"], rila_data["coordinates"])

        plt.plot(d1)
        plt.plot(d2)
        plt.plot(d3)
        plt.show()

    def test_plot_data_summary_on_sos(self):
        with open("test_data/rila_data.pickle", "rb") as f:
            all_rila_data = pickle.load(f)

        sos_fn = "test_data/SOS.json"
        with open(sos_fn, 'r') as f:
            sos_data = json.load(f)

        fugro.plot_data_summary_on_sos(all_rila_data, sos_data, "output")


    def test_convert_prorail_chainage_to_RD(self):

        chainage_fn = r"D:\software_development\rose\data\Fugro\Cul_Tricht_Deltares.csv"
        fugro.convert_prorail_chainage_to_RD(chainage_fn)

    def test_read_rtg(self):

        xls_fn = r"test_data\RTG_test_data.xlsx"
        pickle_fn = r"test_data\RTG_test_data.pickle"
        rtg_data1 = fugro.read_rtg(xls_fn, "xls")
        rtg_data2 = fugro.read_rtg(pickle_fn, "pickle")

        np.testing.assert_array_almost_equal(rtg_data1["h1l_data"], rtg_data2["h1l_data"])

        pathlib.Path(pickle_fn).unlink()

    def test_plot_data_colormesh(self):

        pickle_fn = r"test_data\rtg_data.pickle"
        rtg_data = fugro.read_rtg(pickle_fn, "pickle")

        fig = plt.figure(figsize=(20, 10))
        # fig, ax = plt.subplots(2,2,figsize=(6, 5))

        data_sets= [rtg_data["h1l_data"], rtg_data["h1r_data"] , rtg_data["h2l_data"], rtg_data["h2r_data"]]
        titles = ["D1 left", "D1 right", "D2 left", "D2 right"]
        k = 0
        position = 221
        for i in range(2):
            for j in range(2):

                fugro.plot_data_colormesh(rtg_data["dates"], rtg_data["prorail_chainage"], data_sets[k], titles[k], fig=fig, position=position)
                k += 1
                position += 1

        plt.show()


