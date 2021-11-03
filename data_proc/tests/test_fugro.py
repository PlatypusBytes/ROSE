import pickle
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
