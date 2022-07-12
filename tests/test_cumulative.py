# unit test for cumulative
import unittest
from rose.model import accumulation_model
import numpy as np
import os
import pickle


TEST_PATH = "tests"
tol = 1e-6


class TestVarandas(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0, 1, 1000)
        self.force1 = 41000 * np.sin(self.time * np.pi).reshape((1, 1000))
        self.force2 = 52000 * np.sin(self.time * np.pi).reshape((1, 1000))

        self.traininfo = {"dubbeldekker": {"forces": self.force1,
                                           "nb-per-hour": 13000,
                                           "nb-hours": 1,
                                           "nb-axles": 1},
                          "sprinter": {"forces": self.force2,
                                       "nb-per-hour": 22000,
                                       "nb-hours": 1,
                                       "nb-axles": 1},
                          }
        return

    def test_settlement_1(self):

        sett = accumulation_model.Varandas()
        sett.read_traffic(self.traininfo, 1)
        sett.settlement()
        sett.dump(os.path.join(TEST_PATH, "./example.pickle"))

        # compare with existing file
        with open(os.path.join(TEST_PATH, "./test_data/varandas_sin.pickle"), "rb") as f:
            data = pickle.load(f)

        res = compare_dics(sett.results, data)
        self.assertTrue(res)
        return

    def test_settlement_2(self):
        self.traininfo["dubbeldekker"]["forces"] = np.repeat(self.traininfo["dubbeldekker"]["forces"], 20, axis=0)
        self.traininfo["sprinter"]["forces"] = np.repeat(self.traininfo["sprinter"]["forces"], 20, axis=0)

        sett = accumulation_model.Varandas()
        sett.read_traffic(self.traininfo, 1)
        sett.settlement()
        sett.dump(os.path.join(TEST_PATH, "./example.pickle"))

        # compare with existing file
        with open(os.path.join(TEST_PATH, "./test_data/varandas_sin_full.pickle"), "rb") as f:
            data = pickle.load(f)

        res = compare_dics(sett.results, data)
        self.assertTrue(res)
        return

    def test_settlement_3(self):
        self.traininfo["dubbeldekker"]["forces"] = np.repeat(self.traininfo["dubbeldekker"]["forces"], 50, axis=0)
        self.traininfo["sprinter"]["forces"] = np.repeat(self.traininfo["sprinter"]["forces"], 50, axis=0)

        sett = accumulation_model.Varandas()
        sett.read_traffic(self.traininfo, 1)
        sett.settlement(idx=[40])
        sett.dump(os.path.join(TEST_PATH, "./example.pickle"))

        # compare with existing file
        with open(os.path.join(TEST_PATH, "./test_data/varandas_sin_full_node.pickle"), "rb") as f:
            data = pickle.load(f)

        res = compare_dics(sett.results, data)
        self.assertTrue(res)
        return

    def tearDown(self):
        # delete json file
        os.remove(os.path.join(TEST_PATH, "./example.pickle"))
        return


def compare_dics(dic1, dic2):
    result = []
    for key in dic1:
        print(key)

        if isinstance(dic1[key], list):
            for j in range(len(dic1[key])):
                if np.abs(dic1[key][j] - dic2[key][j]) < tol:
                    result.append(True)
                else:
                    result.append(False)
        elif isinstance(dic1[key], dict):
            for k in dic1[key]:
                for j in range(len(dic1[key][k])):
                    if np.abs(dic1[key][k][j] - dic2[key][k][j]) < tol:
                        result.append(True)
                    else:
                        result.append(False)

    if all(result):
        result = True
    else:
        result = False
    return result


if __name__ == "__main__":
    unittest.main()
