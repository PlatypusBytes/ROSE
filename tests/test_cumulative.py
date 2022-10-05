# unit test for cumulative
import unittest
from rose.model.accumulation_model import Varandas, LiSelig
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

        sett = Varandas()
        sett.read_traffic(self.traininfo, 1)
        sett.settlement()
        sett.dump(os.path.join(TEST_PATH, "./example.pickle"))

        # compare with existing file
        with open(os.path.join(TEST_PATH, "./test_data/varandas_sin.pickle"), "rb") as f:
            data = pickle.load(f)

        res = compare_dics(sett.results.__dict__, data.__dict__)
        self.assertTrue(res)
        return

    def test_settlement_2(self):
        self.traininfo["dubbeldekker"]["forces"] = np.repeat(self.traininfo["dubbeldekker"]["forces"], 20, axis=0)
        self.traininfo["sprinter"]["forces"] = np.repeat(self.traininfo["sprinter"]["forces"], 20, axis=0)

        sett = Varandas()
        sett.read_traffic(self.traininfo, 1)
        sett.settlement()
        sett.dump(os.path.join(TEST_PATH, "./example.pickle"))

        # compare with existing file
        with open(os.path.join(TEST_PATH, "./test_data/varandas_sin_full.pickle"), "rb") as f:
            data = pickle.load(f)

        res = compare_dics(sett.results.__dict__, data.__dict__)
        self.assertTrue(res)
        return

    def test_settlement_3(self):
        self.traininfo["dubbeldekker"]["forces"] = np.repeat(self.traininfo["dubbeldekker"]["forces"], 50, axis=0)
        self.traininfo["sprinter"]["forces"] = np.repeat(self.traininfo["sprinter"]["forces"], 50, axis=0)

        sett = Varandas()
        sett.read_traffic(self.traininfo, 1)
        sett.settlement(idx=[40])
        sett.dump(os.path.join(TEST_PATH, "./example.pickle"))

        # compare with existing file
        with open(os.path.join(TEST_PATH, "./test_data/varandas_sin_full_node.pickle"), "rb") as f:
            data = pickle.load(f)

        res = compare_dics(sett.results.__dict__, data.__dict__)
        self.assertTrue(res)
        return

    def tearDown(self):
        # delete json file
        os.remove(os.path.join(TEST_PATH, "./example.pickle"))
        return


class TestLiSelig(unittest.TestCase):
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

        self.soil1 ={'probability': 50.0, 'soil_layers': {'soil_name': ['H_Aa_ht', 'H_Ro_z&k', 'H_Rg_zm', 'P_Rg_zm'], 'top_level': [2.5, 1.0, -0.5, -6.0], 'damping': [0.05, 0.05, 0.05, 0.05], 'gamma_dry': [17.0, 18.0, 17.0, 17.0], 'gamma_wet': [19.0, 18.0, 19.0, 19.0], 'a': [0.001, 0.0046, 0.001, 0.001], 'b': [0.004, 0.0473, 0.004, 0.004], 'c': [0.0001, 0.0015, 0.0001, 0.0001], 'cohesion': [0.0, 0.0, 0.0, 0.0], 'poisson': [0.25, 0.4, 0.25, 0.25], 'm': ['NaN', 0.8, 'NaN', 'NaN'], 'Su': ['NaN', 40.0, 'NaN', 'NaN'], 'POP': ['NaN', 8.0, 'NaN', 'NaN'], 'Young_modulus': [103.3, 98.8, 94.3, 118.3], 'friction_angle': [32.5, 27.5, 30.0, 30.0], 'formation': ['NaN', 'Echteld', 'Echteld', 'Kreftenheye'], 'shear_modulus': [41.3, 35.3, 37.7, 47.3]}, 'coordinates': [[132624.8664790446, 459118.1975268294], [132567.681000006, 459172.4830000017], [132495.66700000237, 459242.2729999986], [132424.68100000356, 459312.6020000153], [132353.2759999997, 459384.92299999844], [132283.28600000177, 459455.2400000034], [132211.66200000458, 459525.540000002], [132139.49700000416, 459594.5520000001], [132104.12158360792, 459627.9323876054]]}
        self.soil2 = {'probability': 17.15, 'soil_layers': {'soil_name': ['H_Aa_ht', 'H_Rk_ko', 'H_Vhv_v', 'P_Wdz_zf', 'P_Rg_zf', 'P_Rg_zg'], 'top_level': [0.5, -1.0, -1.5, -6.5, -7.0, -10.0], 'damping': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05], 'gamma_dry': [17.0, 13.0, 10.0, 17.0, 17.0, 18.0], 'gamma_wet': [19.0, 13.0, 10.0, 19.0, 19.0, 20.0], 'a': [0.001, 0.0145, 0.0364, 0.001, 0.001, 0.001], 'b': [0.004, 0.189, 0.3335, 0.004, 0.004, 0.004], 'c': [0.0001, 0.0103, 0.0245, 0.0001, 0.0001, 0.0001], 'cohesion': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'poisson': [0.25, 0.45, 0.45, 0.25, 0.25, 0.25], 'm': ['NaN', 0.8, 0.8, 'NaN', 'NaN', 'NaN'], 'Su': ['NaN', 10.0, 10.0, 'NaN', 'NaN', 'NaN'], 'POP': ['NaN', 15.0, 1.0, 'NaN', 'NaN', 'NaN'], 'Young_modulus': [103.3, 35.7, 7.8, 244.3, 169.3, 100.5], 'friction_angle': [32.5, 15.0, 15.0, 30.0, 30.0, 32.5], 'formation': ['NaN', 'Echteld', 'Nieuwkoop', 'Boxtel', 'Kreftenheye', 'Kreftenheye'], 'shear_modulus': [41.3, 12.3, 2.7, 97.7, 67.7, 40.2]}, 'coordinates': [[130767.76299999819, 460903.6770000044], [130695.38699999973, 460973.0450000067], [130622.37999999814, 461041.7010000041], [130550.45200000235, 461112.12300001393], [130478.55700000132, 461181.65200000233], [130407.00599999845, 461250.5530000042], [130333.91199999914, 461319.68299999856], [130261.37300000092, 461388.03200000146], [130188.42500000136, 461456.6170000084], [130116.11499999781, 461526.74299999594], [130044.26900000208, 461595.13700000977], [129971.58499999714, 461663.5380000041], [129898.5630000037, 461732.2910000008], [129826.1810000057, 461800.38899999845], [129753.32899999926, 461868.96400000114], [129680.82100000037, 461937.18100001453], [129608.41699999926, 462005.3900000092], [129537.18900000013, 462072.424999994], [129463.17499999951, 462142.0399999967], [129390.61100000035, 462210.36400000023], [129317.93999999958, 462278.7440000052], [129245.28200000591, 462347.13299999473], [129172.67100000178, 462414.7560000158], [129098.3330000026, 462481.5210000115], [129021.67900000449, 462546.5620000034], [128943.74200000227, 462610.22200001555], [128867.48699999944, 462671.3820000115], [128789.02600000426, 462733.32700001507], [128710.81700000464, 462795.99500001216], [128633.32500000327, 462861.57099999965], [128558.47300000419, 462930.7580000027], [128486.57899999917, 463003.4650000122], [128418.69999999886, 463078.7290000011], [128354.60900000452, 463156.5730000082], [128293.98900000456, 463237.2300000029], [128236.77400000553, 463321.16200000735], [128183.69700000204, 463407.1220000066], [128135.82800000597, 463495.2220000027], [128092.0740000045, 463584.9330000117], [128052.19700000298, 463676.8760000037], [128016.35600000335, 463770.5540000182], [127984.73600000315, 463865.4000000003], [127957.09800000352, 463962.60999999964]]}
        return

    def test_settlement_1(self):

        sett = LiSelig()
        sett.read_traffic(self.traininfo, 1)
        sett.read_SoS([self.soil1], [0])
        sett.calculate(0.25, 3.5)
        sett.dump(os.path.join(TEST_PATH, "./example.pickle"))

        # compare with existing file
        with open(os.path.join(TEST_PATH, "./test_data/li_selig_1.pickle"), "rb") as f:
            data = pickle.load(f)

        res = compare_dics(sett.results.__dict__, data.__dict__)
        self.assertTrue(res)
        return

    def test_settlement_2(self):

        self.traininfo["dubbeldekker"]["forces"] = np.repeat(self.traininfo["dubbeldekker"]["forces"], 20, axis=0)
        self.traininfo["sprinter"]["forces"] = np.repeat(self.traininfo["sprinter"]["forces"], 20, axis=0)

        sett = LiSelig()
        sett.read_traffic(self.traininfo, 1)
        sett.read_SoS([self.soil1], [0] * 20)
        sett.calculate(0.25, 3.5)
        sett.dump(os.path.join(TEST_PATH, "./example.pickle"))

        # compare with existing file
        with open(os.path.join(TEST_PATH, "./test_data/li_selig_2.pickle"), "rb") as f:
            data = pickle.load(f)

        res = compare_dics(sett.results.__dict__, data.__dict__)
        self.assertTrue(res)
        return

    def test_settlement_3(self):

        self.traininfo["dubbeldekker"]["forces"] = np.repeat(self.traininfo["dubbeldekker"]["forces"], 20, axis=0)
        self.traininfo["sprinter"]["forces"] = np.repeat(self.traininfo["sprinter"]["forces"], 20, axis=0)

        sett = LiSelig()
        sett.read_traffic(self.traininfo, 1)
        sett.read_SoS([self.soil1, self.soil2], [0] * 10 + [1] * 10)
        sett.calculate(0.25, 3.5)
        sett.dump(os.path.join(TEST_PATH, "./example.pickle"))

        # compare with existing file
        with open(os.path.join(TEST_PATH, "./test_data/li_selig_3.pickle"), "rb") as f:
            data = pickle.load(f)

        res = compare_dics(sett.results.__dict__, data.__dict__)
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
        res = False
        if isinstance(dic1[key], list):
            for j in range(len(dic1[key])):
                if isinstance(dic1[key][j], list):
                    if all(np.abs(np.array(dic1[key][j]) - np.array(dic2[key][j])) < tol):
                        res = True
                else:
                    if np.abs(dic1[key][j] - dic2[key][j]) < tol:
                        res = True

        result.append(res)

    if all(result):
        result = True
    else:
        result = False
    return result


if __name__ == "__main__":
    unittest.main()
