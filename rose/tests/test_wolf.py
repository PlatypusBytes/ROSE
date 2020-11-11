import unittest
import os
import json
import shutil
import numpy as np
# import package
from rose.wolf import LayeredHalfSpace

TEST_PATH = "./rose/tests"
tol = 1e-12


class TestWolf(unittest.TestCase):
    def setUp(self):
        self.omega = np.linspace(0, 5, 150)
        self.output_folder = os.path.join(TEST_PATH, "./results")
        self.freq = False

        # load datasets
        with open(os.path.join(TEST_PATH, "./test_data/Kdyn_vertical.json")) as f:
            self.vertical = json.load(f)

        with open(os.path.join(TEST_PATH, "./test_data/Kdyn_horizontal.json")) as f:
            self.horizontal = json.load(f)

        return

    def test_vertical_solution(self):
        layer_file = os.path.join(TEST_PATH, "./test_data/input_V.csv")

        layers = LayeredHalfSpace.read_file(layer_file)

        data = LayeredHalfSpace.Layers(layers)

        data.assign_properties()
        data.correction_incompressible()
        data.static_cone()
        data.dynamic_stiffness(self.omega)

        LayeredHalfSpace.write_output(self.output_folder, os.path.splitext(os.path.split(layer_file)[-1])[0],
                                      data, self.omega, self.freq)

        # compare dicts
        with open(os.path.join(self.output_folder, "Kdyn_input_V.json")) as f:
            data = json.load(f)

        compare_dics(self.vertical, data)

        return

    def test_horizontal_solution(self):
        layer_file = os.path.join(TEST_PATH, "./test_data/input_H.csv")

        layers = LayeredHalfSpace.read_file(layer_file)

        data = LayeredHalfSpace.Layers(layers)

        data.assign_properties()
        data.correction_incompressible()
        data.static_cone()
        data.dynamic_stiffness(self.omega)

        LayeredHalfSpace.write_output(self.output_folder, os.path.splitext(os.path.split(layer_file)[-1])[0],
                                      data, self.omega, self.freq)

        # compare dicts
        with open(os.path.join(self.output_folder, "Kdyn_input_H.json")) as f:
            data = json.load(f)

        compare_dics(self.horizontal, data)
        return

    def test_files(self):
        layer_file = os.path.join(TEST_PATH, "./test_data/input_V.csv")

        layers = LayeredHalfSpace.read_file(layer_file)

        data = LayeredHalfSpace.Layers(layers)

        data.assign_properties()
        data.correction_incompressible()
        data.static_cone()
        data.dynamic_stiffness(self.omega)

        LayeredHalfSpace.write_output(self.output_folder, os.path.splitext(os.path.split(layer_file)[-1])[0],
                                      data, self.omega, self.freq)

        # check if folders and files exist
        self.assertTrue(os.path.isdir(self.output_folder))
        print(os.getcwd())
        print(self.output_folder)
        self.assertTrue(os.path.isfile(os.path.join(self.output_folder, "Kdyn_input_V.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.output_folder, "input_V.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.output_folder, "input_V.pdf")))

        return

    def tearDown(self):
        shutil.rmtree(self.output_folder)
        return


def compare_dics(dic1, dic2):

    result = []
    for key in dic1:
        for j in range(len(dic1[key])):
            if isinstance(dic1[key][j], list):
                for k in range(len(dic1[key][j])):
                    if (dic1[key][j][k] - dic2[key][j][k]) < tol:
                        result.append(True)
                    else:
                        result.append(False)
            else:
                if (dic1[key][j] - dic2[key][j]) < tol:
                    result.append(True)
                elif (dic1[key][j] == np.inf) and (dic2[key][j] == np.inf):
                    result.append(True)
                else:
                    result.append(False)

    if all(result):
        result = True
    else:
        result = False
    return result


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
