import pytest

from src.geometry import Element

from src.track import *
from src.global_system import *

from src.model_part import *

class TestRodElement:

    def test_initialize_rod_element(self):
        rod_element = RodElementModelPart()

        assert rod_element.normal_dof == True
        assert rod_element.y_disp_dof == False
        assert rod_element.z_disp_dof == False

        assert rod_element.x_rot_dof == False
        assert rod_element.y_rot_dof == False
        assert rod_element.z_rot_dof == False


    def test_set_2d_rotation_matrix_0_rot(self):
        rod_element = RodElementModelPart()

        rod_element.set_rotation_matrix(0)
        calculated_rotation_matrix = rod_element.rotation_matrix

        expected_rotation_matrix = [[1, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]]

        for row in range(len(expected_rotation_matrix)):
            for col in range(len(expected_rotation_matrix[row])):
                assert pytest.approx(expected_rotation_matrix[row][col], calculated_rotation_matrix[row,col] )

    def test_set_2d_rotation_matrix_90_rot(self):
        rod_element = RodElementModelPart()

        rod_element.set_rotation_matrix(0.5*np.pi)
        calculated_rotation_matrix = rod_element.rotation_matrix

        expected_rotation_matrix = [[0, -1, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, -1, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 1]]

        for row in range(len(expected_rotation_matrix)):
            for col in range(len(expected_rotation_matrix[row])):
                assert pytest.approx(expected_rotation_matrix[row][col], calculated_rotation_matrix[row,col])

    def test_set_aux_mass_matrix(self):
        rod_element = RodElementModelPart()
        rod_element.mass = 1
        rod_element.set_aux_mass_matrix()
        expected_mass_matrix = [[1/3, 1/6],
                                [1/6, 1/3]]

        for row in range(len(expected_mass_matrix)):
            for col in range(len(expected_mass_matrix[row])):
                assert pytest.approx(expected_mass_matrix[row][col], rod_element.aux_mass_matrix[row,col])

    def test_set_aux_stiffness_matrix(self):
        rod_element = RodElementModelPart()
        rod_element.stiffness = 1
        rod_element.set_aux_stiffness_matrix()
        expected_stiffness_matrix = [[1, -1],
                                    [1, -1]]

        for row in range(len(expected_stiffness_matrix)):
            for col in range(len(expected_stiffness_matrix[row])):
                assert pytest.approx(expected_stiffness_matrix[row][col], rod_element.aux_stiffness_matrix[row,col])

    def test_set_aux_damping_matrix(self):
        rod_element = RodElementModelPart()
        rod_element.damping = 1
        rod_element.set_aux_damping_matrix()
        expected_damping_matrix = [[1, -1],
                                     [1, -1]]

        for row in range(len(expected_damping_matrix)):
            for col in range(len(expected_damping_matrix[row])):
                assert pytest.approx(expected_damping_matrix[row][col], rod_element.aux_damping_matrix[row, col])


# class TestModelParts:
#     def test_()

