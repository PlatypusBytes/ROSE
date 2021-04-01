import pytest

from rose.utils.utils import *
from rose.model.model_part import RodElementModelPart,TimoshenkoBeamElementModelPart, ElementModelPart
from rose.model.geometry import Element

import numpy as np

class TestUtils:

    @pytest.mark.parametrize("node1, node2, expected_rotation",[
        pytest.param(Node(0, 0, 0), Node(0, 0, 0), 0),
        pytest.param(Node(0, 0, 0), Node(1,0,0), 0),
        pytest.param(Node(0, 0, 0), Node(0, 1, 0), 0.5 * np.pi),
        pytest.param(Node(0, 0, 0), Node(-1, 0, 0), np.pi),
        pytest.param(Node(0, 0, 0), Node(0, -1, 0), -0.5 * np.pi),
        pytest.param(Node(0, 0, 0), Node(1, 1, 0), np.pi/4),
        pytest.param(Node(0, 0, 0), Node(-1, 1, 0), 3 * np.pi / 4),
        pytest.param(Node(0, 0, 0), Node(-1, -1, 0), 5 * np.pi / 4),
        pytest.param(Node(0, 0, 0), Node(1, -1, 0), -np.pi / 4),
    ]
                             )
    def test_calculate_rotation(self, node1, node2, expected_rotation):
        """
        Checked rotation between 2 nodes
        :param node1:
        :param node2:
        :param expected_rotation:
        :return:
        """
        assert expected_rotation == pytest.approx(calculate_rotation(node1.coordinates[None,:], node2.coordinates[None,:]))

    def test_reshape_aux_matrix(self):
        rod = RodElementModelPart()

        rod.stiffness = 1
        rod.initialize()
        aux_matrix = rod.aux_stiffness_matrix
        reshaped_aux_matrix = reshape_aux_matrix(2, [True, False, False], aux_matrix)

        expected_aux_matrix = np.zeros((6,6))
        expected_aux_matrix[0, 0] = 1
        expected_aux_matrix[3, 3] = 1
        expected_aux_matrix[0, 3] = -1
        expected_aux_matrix[3, 0] = -1

        for i in range(len(expected_aux_matrix)):
            for j in range(len(expected_aux_matrix[i])):
                assert expected_aux_matrix[i,j] == pytest.approx(reshaped_aux_matrix[i,j])


    def test_rotate_aux_matrix_no_rotation(self):
        """
        Checks a 0 degrees rotated rod aux matrix
        :return:
        """

        rod = RodElementModelPart()
        rod.stiffness = 1
        rod.initialize()

        element = Element([Node(0, 0, 0), Node(1, 0, 0)])

        aux_matrix = rod.aux_stiffness_matrix
        aux_matrix = reshape_aux_matrix(2, [True, False, False], aux_matrix)

        rotated_matrix = rotate_aux_matrix(element,  rod, aux_matrix )

        expected_aux_matrix = np.zeros((6, 6))
        expected_aux_matrix[0, 0] = 1
        expected_aux_matrix[3, 3] = 1
        expected_aux_matrix[0, 3] = -1
        expected_aux_matrix[3, 0] = -1

        for i in range(len(expected_aux_matrix)):
            for j in range(len(expected_aux_matrix[i])):
                assert expected_aux_matrix[i, j] == pytest.approx(rotated_matrix[i, j])

    def test_rotate_aux_matrix_90_rotation(self):
        """
        Checks a 90 degrees rotated rod aux matrix
        :return:
        """
        rod = RodElementModelPart()
        rod.stiffness = 1
        rod.initialize()

        element = Element([Node(0, 0, 0), Node(0, 1, 0)])

        aux_matrix = rod.aux_stiffness_matrix
        aux_matrix = reshape_aux_matrix(2, [True, False, False], aux_matrix)

        rotated_matrix = rotate_aux_matrix(element, rod, aux_matrix)

        expected_aux_matrix = np.zeros((6, 6))
        expected_aux_matrix[1, 1] = 1
        expected_aux_matrix[4, 4] = 1
        expected_aux_matrix[1, 4] = -1
        expected_aux_matrix[4, 1] = -1

        for i in range(len(expected_aux_matrix)):
            for j in range(len(expected_aux_matrix[i])):
                assert expected_aux_matrix[i, j] == pytest.approx(rotated_matrix[i, j])

    def test_rotate_aux_matrix_135_rotation(self):
        """
        Checks a 135 degrees rotated rod aux matrix
        :return:
        """

        # initialise rod element
        rod = RodElementModelPart()
        rod.stiffness = 1
        rod.initialize()
        element = Element([Node(0, 0, 0), Node(-1, 1, 0)])

        aux_matrix = rod.aux_stiffness_matrix

        # reshape aux matrix
        aux_matrix = reshape_aux_matrix(2, [True, False, False], aux_matrix)

        # rotate aux matrix
        rotated_matrix = rotate_aux_matrix(element, rod, aux_matrix)

        # set up expected matrix
        expected_aux_matrix = np.zeros((6, 6))
        expected_aux_matrix[[0, 1, 3, 4], [0, 0, 0, 0]] = [0.5, 0.5, -0.5, -0.5]
        expected_aux_matrix[[0, 1, 3, 4], [1, 1, 1, 1]] = [0.5, 0.5, -0.5, -0.5]
        expected_aux_matrix[[0, 1, 3, 4], [3, 3, 3, 3]] = [-0.5, -0.5, 0.5, 0.5]
        expected_aux_matrix[[0, 1, 3, 4], [4, 4, 4, 4]] = [-0.5, -0.5, 0.5, 0.5]

        # assert rotated aux matrix
        for i in range(len(expected_aux_matrix)):
            for j in range(len(expected_aux_matrix[i])):
                assert expected_aux_matrix[i, j] == pytest.approx(rotated_matrix[i, j])

    def test_rotate_aux_matrix_without_existing_rot_matrix(self):

        # Create non existing element model part
        test_element = ElementModelPart()
        test_element.aux_stiffness_matrix = np.zeros((9,9))
        test_element.elements = [Element([Node(0, 0, 0), Node(-1, 1, 0), Node(1, 1, 0)])]

        # fill arbitrary aux_matrix
        k = 0
        for i in range(len(test_element.aux_stiffness_matrix)):
            for j in range(len(test_element.aux_stiffness_matrix[i])):
                k += 1
                test_element.aux_stiffness_matrix[i,j] = k
        expected_matrix = copy.copy(test_element.aux_stiffness_matrix)

        # rotate matrix
        rotated_matrix = rotate_aux_matrix(test_element.elements[0],
                                           test_element, test_element.aux_stiffness_matrix)

        # Check if aux matrix has not changed
        for i in range(len(expected_matrix)):
            for j in range(len(expected_matrix[i])):
                assert expected_matrix[i, j] == pytest.approx(rotated_matrix[i,j])

    def test_rotate_force_vector_30_rotation(self):
        beam = TimoshenkoBeamElementModelPart()
        beam.length_element = 1
        beam.mass = 1

        element = Element([Node(0,0,0), Node(np.sqrt(3)/2, 0.5,0)])
        # test_element.elements = [Element([Node(0,0,0), Node(np.sqrt(3)/2, 0.5,0)])]

        # global force vector
        force_vector = np.array([-1000,0000,0])

        # calculate element rotation
        rot = calculate_rotation(element.nodes[0].coordinates[None,:], element.nodes[1].coordinates[None,:])

        # calculate rotated force vector
        rotated_vector = rotate_point_around_z_axis([rot],force_vector[None,:])[0]

        # define expected rotated force vector
        expected_force_vector = np.array([-np.sqrt(3)*1000/2,500,0])

        # assert
        np.testing.assert_array_almost_equal(rotated_vector,expected_force_vector)



