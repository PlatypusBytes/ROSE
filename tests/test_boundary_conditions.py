from rose.model.geometry import Node, Element
from rose.model.boundary_conditions import MovingPointLoad

import numpy as np
import pytest

from rose.model.model_part import Section, Material, TimoshenkoBeamElementModelPart


class TestBoundaryConditions:
    def test_moving_load_on_euler_beam(self, set_up_material, set_up_euler_section):
        """
        Tests moving load on a timoshenko beam. Checks per timestep how much force is applied on each node
        :param set_up_material:
        :param set_up_euler_section:
        :return:
        """

        # set geometry
        nodes_track = [Node(0.0, 0.0, 0.0), Node(1.0, 0.0, 0.0), Node(2.0, 0.0, 0.0)]
        elements_track = [
            Element([nodes_track[0], nodes_track[1]]),
            Element([nodes_track[1], nodes_track[2]]),
        ]

        # set beam input
        beam = TimoshenkoBeamElementModelPart()
        beam.elements = elements_track
        beam.nodes = nodes_track

        beam.material = set_up_material
        beam.section = set_up_euler_section

        beam.length_element = 1
        beam.calculate_mass()

        beam.damping_ratio = 0.0502
        beam.radial_frequency_one = 2
        beam.radial_frequency_two = 500

        # initialise beam matrices
        beam.initialize()

        # set time
        time = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # set moving load
        force = MovingPointLoad(x_disp_dof=True, y_disp_dof=True, z_rot_dof=True)
        force.nodes = nodes_track
        force.elements = elements_track
        force.time = time

        force.initialize_matrices()

        # set coordinate of moving load per timestep
        moving_coords = [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [1, 0.0, 0.0],
            [1.25, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ]

        active_elements = np.array([[True, True,True, True,False, False, False],
                                   [False, False,False, False,True, True, True]])

        force.moving_coords = moving_coords
        force.moving_y_force = np.array([1, 1, 1, 1, 1, 1, 1])
        force.time = time
        force.model_part_at_t = [beam for t in range(len(force.time))]

        force.active_elements = active_elements

        y_force_matrix = []
        for t in range(len(time)):
            force.update_force(t)
            y_force_matrix.append(force.y_force_vector)

        y_force_matrix = np.array(y_force_matrix).T

        # set expected values
        expected_y_force_matrix = [
            [1, 0.84375, 0.5, 0.15625, 0.0, 0.0, 0.0],
            [0.0, 0.15625, 0.5, 0.84375, 1.0, 0.84375, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.15625, 0.5],
        ]

        # assert each value in force matrix
        for i in range(len(expected_y_force_matrix)):
            for j in range(len(expected_y_force_matrix[i])):
                assert y_force_matrix[i, j] == pytest.approx(
                    expected_y_force_matrix[i][j]
                )

    def test_moving_load_on_inclined_euler_beam(self, set_up_material, set_up_euler_section):
        """
        Tests moving load on a inclined timoshenko beam. Checks per timestep how much horizontal and vertical force is
        applied on each node.
        :param set_up_material:
        :param set_up_euler_section:
        :return:
        """

        # set geometry, 2 elements of 1 meter with a 45 degree angle
        nodes_track = [Node(0.0, 0.0, 0.0), Node(np.sqrt(2)/2, np.sqrt(2)/2, 0.0), Node(np.sqrt(2), np.sqrt(2), 0.0)]

        elements_track = [
            Element([nodes_track[0], nodes_track[1]]),
            Element([nodes_track[1], nodes_track[2]]),
        ]

        # set beam input
        beam = TimoshenkoBeamElementModelPart()
        beam.elements = elements_track
        beam.nodes = nodes_track

        beam.material = set_up_material
        beam.section = set_up_euler_section

        beam.length_element = 1
        beam.calculate_mass()

        # initialise beam matrices
        beam.initialize()

        # set time
        time = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # set moving load
        force = MovingPointLoad(x_disp_dof=True, y_disp_dof=True, z_rot_dof=True)
        force.nodes = nodes_track
        force.elements = elements_track
        force.time = time

        force.initialize_matrices()

        # set coordinate of moving load per timestep

        moving_coords = [
            [0.0, 0.0, 0.0],
            [np.sqrt(2)/8, np.sqrt(2)/8, 0.0], #0.25
            [np.sqrt(2)/4, np.sqrt(2)/4, 0.0], # 0.5
            [np.sqrt(2)*3/8, np.sqrt(2)*3/8, 0.0], # 0.75
            [np.sqrt(2)/2, np.sqrt(2)/2, 0.0], # 1.0
            [np.sqrt(2)*5/8, np.sqrt(2)*5/8, 0.0], # 1.25
            [np.sqrt(2)*6/8, np.sqrt(2)*6/8, 0.0], # 1.5
        ]

        active_elements = np.array([[True, True,True, True,False, False, False],
                                    [False, False,False, False,True, True, True]])

        force.moving_coords = moving_coords
        force.moving_y_force = np.array([1, 1, 1, 1, 1, 1, 1])
        force.time = time
        force.model_part_at_t = [beam for t in range(len(force.time))]

        force.active_elements = active_elements

        # set expected values
        expected_y_force_matrix = [
            [1, 0.796875, 0.5, 0.203125, 0.0, 0.0, 0.0],
            [0.0, 0.203125, 0.5, 0.796875, 1.0, 0.796875, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.203125, 0.5],
        ]

        expected_x_force_matrix = [
            [0, -0.046875, 0.0, 0.046875, 0.0, 0.0, 0.0],
            [0.0, 0.046875, 0.0, -0.046875, 0.0, -0.046875, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.046875, 0.0],
        ]

        x_force_matrix = []
        y_force_matrix = []
        for t in range(len(time)):
            force.update_force(t)
            x_force_matrix.append(force.x_force_vector)
            y_force_matrix.append(force.y_force_vector)

        x_force_matrix = np.array(x_force_matrix).T
        y_force_matrix = np.array(y_force_matrix).T

        # assert each value in force matrix
        for i in range(len(expected_y_force_matrix)):
            for j in range(len(expected_y_force_matrix[i])):
                # assert horizontal force
                assert x_force_matrix[i, j] == pytest.approx(
                    expected_x_force_matrix[i][j]
                )

                # assert vertical force
                assert y_force_matrix[i, j] == pytest.approx(
                    expected_y_force_matrix[i][j]
                )

@pytest.fixture
def set_up_material():
    # Steel
    material = Material()
    material.youngs_modulus = 200e9  # Pa
    material.poisson_ratio = 0.0
    material.density = 8000
    return material


@pytest.fixture
def set_up_euler_section():
    section = Section()
    section.area = 1e-3
    section.sec_moment_of_inertia = 2e-5
    section.shear_factor = 0
    return section

