from src.geometry import Node, Element
from src.boundary_conditions import LineLoadCondition

from scipy import sparse
import numpy as np
import pytest

from src.model_part import Section, Material, TimoshenkoBeamElementModelPart


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
        force = LineLoadCondition(normal_dof=True, y_disp_dof=True, z_rot_dof=True)
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

        # sets moving load on timoshenko beam
        force.set_moving_point_load(beam, moving_coords, time, y_force=np.array([1, 1, 1, 1, 1, 1, 1]))

        # set expected values
        expected_y_force_matrix = [
            [1, 0.84375, 0.5, 0.15625, 0.0, 0.0, 0.0],
            [0.0, 0.15625, 0.5, 0.84375, 1.0, 0.84375, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.15625, 0.5],
        ]

        # assert each value in force matrix
        for i in range(len(expected_y_force_matrix)):
            for j in range(len(expected_y_force_matrix[i])):
                assert force.y_force[i, j] == pytest.approx(
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

