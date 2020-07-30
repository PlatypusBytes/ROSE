import pytest

from src.geometry import Node, Element

from src.track import *
from src.soil import Soil
from src.global_system import *
import matplotlib.pyplot as plt
from one_dimensional.solver import Solver


class TestTrack:
    def test_euler_stiffness_matrix_track(self, set_up_euler_rail, expected_euler_rail_stiffness_matrix):
        rail = set_up_euler_rail
        rail.set_aux_stiffness_matrix()
        rail_euler_stiffness_matrix = rail.aux_stiffness_matrix

        expected_stiffness_matrix = expected_euler_rail_stiffness_matrix

        for i in range(len(expected_stiffness_matrix)):
            for j in range(len(expected_stiffness_matrix[i])):
                assert rail_euler_stiffness_matrix[i, j] == pytest.approx(expected_stiffness_matrix[i][j])

    def test_euler_mass_matrix_track(self, set_up_euler_rail, expected_euler_rail_mass_matrix):
        rail = set_up_euler_rail
        rail.set_aux_mass_matrix()
        rail_euler_mass_matrix = rail.aux_mass_matrix

        expected_mass_matrix = expected_euler_rail_mass_matrix

        for i in range(len(expected_mass_matrix)):
            for j in range(len(expected_mass_matrix[i])):
                assert rail_euler_mass_matrix[i, j] == pytest.approx(expected_mass_matrix[i][j])

    def test_euler_damping_matrix(self, set_up_euler_rail, expected_euler_rail_damping_matrix):
        rail = set_up_euler_rail
        rail.set_aux_mass_matrix()
        rail.set_aux_stiffness_matrix()

        rail.set_aux_damping_matrix()
        rail_damping_matrix = rail.aux_damping_matrix

        expected_damping_matrix = expected_euler_rail_damping_matrix

        for i in range(len(expected_damping_matrix)):
            for j in range(len(expected_damping_matrix[i])):
                assert rail_damping_matrix[i, j] == pytest.approx(expected_damping_matrix[i][j])

    def test_initialize_euler_beam(self, set_up_euler_rail, expected_euler_rail_damping_matrix):
        rail = set_up_euler_rail
        rail.initialize()

        rail_damping_matrix = rail.aux_damping_matrix
        expected_damping_matrix = expected_euler_rail_damping_matrix

        assert rail.timoshenko_factor == 0
        for i in range(len(expected_damping_matrix)):
            for j in range(len(expected_damping_matrix[i])):
                assert rail_damping_matrix[i, j] == pytest.approx(expected_damping_matrix[i][j])

    def test_initialize_timoshenko_beam(self, set_up_euler_rail):
        rail = set_up_euler_rail
        rail.section.shear_factor = 0.5

        rail.initialize()

        rail_damping_matrix = rail.aux_damping_matrix
        assert rail.timoshenko_factor == pytest.approx(0.0096)

        # for i in range(len(expected_damping_matrix)):
        #     for j in range(len(expected_damping_matrix[i])):
        #         assert rail_damping_matrix[i, j] == pytest.approx(expected_damping_matrix[i][j])


@pytest.fixture
def set_up_material():
    # Steel
    material = Material()
    material.youngs_modulus = 200E9  # Pa
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


@pytest.fixture
def set_up_euler_rail(set_up_material, set_up_euler_section):

    nodes = [Node(0, 0, 0), Node(10, 0, 0)]
    elements = [Element([nodes])]
    rail = Rail()
    rail.nodes = nodes
    rail.elements = elements
    rail.section = set_up_euler_section
    rail.material = set_up_material

    rail.calculate_length_rail()
    rail.calculate_mass()
    rail.calculate_n_dof()

    rail.damping_ratio = 0.0502
    rail.radial_frequency_one = 2
    rail.radial_frequency_two = 500
    return rail


@pytest.fixture
def expected_euler_rail_stiffness_matrix():
    expected_stiffness_matrix = [[2e7, 0, 0, -2e7, 0, 0],
                                 [0, 4.8e4, 2.4e5, 0, -4.8e4, 2.4e5],
                                 [0, 2.4e5, 1.6e6, 0, -2.4e5, 8e5],
                                 [-2e7, 0, 0, 2e7, 0, 0],
                                 [0, -4.8e4, -2.4e5, 0, 4.8e4, -2.4e5],
                                 [0, 2.4e5, 8e5, 0, -2.4e5, 1.6e6]]
    return expected_stiffness_matrix


@pytest.fixture
def expected_euler_rail_mass_matrix():
    expected_mass_matrix = [[26.66666667, 0, 0, 13.33333333, 0, 0],
                            [0, 29.73348571, 41.9207619, 0, 10.26651429, -24.74590476],
                            [0, 41.9207619, 76.40380952, 0, 24.74590476, -57.19619048],
                            [13.33333333, 0, 0, 26.66666667, 0, 0],
                            [0, 10.26651429, 24.74590476, 0, 29.73348571, -41.9207619],
                            [0, -24.74590476, -57.19619048, 0, -41.9207619, 76.40380952]]
    return expected_mass_matrix


@pytest.fixture
def expected_euler_rail_damping_matrix():
    expected_damping_matrix = [[4005.33333333, 0.00000000, 0.00000000, -3997.33333333, 0.00000000, 0.00000000],
                               [0.00000000, 15.54669714, 56.38415238, 0.00000000, -7.54669714, 43.05081905],
                               [0.00000000, 56.38415238, 335.28076190, 0.00000000, -43.05081905, 148.56076190],
                               [-3997.33333333, 0.00000000, 0.00000000, 4005.33333333, 0.00000000, 0.00000000],
                               [0.00000000, -7.54669714, -43.05081905, 0.00000000, 15.54669714, -56.38415238],
                               [0.00000000, 43.05081905, 148.56076190, 0.00000000, - 56.38415238, 335.28076190]]
    return expected_damping_matrix


@pytest.fixture
def set_up_sleeper():
    sleeper = Sleeper()
    sleeper.mass = 162.5
    sleeper.distance_between_sleepers = 0.6
    return sleeper


@pytest.fixture
def set_up_rail_pad():
    rail_pad = RailPad()
    rail_pad.mass = 5
    rail_pad.stiffness = 145e6
    rail_pad.damping = 12e3
    return rail_pad
