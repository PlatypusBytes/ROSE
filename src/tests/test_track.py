
import pytest

from src.track import *
from src.soil import Soil
from src.global_system import *
import matplotlib.pyplot as plt
from one_dimensional.solver import Solver


class TestTrack:
    def test_stiffness_matrix_track(self, set_up_euler_rail):
        rail = set_up_euler_rail
        rail.set_aux_stiffness_matrix()
        rail_euler_stiffness_matrix = rail.aux_stiffness_matrix

        expected_stiffness_matrix = [[2e7, 0, 0, -2e7, 0, 0],
                                     [0, 4.8e4, 2.4e5, 0, -4.8e4, 2.4e5],
                                     [0, 2.4e5, 1.6e6, 0, -2.4e5, 8e5],
                                     [-2e7, 0, 0, 2e7, 0, 0],
                                     [0, -4.8e4, -2.4e5, 0, 4.8e4, -2.4e5],
                                     [0, 2.4e5, 8e5, 0, -2.4e5, 1.6e6]]

        for i in range(len(expected_stiffness_matrix)):
            for j in range(len(expected_stiffness_matrix[i])):
                assert rail_euler_stiffness_matrix[i,j] == pytest.approx(expected_stiffness_matrix[i][j])






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
    rail = Rail(2)
    rail.section = set_up_euler_section
    rail.material = set_up_material

    rail.calculate_length_rail(10)
    rail.calculate_mass()
    rail.calculate_n_dof()

    rail.damping_ratio = 0.04
    rail.radial_frequency_one = 2
    rail.radial_frequency_two = 500
    return rail


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