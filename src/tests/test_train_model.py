import pytest

from src.train_model.track import *
import matplotlib.pyplot as plt
from one_dimensional.solver import Solver

class TestTrainModel:

    def test_stiffness_matrix_track(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):

        utrack = UTrack(3)

        utrack.rail = set_up_rail
        utrack.sleeper = set_up_sleeper
        utrack.rail_pads = set_up_rail_pad
        utrack.soil = set_up_soil

        utrack.calculate_n_dofs()
        utrack.set_global_stiffness_matrix()

        plt.spy(utrack.global_stiffness_matrix)
        plt.show()


    def test_mass_matrix_track(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):
        utrack = UTrack(3)

        utrack.rail = set_up_rail
        utrack.sleeper = set_up_sleeper
        utrack.rail_pads = set_up_rail_pad
        utrack.soil = set_up_soil

        utrack.calculate_n_dofs()
        utrack.set_global_mass_matrix()

        plt.spy(utrack.global_mass_matrix)
        plt.show()

    def test_damping_matrix_track(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):
        utrack = UTrack(3)

        utrack.rail = set_up_rail
        utrack.sleeper = set_up_sleeper
        utrack.rail_pads = set_up_rail_pad
        utrack.soil = set_up_soil

        utrack.calculate_n_dofs()

        utrack.set_global_stiffness_matrix()
        utrack.set_global_mass_matrix()
        utrack.set_global_damping_matrix()

        plt.spy(utrack.global_damping_matrix)
        plt.show()

    def test_force_array(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):
        utrack = UTrack(3)

        utrack.rail = set_up_rail
        utrack.sleeper = set_up_sleeper
        utrack.rail_pads = set_up_rail_pad
        utrack.soil = set_up_soil

        utrack.calculate_n_dofs()

        utrack.set_force()

        utrack.apply_boundary_condition()

        plt.spy(utrack.force)
        plt.show()

    def test_calculate_point_load(self, set_up_rail, set_up_sleeper, set_up_rail_pad, set_up_soil):
        utrack = UTrack(3)

        utrack.rail = set_up_rail
        utrack.sleeper = set_up_sleeper
        utrack.rail_pads = set_up_rail_pad
        utrack.soil = set_up_soil

        utrack.calculate_n_dofs()

        utrack.set_global_stiffness_matrix()
        utrack.set_global_mass_matrix()
        utrack.set_global_damping_matrix()

        utrack.set_force()

        utrack.apply_boundary_condition()
        solver = Solver(utrack.n_dof_track-3)

        settings_newmark = {"gamma": 0.5,
                            "beta": 0.25}

        time = utrack.time

        solver.newmark(settings_newmark, utrack.global_mass_matrix, utrack.global_damping_matrix, utrack.global_stiffness_matrix, utrack.force, time[1] - time[0], time[-1], t_start=time[0])


        plt.plot(solver.time, solver.u[:, 1])
        plt.plot(solver.time, solver.u[:, 4])
        plt.plot(solver.time, solver.u[:, 7])
        plt.plot(solver.time, solver.u[:, 9])
        plt.plot(solver.time, solver.u[:, 10])
        plt.plot(solver.time, solver.u[:, 11])

        plt.legend(["1","4","7","9","10","11"])
        plt.show()



@pytest.fixture
def set_up_material():
    # Steel
    material = Material()
    material.youngs_modulus = 210E9 # Pa
    material.poisson_ratio = 0.3
    material.density = 7860
    return material


@pytest.fixture
def set_up_section():
    section = Section()
    section.area = 69.682e-4
    section.sec_moment_of_inertia = 2337.9e-8
    section.shear_factor = 0
    section.n_rail_per_sleeper = 2
    return section

@pytest.fixture
def set_up_rail(set_up_material, set_up_section):
    rail = Rail(3)
    rail.section = set_up_section
    rail.material = set_up_material

    rail.calculate_length_rail(0.6)
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
    return  rail_pad

@pytest.fixture
def set_up_soil():
    soil = Soil()
    soil.stiffness = 300e6
    soil.damping = 0
    return soil



# @pytest.fixture()
# def set_up_rail():
