import pytest

from src.train_model.track import *

class TestTrainModel:

    def test_mass_matrix_rail(self, set_up_rail, set_up_sleeper, set_up_rail_pad):

        utrack = UTrack(3)

        utrack.rail = set_up_rail
        utrack.sleeper = set_up_sleeper
        utrack.rail_pads = set_up_rail_pad


        utrack.calculate_n_dofs()
        utrack.calculate_mass_matrix()

        utrack.rail.calculate_timoshenko_factor()
        utrack.rail.set_aux_stiffness_matrix()

        utrack.set_global_stiffness_matrix()

        import matplotlib.pyplot as plt

        plt.spy(utrack.global_stiffness_matrix)
        plt.show()
        test = utrack.mass_matrix_track



        print('test')






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
    return rail
    # rail.section = request.

@pytest.fixture
def set_up_sleeper():
    sleeper = Sleeper()
    sleeper.mass = 162.5
    sleeper.distance_between_sleepers = 0.6
    sleeper.damping_ratio = 0.04
    sleeper.radial_frequency_one = 2
    sleeper.radial_frequency_two = 500
    return sleeper

@pytest.fixture
def set_up_rail_pad():
    rail_pad = RailPad()
    rail_pad.mass = 5
    rail_pad.stiffness = 145e6
    rail_pad.damping = 12e3
    return  rail_pad



# @pytest.fixture()
# def set_up_rail():
