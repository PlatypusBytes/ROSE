import pytest

from src.train_model.track import *

class TestTrainModel:

    def test_mass_matrix_rail(self, set_up_rail):

        utrack = UTrack(10)


        utrack.rail = set_up_rail

        utrack.rail.calculate_mass_matrix()

        test = utrack.rail.mass_matrix

        # utrack.rail.calculate_length_rail()


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
    section.n_rail_per_sleeper = 3
    return section

@pytest.fixture
def set_up_rail(set_up_material, set_up_section):
    rail = Rail(10)
    rail.section = set_up_section
    rail.material = set_up_material

    rail.calculate_length_rail(0.6)
    rail.calculate_mass()
    rail.calculate_n_dof()
    return rail
    # rail.section = request.


# @pytest.fixture()
# def set_up_rail():
