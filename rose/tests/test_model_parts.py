import pytest

from rose.base.geometry import Element, Node

from rose.track.track import *
from rose.base.global_system import *

from rose.base.model_part import *

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

        rod_element.set_rotation_matrix(0, 2)
        calculated_rotation_matrix = rod_element.rotation_matrix

        expected_rotation_matrix = [[1., 0., 0., 0., 0., 0.],
                                    [0., 1., 0., 0., 0., 0.],
                                    [0., 0., 1., 0., 0., 0.],
                                    [0., 0., 0., 1., 0., 0.],
                                    [0., 0., 0., 0., 1., 0.],
                                    [0., 0., 0., 0., 0., 1.]]

        for row in range(len(expected_rotation_matrix)):
            for col in range(len(expected_rotation_matrix[row])):
                assert expected_rotation_matrix[row][col] == pytest.approx(calculated_rotation_matrix[row,col])

    def test_set_2d_rotation_matrix_90_rot(self):
        rod_element = RodElementModelPart()

        rod_element.set_rotation_matrix(0.5*np.pi, 2)
        calculated_rotation_matrix = rod_element.rotation_matrix

        expected_rotation_matrix = [[0, 1, 0, 0, 0, 0],
                                    [-1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, -1, 0, 0],
                                    [0, 0, 0, 0, 0, 1]]

        for row in range(len(expected_rotation_matrix)):
            for col in range(len(expected_rotation_matrix[row])):
                assert expected_rotation_matrix[row][col] == pytest.approx(calculated_rotation_matrix[row, col])

    def test_set_aux_mass_matrix(self):
        rod_element = RodElementModelPart()
        rod_element.mass = 1
        rod_element.set_aux_mass_matrix()
        expected_mass_matrix = [[1/3, 1/6],
                                [1/6, 1/3]]

        for row in range(len(expected_mass_matrix)):
            for col in range(len(expected_mass_matrix[row])):
                assert expected_mass_matrix[row][col] == pytest.approx(rod_element.aux_mass_matrix[row,col])

    def test_set_aux_stiffness_matrix(self):
        rod_element = RodElementModelPart()
        rod_element.stiffness = 1
        rod_element.set_aux_stiffness_matrix()
        expected_stiffness_matrix = [[1, -1],
                                    [-1, 1]]

        for row in range(len(expected_stiffness_matrix)):
            for col in range(len(expected_stiffness_matrix[row])):
                assert expected_stiffness_matrix[row][col] == pytest.approx(rod_element.aux_stiffness_matrix[row,col])

    def test_set_aux_damping_matrix(self):
        rod_element = RodElementModelPart()
        rod_element.damping = 1
        rod_element.set_aux_damping_matrix()
        expected_damping_matrix = [[1, -1],
                                     [-1, 1]]

        for row in range(len(expected_damping_matrix)):
            for col in range(len(expected_damping_matrix[row])):
                assert expected_damping_matrix[row][col] == pytest.approx(rod_element.aux_damping_matrix[row, col])

class TestTimoshenkoBeamElementModelPart:

    def test_euler_stiffness_matrix_track(
            self, set_up_euler_beam, expected_euler_beam_stiffness_matrix
    ):
        beam = set_up_euler_beam
        beam.set_aux_stiffness_matrix()
        beam_stiffness_matrix = beam.aux_stiffness_matrix

        expected_stiffness_matrix = expected_euler_beam_stiffness_matrix

        for i in range(len(expected_stiffness_matrix)):
            for j in range(len(expected_stiffness_matrix[i])):
                assert beam_stiffness_matrix[i, j] == pytest.approx(
                    expected_stiffness_matrix[i][j]
                )

    def test_euler_mass_matrix_track(
            self, set_up_euler_beam, expected_euler_beam_mass_matrix
    ):
        beam = set_up_euler_beam
        beam.set_aux_mass_matrix()
        beam_mass_matrix = beam.aux_mass_matrix

        expected_mass_matrix = expected_euler_beam_mass_matrix

        for i in range(len(expected_mass_matrix)):
            for j in range(len(expected_mass_matrix[i])):
                assert beam_mass_matrix[i, j] == pytest.approx(
                    expected_mass_matrix[i][j]
                )

    def test_initialize_euler_beam(
            self, set_up_euler_beam, expected_euler_beam_stiffness_matrix, expected_euler_beam_mass_matrix
    ):
        beam = set_up_euler_beam
        beam.initialize()

        for i in range(len(expected_euler_beam_mass_matrix)):
            for j in range(len(expected_euler_beam_mass_matrix[i])):
                assert beam.aux_mass_matrix[i, j] == pytest.approx(
                    expected_euler_beam_mass_matrix[i][j]
                )
                assert beam.aux_stiffness_matrix[i, j] == pytest.approx(
                    expected_euler_beam_stiffness_matrix[i][j]
                )

    def test_initialize_timoshenko_beam(self, set_up_euler_beam):
        rail = set_up_euler_beam
        rail.section.shear_factor = 0.5

        rail.initialize()

        assert rail.timoshenko_factor == pytest.approx(0.0096)


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


@pytest.fixture
def set_up_euler_beam(set_up_material, set_up_euler_section):
    nodes = [Node(0, 0, 0), Node(10, 0, 0)]
    elements = [Element(nodes)]

    beam = TimoshenkoBeamElementModelPart()
    beam.nodes = nodes
    beam.elements = elements
    beam.section = set_up_euler_section
    beam.material = set_up_material

    beam.length_element = 10

    beam.calculate_mass()

    return beam


@pytest.fixture
def expected_euler_beam_stiffness_matrix():
    expected_stiffness_matrix = [
        [2e7, 0, 0, -2e7, 0, 0],
        [0, 4.8e4, 2.4e5, 0, -4.8e4, 2.4e5],
        [0, 2.4e5, 1.6e6, 0, -2.4e5, 8e5],
        [-2e7, 0, 0, 2e7, 0, 0],
        [0, -4.8e4, -2.4e5, 0, 4.8e4, -2.4e5],
        [0, 2.4e5, 8e5, 0, -2.4e5, 1.6e6],
    ]
    return expected_stiffness_matrix


@pytest.fixture
def expected_euler_beam_mass_matrix():
    expected_mass_matrix = [
        [26.66666667, 0, 0, 13.33333333, 0, 0],
        [0.00000000, 29.71428571, 41.90476190, 0.00000000, 10.28571429, -24.76190476],
        [0.00000000, 41.90476190, 76.19047619, 0.00000000, 24.76190476, -57.14285714],
        [13.33333333, 0.00000000, 0.00000000, 26.66666667, 0.00000000, 0.00000000],
        [0.00000000, 10.28571429, 24.76190476, 0.00000000, 29.71428571, -41.90476190],
        [0.00000000, -24.76190476, -57.14285714, 0.00000000, -41.90476190, 76.19047619]
    ]

    return expected_mass_matrix


@pytest.fixture
def expected_euler_beam_mass_matrix_with_rotation():
    expected_mass_matrix = [
        [26.66666667, 0, 0, 13.33333333, 0, 0],
        [0, 29.73348571, 41.9207619, 0, 10.26651429, -24.74590476],
        [0, 41.9207619, 76.40380952, 0, 24.74590476, -57.19619048],
        [13.33333333, 0, 0, 26.66666667, 0, 0],
        [0, 10.26651429, 24.74590476, 0, 29.73348571, -41.9207619],
        [0, -24.74590476, -57.19619048, 0, -41.9207619, 76.40380952],
    ]
    return expected_mass_matrix


# @pytest.fixture
# def expected_euler_beam_damping_matrix():
#     expected_damping_matrix = [
#         [4005.33333333, 0.00000000, 0.00000000, -3997.33333333, 0.00000000, 0.00000000],
#         [0.00000000, 15.54285714, 56.38095238, 0.00000000, -7.54285714, 43.04761905],
#         [0.00000000, 56.38095238, 335.23809524, 0.00000000, -43.04761905, 148.57142857],
#         [- 3997.33333333, 0.00000000, 0.00000000, 4005.33333333, 0.00000000, 0.00000000],
#         [0.00000000, -7.54285714, -43.04761905, 0.00000000, 15.54285714, -56.38095238],
#         [0.00000000, 43.04761905, 148.57142857, 0.00000000, -56.38095238, 335.23809524]
#     ]
#     return expected_damping_matrix

#
# @pytest.fixture
# def expected_euler_beam_damping_matrix_with_rotation():
#     expected_damping_matrix = [
#         [4005.33333333, 0.00000000, 0.00000000, -3997.33333333, 0.00000000, 0.00000000],
#         [0.00000000, 15.54669714, 56.38415238, 0.00000000, -7.54669714, 43.05081905],
#         [0.00000000, 56.38415238, 335.28076190, 0.00000000, -43.05081905, 148.56076190],
#         [-3997.33333333, 0.00000000, 0.00000000, 4005.33333333, 0.00000000, 0.00000000],
#         [0.00000000, -7.54669714, -43.05081905, 0.00000000, 15.54669714, -56.38415238],
#         [0.00000000, 43.05081905, 148.56076190, 0.00000000, -56.38415238, 335.28076190],
#     ]
#     return expected_damping_matrix




# class TestModelParts:
#     def test_()

