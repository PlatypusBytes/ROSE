import pytest

from rose.base.global_system import *

from rose.base.model_part import *

class TestGlobalSystem:
    def test_calculate_rayleigh_damping(self,euler_beam_stiffness_matrix,euler_beam_mass_matrix,
                                        expected_euler_beam_damping_matrix):

        # setup global system
        global_system = GlobalSystem()
        global_system.is_rayleigh_damping = True
        global_system.damping_ratio = 0.0502
        global_system.radial_frequency_one = 2
        global_system.radial_frequency_two = 500

        # setup global matrices
        global_system.global_stiffness_matrix = euler_beam_stiffness_matrix
        global_system.global_mass_matrix = euler_beam_mass_matrix
        global_system.global_damping_matrix = np.zeros(euler_beam_mass_matrix.shape)

        # calculate damping matrix
        global_system.calculate_rayleigh_damping()

        # assert each value in matrix
        expected_damping_matrix = expected_euler_beam_damping_matrix
        for i in range(len(expected_damping_matrix)):
            for j in range(len(expected_damping_matrix[i])):
                assert global_system.global_damping_matrix[i, j] == pytest.approx(
                    expected_damping_matrix[i][j]
                )


    @pytest.fixture
    def euler_beam_stiffness_matrix(self):
        stiffness_matrix_euler_beam = [
            [2e7, 0, 0, -2e7, 0, 0],
            [0, 4.8e4, 2.4e5, 0, -4.8e4, 2.4e5],
            [0, 2.4e5, 1.6e6, 0, -2.4e5, 8e5],
            [-2e7, 0, 0, 2e7, 0, 0],
            [0, -4.8e4, -2.4e5, 0, 4.8e4, -2.4e5],
            [0, 2.4e5, 8e5, 0, -2.4e5, 1.6e6],
        ]

        stiffness_matrix_euler_beam = np.array([np.array(row) for row in stiffness_matrix_euler_beam])

        return stiffness_matrix_euler_beam

    @pytest.fixture
    def euler_beam_mass_matrix(self):
        mass_matrix_euler_beam = [
            [26.66666667, 0, 0, 13.33333333, 0, 0],
            [0.00000000, 29.71428571, 41.90476190, 0.00000000, 10.28571429, -24.76190476],
            [0.00000000, 41.90476190, 76.19047619, 0.00000000, 24.76190476, -57.14285714],
            [13.33333333, 0.00000000, 0.00000000, 26.66666667, 0.00000000, 0.00000000],
            [0.00000000, 10.28571429, 24.76190476, 0.00000000, 29.71428571, -41.90476190],
            [0.00000000, -24.76190476, -57.14285714, 0.00000000, -41.90476190, 76.19047619]
        ]

        mass_matrix_euler_beam = np.array([np.array(row) for row in mass_matrix_euler_beam])
        return mass_matrix_euler_beam

    @pytest.fixture
    def expected_euler_beam_damping_matrix(self):
        expected_damping_matrix = [
            [4005.33333333, 0.00000000, 0.00000000, -3997.33333333, 0.00000000, 0.00000000],
            [0.00000000, 15.54285714, 56.38095238, 0.00000000, -7.54285714, 43.04761905],
            [0.00000000, 56.38095238, 335.23809524, 0.00000000, -43.04761905, 148.57142857],
            [- 3997.33333333, 0.00000000, 0.00000000, 4005.33333333, 0.00000000, 0.00000000],
            [0.00000000, -7.54285714, -43.04761905, 0.00000000, 15.54285714, -56.38095238],
            [0.00000000, 43.04761905, 148.57142857, 0.00000000, -56.38095238, 335.23809524]
        ]
        return expected_damping_matrix