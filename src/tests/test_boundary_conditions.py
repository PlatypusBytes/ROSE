# from src.geometry import Node, Element
# from src.boundary_conditions import LoadCondition
#
# from scipy import sparse
# import numpy as np
# import pytest
#
# from src.model_part import Section
# from src.track import Rail
#
#
# class TestBoundaryConditions:
#     def test_moving_load(self):
#
#         # set geometry
#         nodes_track = [Node(0.0, 0.0, 0.0), Node(1.0, 0.0, 0.0), Node(2.0, 0.0, 0.0)]
#         elements_track = [
#             Element([nodes_track[0], nodes_track[1]]),
#             Element([nodes_track[1], nodes_track[2]]),
#         ]
#         section = Sec
#
#         rail = Rail()
#         rail.elements = elements_track
#         rail.nodes = nodes_track
#
#         rail.section
#         rail.initialize()
#
#         # set time
#         time = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
#
#         # set coordinate of moving load per timestep
#         moving_coords = np.array(
#             [
#                 Node(0.0, 0.0, 0.0),
#                 Node(0.25, 0.0, 0.0),
#                 Node(0.5, 0.0, 0.0),
#                 Node(0.75, 0.0, 0.0),
#                 Node(1, 0.0, 0.0),
#                 Node(1.25, 0.0, 0.0),
#                 Node(1.5, 0.0, 0.0),
#             ]
#         )
#
#         # set moving load
#         force = LoadCondition(y_disp_dof=True)
#         force.nodes = nodes_track
#         force.elements = elements_track
#
#         force.y_force = sparse.lil_matrix((len(nodes_track), len(time)))
#
#         coordinates = np.
#         force.set_moving_point_load(
#             moving_coords, time, y_force=np.array([1, 1, 1, 1, 1, 1, 1])
#         )
#
#         rail_model_part,
#         moving_coords,
#         time,
#         element_idxs=element_idxs,
#         normal_force=moving_normal_force,
#         y_force=moving_y_force,
#         z_moment=moving_z_moment,
#
#             model_part,
#             coordinates,
#             time,
#             element_idxs=None,
#             normal_force=None,
#             y_force=None,
#             z_moment=None,
#
#         # set expected values
#         expected_y_force_matrix = [
#             [1, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0],
#             [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5],
#         ]
#
#         # assert each value in force matrix
#         for i in range(len(expected_y_force_matrix)):
#             for j in range(len(expected_y_force_matrix[i])):
#                 assert force.y_force[i, j] == pytest.approx(
#                     expected_y_force_matrix[i][j]
#                 )
#
#
#
# @pytest.fixture
# def set_up_material():
#     # Steel
#     material = Material()
#     material.youngs_modulus = 200e9  # Pa
#     material.poisson_ratio = 0.0
#     material.density = 8000
#     return material
#
#
# @pytest.fixture
# def set_up_euler_section():
#     section = Section()
#     section.area = 1e-3
#     section.sec_moment_of_inertia = 2e-5
#     section.shear_factor = 0
#     return section
#
#
# @pytest.fixture
# def set_up_euler_beam(set_up_material, set_up_euler_section):
#     nodes = [Node(0, 0, 0), Node(10, 0, 0)]
#     elements = [Element([nodes])]
#
#     beam = TimoshenkoBeamElementModelPart()
#     beam.nodes = nodes
#     beam.elements = elements
#     beam.section = set_up_euler_section
#     beam.material = set_up_material
#
#     beam.length_element = 10
#
#     beam.calculate_mass()
#
#     beam.damping_ratio = 0.0502
#     beam.radial_frequency_one = 2
#     beam.radial_frequency_two = 500
#     return beam
#
#
# @pytest.fixture
# def expected_euler_beam_stiffness_matrix():
#     expected_stiffness_matrix = [
#         [2e7, 0, 0, -2e7, 0, 0],
#         [0, 4.8e4, 2.4e5, 0, -4.8e4, 2.4e5],
#         [0, 2.4e5, 1.6e6, 0, -2.4e5, 8e5],
#         [-2e7, 0, 0, 2e7, 0, 0],
#         [0, -4.8e4, -2.4e5, 0, 4.8e4, -2.4e5],
#         [0, 2.4e5, 8e5, 0, -2.4e5, 1.6e6],
#     ]
#     return expected_stiffness_matrix
#
#
# @pytest.fixture
# def expected_euler_beam_mass_matrix():
#     expected_mass_matrix = [
#         [26.66666667, 0, 0, 13.33333333, 0, 0],
#         [0, 29.73348571, 41.9207619, 0, 10.26651429, -24.74590476],
#         [0, 41.9207619, 76.40380952, 0, 24.74590476, -57.19619048],
#         [13.33333333, 0, 0, 26.66666667, 0, 0],
#         [0, 10.26651429, 24.74590476, 0, 29.73348571, -41.9207619],
#         [0, -24.74590476, -57.19619048, 0, -41.9207619, 76.40380952],
#     ]
#     return expected_mass_matrix
#
#
# @pytest.fixture
# def expected_euler_beam_damping_matrix():
#     expected_damping_matrix = [
#         [4005.33333333, 0.00000000, 0.00000000, -3997.33333333, 0.00000000, 0.00000000],
#         [0.00000000, 15.54669714, 56.38415238, 0.00000000, -7.54669714, 43.05081905],
#         [0.00000000, 56.38415238, 335.28076190, 0.00000000, -43.05081905, 148.56076190],
#         [-3997.33333333, 0.00000000, 0.00000000, 4005.33333333, 0.00000000, 0.00000000],
#         [0.00000000, -7.54669714, -43.05081905, 0.00000000, 15.54669714, -56.38415238],
#         [0.00000000, 43.05081905, 148.56076190, 0.00000000, -56.38415238, 335.28076190],
#     ]
#     return expected_damping_matrix
#
