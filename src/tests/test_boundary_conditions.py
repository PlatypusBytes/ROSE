from src.geometry import Node, Element
from src.boundary_conditions import LoadCondition

from scipy import sparse
import numpy as np
import pytest


class TestBoundaryConditions:
    def test_moving_load(self):

        # set geometry
        nodes_track = [Node(0.0, 0.0, 0.0), Node(1.0, 0.0, 0.0), Node(2.0, 0.0, 0.0)]
        elements_track = [Element([nodes_track[0], nodes_track[1]]), Element([nodes_track[1], nodes_track[2]])]

        # set time
        time = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # set coordinate of moving load per timestep
        moving_coords = np.array([Node(0.0, 0.0, 0.0), Node(0.25, 0.0, 0.0), Node(0.5, 0.0, 0.0),
                                  Node(0.75, 0.0, 0.0), Node(1, 0.0, 0.0), Node(1.25, 0.0, 0.0), Node(1.5, 0.0, 0.0)])

        # set moving load
        force = LoadCondition(y_disp_dof=True)
        force.nodes = nodes_track
        force.elements = elements_track

        force.y_force = sparse.lil_matrix((len(nodes_track), len(time)))
        force.set_moving_point_load(moving_coords, time, y_force=np.array([1, 1, 1, 1, 1, 1, 1]))

        # set expected values
        expected_y_force_matrix = [[1, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0],
                                   [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5]]

        # assert each value in force matrix
        for i in range(len(expected_y_force_matrix)):
            for j in range(len(expected_y_force_matrix[i])):
                assert force.y_force[i, j] == pytest.approx(expected_y_force_matrix[i][j])


