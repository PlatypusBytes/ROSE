from src.model_part import ConditionModelPart
from src.boundary_conditions import CauchyCondition
from src.geometry import Node, Element

from scipy import sparse
import numpy as np

class TestBoundaryConditions:
    def test_moving_load(self):

        nodes_track = [Node(0.0, 0.0, 0.0), Node(1.0, 0.0, 0.0), Node(2.0, 0.0, 0.0)]
        elements_track = [Element([nodes_track[0], nodes_track[1]]), Element([nodes_track[1], nodes_track[2]])]

        force = CauchyCondition(y_disp_dof=True)
        force.nodes = nodes_track
        force.elements = elements_track

        time = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        moving_coords = np.array([Node(0.0, 0.0, 0.0), Node(0.25, 0.0, 0.0),Node(0.5, 0.0, 0.0),
                                  Node(0.75, 0.0, 0.0),Node(1, 0.0, 0.0),Node(1.25, 0.0, 0.0),Node(2.5, 0.0, 0.0)])

        force.y_force = sparse.csr_matrix((len(nodes_track), len(time)))
        force.set_moving_point_load(moving_coords,time, y_force=np.array([1, 1, 1, 1, 1, 1, 1]))


