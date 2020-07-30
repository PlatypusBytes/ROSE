import numpy as np
import pygmsh
from src.geometry import Node, Element
from src import utils
import copy
from scipy import sparse
from src.model_part import ElementModelPart

class Soil(ElementModelPart):
    def __init__(self):
        super(Soil, self).__init__()

        self.stiffness = None
        self.damping = None

        self.aux_stiffness_matrix = None
        self.aux_damping_matrix = None
        self.aux_mass_matrix = None

        self.normal_dof = False
        self.z_rot_dof = False
        self.y_disp_dof = True

        self.nodal_ndof = 1

    def set_aux_stiffness_matrix(self):
        self.aux_stiffness_matrix = np.zeros((2, 2))
        self.aux_stiffness_matrix[0, 0] = self.stiffness
        self.aux_stiffness_matrix[1, 0] = -self.stiffness
        self.aux_stiffness_matrix[0, 1] = -self.stiffness
        self.aux_stiffness_matrix[1, 1] = self.stiffness

    def set_aux_damping_matrix(self):
        self.aux_damping_matrix = np.zeros((2, 2))
        self.aux_damping_matrix[0, 0] = self.damping
        self.aux_damping_matrix[1, 0] = -self.damping
        self.aux_damping_matrix[0, 1] = -self.damping
        self.aux_damping_matrix[1, 1] = self.damping


    def set_1_d_geometry(self, top_nodes, bottom_nodes):
        self.nodes = np.append(self.nodes, top_nodes)
        self.nodes = np.append(self.nodes, bottom_nodes)

        soil_elements = []
        for i in range(len(top_nodes)):
            element = Element()
            element.index = len(self.elements) + i
            element.nodes = [top_nodes[i], bottom_nodes[i]]
            element.add_model_part("SOIL")
            soil_elements.append(element)

        self.elements = np.append(self.elements, soil_elements)






