from __future__ import annotations
import numpy as np


class Node:
    def __init__(self,x,y,z):
        self.index = None
        self.index_dof = np.array([None, None, None])
        self.coordinates = [x, y, z]
        self.normal_dof = False
        self.z_rot_dof = False
        self.y_disp_dof = False

        self.model_parts = []

    def __eq__(self, other: Node):
        abs_tol = 1e-9
        for idx, coordinate in enumerate(self.coordinates):
            if abs(coordinate - other.coordinates[idx]) > abs_tol:
                return False

        self.index_dof = [other.index_dof[idx] if other.index_dof[idx] is not None else self.index_dof[idx] for idx in
                          range(len(self.index_dof))]
        self.normal_dof = self.normal_dof + other.normal_dof
        self.z_rot_dof = self.z_rot_dof + other.z_rot_dof
        self.y_disp_dof = self.y_disp_dof + other.y_disp_dof
        self.model_parts = self.model_parts + other.model_parts
        return True

    @property
    def ndof(self):
        return self.normal_dof + self.z_rot_dof + self.y_disp_dof

class Element:
    def __init__(self, nodes):
        self.index = None
        self.nodes = nodes
        self.model_parts = []

    def __eq__(self, other):
        for node in self.nodes:
            if node not in other.nodes:
                return False

        self.model_parts = self.model_parts + other.model_parts

        return True

    def add_model_part(self, model_part):
        self.model_parts.append(model_part)
        for node in self.nodes:
            node.model_parts.append(model_part)

class Mesh:
    def __init__(self):
        self.nodes = np.array([])
        self.elements = np.array([])

    def reorder_node_ids(self):
        for idx, node in enumerate(self.nodes):
            node.index = idx

    def reorder_element_ids(self):
        for idx, element in enumerate(self.elements):
            element.index = idx

    def add_unique_nodes_to_mesh(self, nodes):
        for node in nodes:
            if node not in self.nodes:
                self.nodes = np.append(self.nodes, [node])

    def add_unique_elements_to_mesh(self, elements):
        for element in elements:
            if element not in self.elements:
                self.elements = np.append(self.elements, element)
