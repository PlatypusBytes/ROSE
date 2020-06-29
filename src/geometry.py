import numpy as np

class Node:
    def __init__(self):
        self.index = None
        self.index_dof = np.array([None, None, None])
        self.coordinates = []
        self.rotation_dof = False
        self.x_disp_dof = False
        self.y_disp_dof = False

        self.model_parts = []

    def __eq__(self, other):
        abs_tol = 1e-9
        for idx, coordinate in self.coordinates:
            if abs(coordinate - other.coordinates[idx]) > abs_tol:
                return False

        self.index_dof = [other.index_dof[idx] if other.index_dof[idx] is not None else self.index_dof[idx] for idx in
                          range(len(self.index_dof))]
        self.rotation_dof = self.rotation_dof + other.rotation_dof
        self.x_disp_dof = self.x_disp_dof + other.x_disp_dof
        self.y_disp_dof = self.y_disp_dof + other.y_disp_dof
        self.model_parts = self.model_parts + other.model_parts
        return True

    @property
    def ndof(self):
        return self.rotation_dof + self.x_disp_dof + self.y_disp_dof

    # def merge_if_equal(self, other):
    #     if other == self:
    #         self.index_dof = [other.index_dof[idx] if other.index_dof[idx] is not None else self.index_dof[idx] for idx in range(len(self.index_dof))]
    #         self.rotation_dof = self.rotation_dof + other.rotation_dof
    #         self.x_disp_dof = self.x_disp_dof + other.x_disp_dof
    #         self.y_disp_dof = self.y_disp_dof + other.y_disp_dof
    #         self.model_parts = self.model_parts + other.model_parts

class Element:
    def __init__(self):
        self.index = None
        self.nodes = None
        self.model_parts = []

    def __eq__(self, other):
        for node in self.nodes:
            if node not in other:
                return False

        self.model_parts = self.model_parts + other.model_parts

        return True

    def add_model_part(self, model_part):
        self.model_parts.append(model_part)
        for node in self.nodes:
            node.model_parts.append(model_part)

    # def merge_if_equal(self, other):
    #     if self == other:
    #         self.model_parts = self.model_parts + other.model_parts