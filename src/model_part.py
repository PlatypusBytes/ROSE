
import numpy as np

class ModelPart:
    """
    The model part has to consist of the same element types
    """
    def __init__(self):
        self.name = ""
        self.nodes = np.array([])
        self.elements = np.array([])
        self.rotation_dof = False
        self.x_disp_dof = False
        self.y_disp_dof = False

        self.aux_stiffness_matrix = []
        self.aux_damping_matrix = []
        self.aux_mass_matrix = []

    def initialize(self):
        self.set_aux_stiffness_matrix()
        self.set_aux_damping_matrix()
        self.set_aux_mass_matrix()

    def set_geometry(self):
        pass

    def set_aux_stiffness_matrix(self):
        pass

    def set_aux_damping_matrix(self):
        pass

    def set_aux_mass_matrix(self):
        pass




