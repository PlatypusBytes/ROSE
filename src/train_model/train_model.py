import numpy as np
from scipy import sparse
from src import utils, geometry
import time
from src.model_part import ElementModelPart

class TrainModel(ElementModelPart):
    def __init__(self):
        super().__init__()
        self.mass_cart = None
        self.inertia_cart = None
        self.mass_bogie = None
        self.inertia_bogie = None
        self.mass_wheel = None

        self.sec_stiffness = None
        self.prim_stiffness = None

        self.sec_damping = None
        self.prim_damping = None

        self.length_cart = None
        self.length_bogie = None

    def set_aux_mass_matrix(self):
        """
        Set mass matrix of train
        :return:
        """
        self.aux_mass_matrix = np.zeros((14, 14))

        self.aux_mass_matrix[0,0] = self.mass_cart
        self.aux_mass_matrix[1,1] = self.inertia_cart
        self.aux_mass_matrix[[2, 4],[2,4]] = self.mass_bogie
        self.aux_mass_matrix[[3, 5],[3, 5]] = self.inertia_bogie
        self.aux_mass_matrix[[6, 8, 10, 12],[6, 8, 10, 12]] = self.mass_wheel

    def set_aux_stiffness_matrix(self):
        """
        Set stiffness matrix of train
        Note that wheels have an added rotational dof with 0 input, this required for adding the train aux matrix
        to the global system.
        :return:
        """
        self.aux_stiffness_matrix = np.zeros((14, 14))

        # cart displacement
        self.aux_stiffness_matrix[0, 0] = 2 * self.sec_stiffness
        self.aux_stiffness_matrix[[0, 0, 2, 4], [2, 4, 0, 0]] = -self.sec_stiffness

        # cart rotation
        self.aux_stiffness_matrix[1, 1] = 2 * self.sec_stiffness * self.length_cart**2
        self.aux_stiffness_matrix[[1, 2], [2, 1]] = -self.sec_stiffness * self.length_cart
        self.aux_stiffness_matrix[[1, 4], [4, 1]] = self.sec_stiffness * self.length_cart

        # bogie 1 displacement
        self.aux_stiffness_matrix[2, 2] = self.sec_stiffness + 2 * self.prim_stiffness
        self.aux_stiffness_matrix[[2, 2, 6, 8], [6, 8, 2, 2]] = -self.prim_stiffness

        # bogie 1 rotation
        self.aux_stiffness_matrix[3, 3] = 2 * self.prim_stiffness * self.length_bogie**2
        self.aux_stiffness_matrix[[3, 6], [6, 3]] = -self.prim_stiffness * self.length_bogie
        self.aux_stiffness_matrix[[3, 8], [8, 3]] = self.prim_stiffness * self.length_bogie

        # bogie 2 displacement
        self.aux_stiffness_matrix[4, 4] = self.sec_stiffness + 2 * self.prim_stiffness
        self.aux_stiffness_matrix[[4, 4, 10, 12], [10, 12, 4, 4]] = -self.prim_stiffness

        # bogie 2 rotation
        self.aux_stiffness_matrix[5, 5] = 2 * self.prim_stiffness * self.length_bogie ** 2
        self.aux_stiffness_matrix[[5, 10], [10, 5]] = -self.prim_stiffness * self.length_bogie
        self.aux_stiffness_matrix[[5, 12], [12, 5]] = self.prim_stiffness * self.length_bogie

        # wheels
        self.aux_stiffness_matrix[[6,8,10,12], [6,8,10,12]] = self.prim_stiffness


    def set_aux_damping_matrix(self):
        """
        Set damping matrix of train
        Note that wheels have an added rotational dof with 0 input, this required for adding the train aux matrix
        to the global system.
        :return:
        """
        self.aux_damping_matrix = np.zeros((14, 14))

        # cart displacement
        self.aux_damping_matrix[0, 0] = 2 * self.sec_damping
        self.aux_damping_matrix[[0, 0, 2, 4], [2, 4, 0, 0]] = -self.sec_damping

        # cart rotation
        self.aux_damping_matrix[1, 1] = 2 * self.sec_damping * self.length_cart ** 2
        self.aux_damping_matrix[[1, 2], [2, 1]] = -self.sec_damping * self.length_cart
        self.aux_damping_matrix[[1, 4], [4, 1]] = self.sec_damping * self.length_cart

        # bogie 1 displacement
        self.aux_damping_matrix[2, 2] = self.sec_damping + 2 * self.prim_damping
        self.aux_damping_matrix[[2, 2, 6, 8], [6, 8, 2, 2]] = -self.prim_damping

        # bogie 1 rotation
        self.aux_damping_matrix[3, 3] = 2 * self.prim_damping * self.length_bogie ** 2
        self.aux_damping_matrix[[3, 6], [6, 3]] = -self.prim_damping * self.length_bogie
        self.aux_damping_matrix[[3, 8], [8, 3]] = self.prim_damping * self.length_bogie

        # bogie 2 displacement
        self.aux_damping_matrix[4, 4] = self.sec_damping + 2 * self.prim_damping
        self.aux_damping_matrix[[4, 4, 10, 12], [10, 12, 4, 4]] = -self.prim_damping

        # bogie 2 rotation
        self.aux_damping_matrix[5, 5] = 2 * self.prim_damping * self.length_bogie ** 2
        self.aux_damping_matrix[[5, 10], [10, 5]] = -self.prim_damping * self.length_bogie
        self.aux_damping_matrix[[5, 12], [12, 5]] = self.prim_damping * self.length_bogie

        # wheels
        self.aux_damping_matrix[[6, 8, 10, 12], [6, 8, 10, 12]] = self.prim_damping

