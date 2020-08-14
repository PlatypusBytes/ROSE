import numpy as np
from scipy import sparse
from src import utils, geometry
from src.geometry import Node, Element, Mesh
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

        self.herzian_contact_cof = None

        self.irregularities = None
        self.static_wheel_load = None
        self.static_wheel_deformation = None

        self.force_vector = None
        self.static_force_vector = None

        self.g = 9.81
        self.velocity = None

        self.deformation_wheels = np.zeros((4, 1))
        self.deformation_track = np.zeros((4, 1))
        self.irregularities = np.zeros(4)

    def set_aux_mass_matrix(self):
        """
        Set mass matrix of train
        :return:
        """
        self.aux_mass_matrix = np.zeros((14, 14))

        self.aux_mass_matrix[0, 0] = self.mass_cart
        self.aux_mass_matrix[1, 1] = self.inertia_cart
        self.aux_mass_matrix[[2, 4], [2, 4]] = self.mass_bogie
        self.aux_mass_matrix[[3, 5], [3, 5]] = self.inertia_bogie
        self.aux_mass_matrix[[6, 8, 10, 12], [6, 8, 10, 12]] = self.mass_wheel

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
        self.aux_stiffness_matrix[1, 1] = 2 * self.sec_stiffness * self.length_cart ** 2
        self.aux_stiffness_matrix[[1, 2], [2, 1]] = (
            -self.sec_stiffness * self.length_cart
        )
        self.aux_stiffness_matrix[[1, 4], [4, 1]] = (
            self.sec_stiffness * self.length_cart
        )

        # bogie 1 displacement
        self.aux_stiffness_matrix[2, 2] = self.sec_stiffness + 2 * self.prim_stiffness
        self.aux_stiffness_matrix[[2, 2, 6, 8], [6, 8, 2, 2]] = -self.prim_stiffness

        # bogie 1 rotation
        self.aux_stiffness_matrix[3, 3] = (
            2 * self.prim_stiffness * self.length_bogie ** 2
        )
        self.aux_stiffness_matrix[[3, 6], [6, 3]] = (
            -self.prim_stiffness * self.length_bogie
        )
        self.aux_stiffness_matrix[[3, 8], [8, 3]] = (
            self.prim_stiffness * self.length_bogie
        )

        # bogie 2 displacement
        self.aux_stiffness_matrix[4, 4] = self.sec_stiffness + 2 * self.prim_stiffness
        self.aux_stiffness_matrix[[4, 4, 10, 12], [10, 12, 4, 4]] = -self.prim_stiffness

        # bogie 2 rotation
        self.aux_stiffness_matrix[5, 5] = (
            2 * self.prim_stiffness * self.length_bogie ** 2
        )
        self.aux_stiffness_matrix[[5, 10], [10, 5]] = (
            -self.prim_stiffness * self.length_bogie
        )
        self.aux_stiffness_matrix[[5, 12], [12, 5]] = (
            self.prim_stiffness * self.length_bogie
        )

        # wheels
        self.aux_stiffness_matrix[[6, 8, 10, 12], [6, 8, 10, 12]] = self.prim_stiffness

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
        self.aux_damping_matrix[[5, 10], [10, 5]] = (
            -self.prim_damping * self.length_bogie
        )
        self.aux_damping_matrix[[5, 12], [12, 5]] = (
            self.prim_damping * self.length_bogie
        )

        # wheels
        self.aux_damping_matrix[[6, 8, 10, 12], [6, 8, 10, 12]] = self.prim_damping

    def calculate_static_wheel_deformation(self):
        pass
        # self.static_wheel_deformation = self.herzian_contact_cof * self.static_wheel_load ** (2/3)

    def __calculate_elastic_wheel_deformation(self):
        pass
        elastic_wheel_deformation = (
            self.static_wheel_deformation
            + self.deformation_wheels
            - self.deformation_track
            - self.irregularities
        )
        return elastic_wheel_deformation

    def calculate_wheel_rail_contact_force(self):
        pass
        contact_force = (
            1 / self.herzian_contact_cof * self.__calculate_elastic_wheel_deformation()
        ) ** (3 / 2)
        return contact_force

    def set_static_force_vector(self):
        self.static_force_vector = np.zeros((14, 1))
        self.static_force_vector[0, 0] = -self.mass_cart * self.g
        self.static_force_vector[2, 0] = -self.mass_bogie * self.g
        self.static_force_vector[4, 0] = -self.mass_bogie * self.g
        self.static_force_vector[[6, 8, 10, 12], [0, 0, 0, 0]] = (
            -self.mass_wheel * self.g
        )

    def set_dynamic_force_vector(self):

        self.dynamic_forve_vector = np.zeros((14, 1))
        self.dynamic_forve_vector[
            [6, 8, 10, 12], [0, 0, 0, 0]
        ] = self.calculate_wheel_rail_contact_force()

    def set_force_vector(self):
        self.force_vector = self.static_force_vector + self.set_dynamic_force_vector()

    def create_cart(self, center_x=0):
        cart_nodes = [
            Node(center_x - 0.5 * self.length_cart, 2, 0),
            Node(center_x + 0.5 * self.length_cart, 2, 0),
        ]
        cart_elements = [Element([cart_nodes[0], cart_nodes[1]])]
        for node in cart_nodes:
            node.z_rot_dof = True
            node.y_disp_dof = True

        bogie_1_nodes = [
            Node(cart_nodes[0].coordinates[0] - 0.5 * self.length_bogie, 1, 0),
            Node(cart_nodes[0].coordinates[0] + 0.5 * self.length_bogie, 1, 0),
        ]
        bogie_1_elements = [Element([bogie_1_nodes[0], bogie_1_nodes[1]])]

        for node in bogie_1_nodes:
            node.z_rot_dof = True
            node.y_disp_dof = True

        bogie_2_nodes = [
            Node(cart_nodes[1].coordinates[0] - 0.5 * self.length_bogie, 1, 0),
            Node(cart_nodes[1].coordinates[0] + 0.5 * self.length_bogie, 1, 0),
        ]
        bogie_2_elements = [Element([bogie_2_nodes[0], bogie_2_nodes[1]])]

        for node in bogie_2_nodes:
            node.z_rot_dof = True
            node.y_disp_dof = True

        wheels_nodes = [
            Node(bogie_1_nodes[0].coordinates[0], 0, 0),
            Node(bogie_1_nodes[1].coordinates[0], 0, 0),
            Node(bogie_2_nodes[0].coordinates[0], 0, 0),
            Node(bogie_2_nodes[1].coordinates[0], 0, 0),
        ]
        wheels_elements = [
            Element([bogie_1_nodes[0]]),
            Element([bogie_1_nodes[1]]),
            Element([bogie_2_nodes[0]]),
            Element([bogie_2_nodes[1]]),
        ]

        for node in wheels_nodes:
            node.y_disp_dof = True

        mesh = Mesh()
        mesh.add_unique_nodes_to_mesh(cart_nodes)
        mesh.add_unique_nodes_to_mesh(bogie_1_nodes)
        mesh.add_unique_nodes_to_mesh(bogie_2_nodes)
        mesh.add_unique_nodes_to_mesh(wheels_nodes)

        mesh.add_unique_elements_to_mesh(cart_elements)
        mesh.add_unique_elements_to_mesh(bogie_1_elements)
        mesh.add_unique_elements_to_mesh(bogie_2_elements)
        mesh.add_unique_elements_to_mesh(wheels_elements)

        self.nodes = list(mesh.nodes)
        self.elements = list(mesh.elements)

    def initialize(self):
        super().initialize()
        self.set_static_force_vector()
        self.calculate_static_wheel_deformation()

    def update(self):
        super().update()
        self.set_force_vector()
