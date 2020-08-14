import numpy as np
from scipy import sparse
from src import utils, geometry
import time
from src.model_part import ElementModelPart


class Material:
    def __init__(self):
        self.youngs_modulus = None
        self.poisson_ratio = None
        self.density = None

    @property
    def shear_modulus(self):
        return self.youngs_modulus / (2 * (1 + self.poisson_ratio))


class Section:
    def __init__(self):
        self.area = None  # [m^2]
        self.sec_moment_of_inertia = None  # [m^4]
        self.shear_factor = (
            0  # shear factor (kr=0 - Euler-Bernoulli beam, kr>0 - Timoshenko beam)
        )


class Rail(ElementModelPart):
    def __init__(self):
        super().__init__()

        self.material = Material()
        self.section = Section()

        self.length_rail = None
        self.mass = None

        self.timoshenko_factor = 0  # ???
        self.ndof = None

        self.damping_ratio = None
        self.radial_frequency_one = None
        self.radial_frequency_two = None

        self.aux_mass_matrix = None
        self.aux_stiffness_matrix = None
        self.aux_damping_matrix = None

        self.aux_force_vector = None

        self.nodal_ndof = 3

        self.normal_dof = True
        self.z_rot_dof = True
        self.y_disp_dof = True

    def calculate_length_rail(self):

        xdiff = np.diff([node.coordinates[0] for node in self.nodes])
        ydiff = np.diff([node.coordinates[1] for node in self.nodes])
        zdiff = np.diff([node.coordinates[2] for node in self.nodes])

        distances = np.sqrt(np.square(xdiff) + np.square(ydiff) + np.square(zdiff))

        if not np.all(np.isclose(distances[0], distances)):
            print("distance between sleepers is not equal")
            raise

        self.length_rail = distances[0]

    def calculate_mass(self):
        self.mass = self.section.area * self.material.density

    def calculate_timoshenko_factor(self):
        if self.section.shear_factor > 0:
            self.timoshenko_factor = (
                12
                * self.material.youngs_modulus
                * self.section.sec_moment_of_inertia
                / (
                    self.length_rail ** 2
                    * self.material.shear_modulus
                    * self.section.area
                    * self.section.shear_factor
                )
            )

    def calculate_n_dof(self):
        # self.n_nodes = self.__n_sleepers
        self.ndof = len(self.nodes) * self.nodal_ndof

    def __set_translational_aux_mass_matrix(self):
        """
        timoshenko straight beam auxiliar mass matrix associated with translational inertia
        :return:
        """
        phi = self.timoshenko_factor
        l = self.length_rail

        constant = (
            self.material.density * self.section.area * l / (210 * (1 + phi) ** 2)
        )

        if self.nodal_ndof == 3:
            trans_aux_mass_matrix = np.zeros((6, 6))

            trans_aux_mass_matrix[[0, 3], [0, 3]] = 70 * (1 + phi) ** 2
            trans_aux_mass_matrix[[3, 0], [0, 3]] = 35 * (1 + phi) ** 2

            trans_aux_mass_matrix[[1, 4], [1, 4]] = 70 * phi ** 2 + 147 * phi + 78
            trans_aux_mass_matrix[[1, 4], [4, 1]] = 35 * phi ** 2 + 63 * phi + 27

            trans_aux_mass_matrix[[1, 2], [2, 1]] = (
                (35 * phi ** 2 + 77 * phi + 44) * l / 4
            )

            trans_aux_mass_matrix[[1, 5], [5, 1]] = (
                -(35 * phi ** 2 + 63 * phi + 26) * l / 4
            )

            trans_aux_mass_matrix[[2, 5], [2, 5]] = (7 * phi ** 2 + 14 * phi + 8) * (
                l ** 2 / 4
            )
            trans_aux_mass_matrix[[2, 5], [5, 2]] = -(7 * phi ** 2 + 14 * phi + 6) * (
                l ** 2 / 4
            )

            trans_aux_mass_matrix[[2, 4], [4, 2]] = (35 * phi ** 2 + 63 * phi + 26) * (
                l / 4
            )

            trans_aux_mass_matrix[[4, 5], [5, 4]] = -(35 * phi ** 2 + 77 * phi + 44) * (
                l / 4
            )

            trans_aux_mass_matrix = utils.reshape_aux_matrix(
                2, [True, True, True], trans_aux_mass_matrix
            )
            return trans_aux_mass_matrix.dot(constant)
        return None

    def __set_rotational_aux_mass_matrix(self):
        """
        timoshenko straight beam auxiliar mass matrix associated with rotatory inertia
        :return:
        """
        phi = self.timoshenko_factor
        l = self.length_rail

        constant = (
            self.material.density
            * self.section.sec_moment_of_inertia
            / (30 * (1 + phi) ** 2 * l)
        )

        if self.nodal_ndof == 3:
            rot_aux_mass_matrix = np.zeros((6, 6))

            rot_aux_mass_matrix[[1, 4], [1, 4]] = 36
            rot_aux_mass_matrix[[1, 4], [4, 1]] = -36

            rot_aux_mass_matrix[[1, 1, 2, 5], [2, 5, 1, 1]] = -(15 * phi - 3) * l

            rot_aux_mass_matrix[[2, 5], [2, 5]] = (10 * phi ** 2 + 5 * phi + 4) * l ** 2
            rot_aux_mass_matrix[[2, 5], [5, 2]] = (5 * phi ** 2 - 5 * phi - 1) * l ** 2

            rot_aux_mass_matrix[[2, 4, 4, 5], [4, 2, 5, 4]] = (15 * phi - 3) * l

            rot_aux_mass_matrix = utils.reshape_aux_matrix(
                2, [True, True, True], rot_aux_mass_matrix
            )
            return rot_aux_mass_matrix.dot(constant)
        return None

    def set_aux_mass_matrix(self):
        """
        timoshenko straight beam auxiliar mass matrix
        :return:
        """
        self.aux_mass_matrix = (
            self.__set_translational_aux_mass_matrix()
            + self.__set_rotational_aux_mass_matrix()
        )

    def set_aux_stiffness_matrix(self):
        """
        timoshenko straight beam auxiliar stiffness matrix
        :return:
        """
        EI = self.material.youngs_modulus * self.section.sec_moment_of_inertia
        constant = EI / ((1 + self.timoshenko_factor) * self.length_rail ** 3)

        if self.nodal_ndof == 3:
            self.aux_stiffness_matrix = np.zeros((6, 6))
            self.aux_stiffness_matrix[[0, 3], [0, 3]] = (
                self.section.area
                / self.section.sec_moment_of_inertia
                * (1 + self.timoshenko_factor)
                * self.length_rail ** 2
            )
            self.aux_stiffness_matrix[[3, 0], [0, 3]] = (
                -self.section.area
                / self.section.sec_moment_of_inertia
                * (1 + self.timoshenko_factor)
                * self.length_rail ** 2
            )

            self.aux_stiffness_matrix[[1, 4], [1, 4]] = 12
            self.aux_stiffness_matrix[[1, 4], [4, 1]] = -12

            self.aux_stiffness_matrix[[1, 1, 2, 5], [2, 5, 1, 1]] = 6 * self.length_rail
            self.aux_stiffness_matrix[[2, 4, 4, 5], [4, 2, 5, 4]] = (
                -6 * self.length_rail
            )

            self.aux_stiffness_matrix[[2, 5], [2, 5]] = (
                4 + self.timoshenko_factor
            ) * self.length_rail ** 2
            self.aux_stiffness_matrix[[2, 5], [5, 2]] = (
                2 - self.timoshenko_factor
            ) * self.length_rail ** 2

            self.aux_stiffness_matrix = self.aux_stiffness_matrix.dot(constant)
            self.aux_stiffness_matrix = utils.reshape_aux_matrix(
                2, [True, True, True], self.aux_stiffness_matrix
            )

    def __calculate_rayleigh_damping_factors(self):
        """
        Calculate rayleigh damping coefficients
        :return:
        """
        constant = (
            2
            * self.damping_ratio
            / (self.radial_frequency_one + self.radial_frequency_two)
        )
        a0 = self.radial_frequency_one * self.radial_frequency_two * constant
        a1 = constant
        return a0, a1

    def set_aux_damping_matrix(self):
        """
        Damping matrix is calculated with the assumption of Rayleigh damping
        :return:
        """
        a0, a1 = self.__calculate_rayleigh_damping_factors()
        self.aux_damping_matrix = self.aux_mass_matrix.dot(
            a0
        ) + self.aux_stiffness_matrix.dot(a1)

    def initialize_shape_functions(self):
        self._normal_shape_functions = np.zeros(2)
        self._y_shape_functions = np.zeros(4)
        self._z_rot_shape_functions = np.zeros(4)

    @property
    def normal_shape_functions(self):
        return self._normal_shape_functions

    @property
    def y_shape_functions(self):
        return self._y_shape_functions

    @property
    def z_rot_shape_functions(self):
        return self._z_rot_shape_functions

    def set_normal_shape_functions(self, x):
        self._normal_shape_functions[0] = 1 - x / self.length_rail
        self._normal_shape_functions[1] = x / self.length_rail

    def set_y_shape_functions(self, x):
        constant = 1 / (1 + self.timoshenko_factor)
        self._y_shape_functions[0] = constant * (
            1
            + 2 * (x / self.length_rail) ** 3
            - 3 * (x / self.length_rail) ** 2
            + self.timoshenko_factor * (1 - x / self.length_rail)
        )
        self._y_shape_functions[1] = constant * (
            x
            + (x ** 3 / self.length_rail ** 2)
            - 2 * (x ** 2 / self.length_rail)
            + self.timoshenko_factor
            / 2
            * (x / self.length_rail - (x / self.length_rail) ** 2)
        )
        self._y_shape_functions[2] = constant * (
            -2 * (x / self.length_rail) ** 3
            + 3 * (x / self.length_rail) ** 2
            + self.timoshenko_factor * (x / self.length_rail)
        )
        self._y_shape_functions[3] = constant * (
            (x ** 3 / self.length_rail ** 2)
            - ((x ** 2) / self.length_rail)
            + self.timoshenko_factor
            / 2
            * ((x / self.length_rail) ** 2 - (x / self.length_rail))
        )

    def set_z_rot_shape_functions(self, x):
        pass

    def initialize(self):
        self.calculate_timoshenko_factor()
        self.calculate_mass()
        self.calculate_n_dof()
        self.calculate_length_rail()

        self.initialize_shape_functions()
        super(Rail, self).initialize()


class Sleeper(ElementModelPart):
    def __init__(self):
        super().__init__()
        self.mass = None
        self.distance_between_sleepers = None
        self.height_sleeper = 0.1

        self.y_disp_dof = True

    def set_aux_stiffness_matrix(self):
        self.aux_stiffness_matrix = np.zeros((1, 1))

    def set_aux_damping_matrix(self):
        self.aux_damping_matrix = np.zeros((1, 1))

    def set_aux_mass_matrix(self):
        self.aux_mass_matrix = np.ones((1, 1)) * self.mass


class RailPad(ElementModelPart):
    def __init__(self):
        super().__init__()
        self.stiffness = None
        self.damping = None
        self.aux_stiffness_matrix = None
        self.aux_damping_matrix = None

        self.y_disp_dof = True

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


class ContactRailWheel:
    def __init__(self):
        self.stiffness = None
        self.damping = None
        self.exponent = None


class ContactSleeperBallast:
    def __init__(self):
        self.stiffness = None
        self.damping = None


class Support:
    def __init__(self, n_sleepers):
        self.linear_stiffness = None
        self.non_linear_stiffness = None
        self.non_linear_exponent = None
        self.initial_voids = None
        self.tensile_stiffness_ballast = None
        self.damping_ratio = None
        self.__n_sleepers = n_sleepers

        self.linear_stiffness_matrix = None
        self.non_linear_stiffness_matrix = None
        self.non_linear_exponent_matrix = None
        self.initial_voids_matrix = None
        self.tensile_stiffness_ballast_matrix = None
        self.damping_ratio_matrix = None

    def initialise_matrices(self):
        self.linear_stiffness_matrix = (
            np.ones((1, self.__n_sleepers)) * self.linear_stiffness
        )
        self.non_linear_stiffness_matrix = (
            np.ones((1, self.__n_sleepers)) * self.non_linear_stiffness
        )
        self.non_linear_exponent_matrix = (
            np.ones((1, self.__n_sleepers)) * self.non_linear_exponent
        )
        self.initial_voids_matrix = np.ones((1, self.__n_sleepers)) * self.initial_voids
        self.tensile_stiffness_ballast_matrix = (
            np.ones((1, self.__n_sleepers)) * self.tensile_stiffness_ballast
        )
        self.damping_ratio_matrix = np.ones((1, self.__n_sleepers)) * self.damping_ratio


class Ballast:
    def __init__(self, n_sleepers):
        self.mass = None
        self.stiffness = None
        self.damping = None
        self.__n_sleepers = n_sleepers

        self.mass_matrix = None
        self.stiffness_matrix = None
        self.damping_matrix = None

    def initialise_matrices(self):
        self.mass_matrix = np.ones((1, self.__n_sleepers - 1)) * self.mass
        self.stiffness_matrix = np.ones((1, self.__n_sleepers - 1)) * self.stiffness
        self.damping_matrix = np.ones((1, self.__n_sleepers - 1)) * self.damping


#
# class UTrack(ElementModelPart):
#     def __init__(self, n_sleepers):
#         super(UTrack, self).__init__()
#
#         self.__n_sleepers = n_sleepers
#
#         self.rail = Rail(n_sleepers)
#         self.sleeper = Sleeper()
#         self.rail_pads = RailPad()
#         self.ballast = Ballast(n_sleepers)
#         self.contact_sleeper_ballast = ContactSleeperBallast()
#         self.Support = Support(n_sleepers)
#         self.contact_rail_wheel = ContactRailWheel()
#
#         self.global_mass_matrix = None
#         self.global_stiffness_matrix = None
#         self.global_damping_matrix = None
#
#         self.force = None
#         self.time = np.linspace(0, 10, 1000)
#
#         self.__total_length = None
#         self.__n_dof_rail = None
#         self.n_dof_track = None
#
#
#     def __add_rail_to_geometry(self):
#
#         rail_nodes = []
#         for i in range(self.__n_sleepers):
#             node = geometry.Node()
#             node.index = len(self.nodes) + i
#             node.normal_dof = True
#             node.z_rot_dof = True
#             node.y_disp_dof = True
#             node.coordinates = self.rail.coordinates[i]
#             rail_nodes.append(node)
#
#         rail_elements = []
#         for i in range(len(rail_nodes) -1):
#             element = geometry.Element()
#             element.index = len(self.elements) + i
#             element.nodes = [rail_nodes[i], rail_nodes[i+1]]
#             element.add_model_part("RAIL")
#             rail_elements.append(element)
#
#         self.nodes = np.append(self.nodes, np.array(rail_nodes))
#         self.elements = np.append(self.elements, np.array(rail_elements))
#
#         return rail_nodes, rail_elements
#
#     def __add_rail_pads_to_geometry(self, rail_nodes):
#
#         rail_pad_nodes = []
#         for i in range(self.__n_sleepers):
#             node = geometry.Node()
#             node.index = len(self.nodes) + i
#             node.normal_dof = False
#             node.z_rot_dof = False
#             node.y_disp_dof = True
#             node.coordinates = np.array([self.rail.coordinates[i][0],
#                                          self.rail.coordinates[i][1]-self.sleeper.height_sleeper])
#             rail_pad_nodes.append(node)
#
#         rail_pad_elements = []
#         for i in range(len(rail_pad_nodes)):
#             element = geometry.Element()
#             element.index = len(self.elements) + i
#             element.nodes = [rail_nodes[i], rail_pad_nodes[i]]
#             element.add_model_part("RAIL_PAD")
#             rail_pad_elements.append(element)
#
#         self.nodes = np.append(self.nodes, np.array(rail_pad_nodes))
#         self.elements = np.append(self.elements, np.array(rail_pad_elements))
#
#         return rail_pad_nodes, rail_pad_elements
#
#     def set_geometry(self):
#         super(UTrack, self).set_geometry()
#         self.rail.calculate_coordinates()
#         rail_nodes, _ = self.__add_rail_to_geometry()
#         self.__add_rail_pads_to_geometry(rail_nodes)
#
#     def set_aux_stiffness_matrix(self):
#         super(UTrack, self).set_aux_stiffness_matrix()
#
#         self.rail.set_aux_stiffness_matrix()
#         self.rail_pads.set_aux_stiffness_matrix()
#
#     def set_aux_damping_matrix(self):
#         super(UTrack, self).set_aux_damping_matrix()
#         # self.rail.set_aux_damping_matrix()
#         self.rail_pads.set_aux_damping_matrix()
#
#     def set_aux_mass_matrix(self):
#         super(UTrack, self).set_aux_mass_matrix()
#         self.rail.set_aux_mass_matrix()
#
#         self.aux_mass_matrix = self.rail.aux_mass_matrix
#
#     def set_global_stiffness_matrix(self):
#         """
#
#         :return:
#         """
#         self.global_stiffness_matrix = sparse.csr_matrix((self.n_dof_track, self.n_dof_track))
#
#         rail_elements = [element for element in self.elements if "RAIL" in element.model_parts]
#         self.global_stiffness_matrix = utils.add_aux_matrix_to_global(
#             self.global_stiffness_matrix, self.rail.aux_stiffness_matrix, rail_elements)
#
#         rail_pad_elements = [element for element in self.elements if "RAIL_PAD" in element.model_parts]
#         self.global_stiffness_matrix = utils.add_aux_matrix_to_global(
#             self.global_stiffness_matrix, self.rail_pads.aux_stiffness_matrix, rail_pad_elements)
#
#     def __add_sleeper_to_global_mass_matrix(self):
#
#         sleeper_nodes = [node for node in self.nodes
#                         if "RAIL" not in node.model_parts and "RAIL_PAD" in node.model_parts]
#         sleeper_y_dof_indices = [node.index_dof[1] for node in sleeper_nodes]
#
#         self.global_mass_matrix[sleeper_y_dof_indices, sleeper_y_dof_indices] += self.sleeper.mass
#
#     def set_global_mass_matrix(self):
#         """
#         Set global mass matrix
#         :return:
#         """
#         self.global_mass_matrix = sparse.csr_matrix((self.n_dof_track, self.n_dof_track))
#
#         self.rail.set_aux_mass_matrix()
#
#         rail_elements = [element for element in self.elements if "RAIL" in element.model_parts]
#         self.global_mass_matrix = utils.add_aux_matrix_to_global(
#             self.global_mass_matrix, self.rail.aux_mass_matrix, rail_elements)
#
#         self.__add_sleeper_to_global_mass_matrix()
#
#     # def __add_rail_to_global_damping_matrix(self):
#     #     damping_matrix = sparse.csr_matrix((self.rail.ndof, self.rail.ndof))
#     #
#     #     ndof_node = self.rail.nodal_ndof
#     #     for i in range(self.rail.n_nodes - 1):
#     #         damping_matrix[i * ndof_node:i * ndof_node + ndof_node * 2, i * ndof_node:i * ndof_node + ndof_node * 2] \
#     #             += self.rail.aux_damping_matrix
#     #
#     #     self.global_damping_matrix[0:self.rail.ndof, 0:self.rail.ndof] \
#     #         += damping_matrix
#
#     # def __add_rail_pad_to_global_damping_matrix(self):
#     #
#     #     top_rail_pad_nodes = [node for node in self.nodes
#     #                     if "RAIL" in node.model_parts and "RAIL_PAD" in node.model_parts]
#     #
#     #     bot_rail_pad_nodes = [node for node in self.nodes
#     #                     if "RAIL" not in node.model_parts and "RAIL_PAD" in node.model_parts]
#     #
#     #     top_rail_pad_y_dof_indices = [node.index_dof[1] for node in top_rail_pad_nodes]
#     #     bot_rail_pad_y_dof_indices = [node.index_dof[1] for node in bot_rail_pad_nodes]
#     #
#     #     self.global_damping_matrix[top_rail_pad_y_dof_indices, top_rail_pad_y_dof_indices] \
#     #         += self.rail_pads.aux_damping_matrix[0, 0]
#     #     self.global_damping_matrix[bot_rail_pad_y_dof_indices, top_rail_pad_y_dof_indices] \
#     #         += self.rail_pads.aux_damping_matrix[1, 0]
#     #     self.global_damping_matrix[top_rail_pad_y_dof_indices, bot_rail_pad_y_dof_indices] \
#     #         += self.rail_pads.aux_damping_matrix[0, 1]
#     #     self.global_damping_matrix[bot_rail_pad_y_dof_indices, bot_rail_pad_y_dof_indices] \
#     #         += self.rail_pads.aux_damping_matrix[1, 1]
#
#     # def __add_soil_to_global_damping_matrix(self):
#     #     for i in range(self.__n_sleepers):
#     #         self.global_damping_matrix[i + self.rail.ndof, i + self.rail.ndof] \
#     #             += self.soil.aux_damping_matrix[0, 0]
#     #         self.global_damping_matrix[i + self.rail.ndof + self.__n_sleepers, i + self.rail.ndof] \
#     #             += self.soil.aux_damping_matrix[1, 0]
#     #         self.global_damping_matrix[i + self.rail.ndof, i + self.rail.ndof + self.__n_sleepers] \
#     #             += self.soil.aux_damping_matrix[0, 1]
#     #         self.global_damping_matrix[i + self.rail.ndof + self.__n_sleepers, i + self.rail.ndof + self.__n_sleepers] \
#     #             += self.soil.aux_damping_matrix[1, 1]
#
#     def set_global_damping_matrix(self):
#         self.global_damping_matrix = sparse.csr_matrix((self.n_dof_track, self.n_dof_track))
#
#         self.rail.set_aux_damping_matrix()
#         self.rail_pads.set_aux_damping_matrix()
#
#         rail_elements = [element for element in self.elements if "RAIL" in element.model_parts]
#         self.global_mass_matrix = utils.add_aux_matrix_to_global(
#             self.global_damping_matrix, self.rail.aux_damping_matrix, rail_elements)
#
#         rail_pad_elements = [element for element in self.elements if "RAIL_PAD" in element.model_parts]
#         self.global_mass_matrix = utils.add_aux_matrix_to_global(
#             self.global_damping_matrix, self.rail_pads.aux_damping_matrix, rail_pad_elements)
#
#         # self.soil.set_aux_damping_matrix()
#
#         # self.__add_rail_to_global_damping_matrix()
#         # self.__add_rail_pad_to_global_damping_matrix()
#         # self.__add_soil_to_global_damping_matrix()
#         pass
#
#     def set_force(self):
#         self.force = sparse.csr_matrix((self.n_dof_track, len(self.time)))
#         frequency=1
#         self.force[4, :] = np.sin(frequency * self.time) * 15000
#
#
#
#         # self.force = np.delete(self.force, idx, axis=0)
#         # self.global_mass_matrix = np.delete(np.delete(self.global_mass_matrix, idx, axis=0),
#         #                                          idx, axis=1)
#         # self.global_stiffness_matrix = np.delete(np.delete(self.global_stiffness_matrix, idx, axis=0),
#         #                                          idx, axis=1)
#         # self.global_damping_matrix = np.delete(np.delete(self.global_damping_matrix, idx, axis=0),
#         #                                          idx, axis=1)
#
#
#     def calculate_n_dofs(self):
#         """
#         todo calculate ndof soil, add train, add ballast
#         :return:
#         """
#         ndof = 0
#         index_dof = 0
#         for node in self.nodes:
#                 node.index_dof[0] = index_dof
#                 index_dof += 1
#                 node.index_dof[1] = index_dof
#                 index_dof += 1
#                 node.index_dof[2] = index_dof
#                 index_dof += 1
#
#         self.n_dof_track = len(self.nodes) * 3
#
#     def calculate_length_track(self):
#         self.__total_length = (self.__n_sleepers - 1) * self.sleeper.distance_between_sleepers
#
#
#     # def initialise_track(self):
#     #     self.set_geometry()
#     #     self.calculate_n_dofs()
#     #
#     #     self.set_global_stiffness_matrix()
#     #     self.set_global_mass_matrix()
#     #     self.set_global_damping_matrix()
#
#         #self.set_force()

if __name__ == "__main__":
    pass
    # # do stuff
    #
    # sleeper = Sleeper()
    # sleeper.ms = 0.1625
    # sleeper.d = 0.6
    # sleeper.damping_ratio = 0.04
    # sleeper.radial_frequency_one = 2
    # sleeper.radial_frequency_two = 500
    #
    # support = Support()
    # support.linear_stiffness = 999
    # support.n_sleepers = 100
    # t = time.time()
    # for i in range(1000):
    #     support.linear_stiffness_matrix
    # elapsed1 = time.time() - t
    # print(elapsed1)
    #
    # support.calculate_matrix()
    # t = time.time()
    # for i in range(1000):
    #     a = support.linear_stiffness_matrix2
    # elapsed1 = time.time() - t
    # print(elapsed1)

    # track = Track()
    #
    # track.damping_ratio = 3
    # track.radial_frequency_one = 2
    # track.radial_frequency_two = 4
    #
    #
    # t = time.time()
    # for i in range(10000):
    #     track.calculate_damping_factors()
    # elapsed1 = time.time() - t
    # print(elapsed1)
