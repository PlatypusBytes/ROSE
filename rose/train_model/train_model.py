import numpy as np
from scipy import sparse
from copy import deepcopy
from typing import List

from rose.utils import utils
from rose.base.geometry import Node, Element, Mesh
from rose.base.model_part import ElementModelPart
from rose.base.global_system import GlobalSystem

from rose.solver.solver import NewmarkSolver, StaticSolver, ZhaiSolver

g = 9.81

class Wheel(ElementModelPart):
    def __init__(self):
        super().__init__()
        self.mass = 0 #1e-10
        self.total_static_load = None
        self.distances = None

        self.index_dof = [None]

        self.nodes = None

    @property
    def y_disp_dof(self):
        return True

    def set_aux_mass_matrix(self):
        self.aux_mass_matrix = np.zeros((1, 1))
        self.aux_mass_matrix[0] = self.mass


    def set_aux_stiffness_matrix(self):
        self.aux_stiffness_matrix = np.zeros((1, 1))

    def set_aux_damping_matrix(self):
        self.aux_damping_matrix = np.zeros((1, 1))

    def set_static_force_vector(self):
        self.static_force_vector = np.zeros((1, 1))
        self.static_force_vector[0,0] = -self.mass * g

    def set_mesh(self,mesh, y=0, z=0):
        self.nodes = [Node(self.distances[0], y, z)]
        mesh.add_unique_nodes_to_mesh(self.nodes)

    def calculate_active_n_dof(self, index_dof):
        self.active_n_dof = 1

        self.nodes[0].index_dof[1] = index_dof
        index_dof += 1

        return index_dof

    def calculate_total_static_load(self, external_load):
        self.total_static_load = self.static_force_vector[0, 0] + external_load


class Bogie(ElementModelPart):
    def __init__(self):
        super().__init__()

        self.wheels = None
        self.wheel_distances = 0  # including sign
        self.mass = 0
        self.inertia = 0
        self.stiffness = 0  # stiffness between bogie and wheels
        self.damping = 0
        self.length = 0

        self.__n_wheels = None

        self.total_static_load = None
        self.distances = None

        self.index_dof = [None, None]

    @property
    def y_disp_dof(self):
        return True

    @property
    def z_rot_dof(self):
        return True

    def set_mesh(self, mesh, y=1, z=0):
        self.nodes = [Node(self.distances[0], y, z)]

        mesh.add_unique_nodes_to_mesh(self.nodes)

        for wheel in self.wheels:
            wheel.set_mesh(mesh)

    def __add_wheel_mass_matrix(self, wheel, aux_mass_matrix):
        pass

    def calculate_active_n_dof(self, index_dof):
        self.active_n_dof = 2 + len(self.wheels)

        self.nodes[0].index_dof[1] = index_dof
        index_dof += 1
        self.nodes[0].index_dof[2] = index_dof
        index_dof += 1

        for idx, wheel in enumerate(self.wheels):
            index_dof = wheel.calculate_active_n_dof(index_dof)

        self.active_n_dof = 2 + sum([wheel.active_n_dof for wheel in self.wheels])
        return index_dof


    def set_aux_mass_matrix(self):

        self.aux_mass_matrix = np.zeros((self.active_n_dof, self.active_n_dof))

        self.aux_mass_matrix[0, 0] = self.mass
        self.aux_mass_matrix[1, 1] = self.inertia

        l = 2
        for wheel in self.wheels:
            wheel.set_aux_mass_matrix()
            n_dof_wheel = wheel.aux_mass_matrix.shape[0]
            for j in range(n_dof_wheel):
                for k in range(n_dof_wheel):
                    self.aux_mass_matrix[l + j, l + k] = wheel.aux_mass_matrix[j, k]
            l += n_dof_wheel

    def set_aux_stiffness_matrix(self):
        self.aux_stiffness_matrix = np.zeros((self.active_n_dof, self.active_n_dof))
        self.aux_stiffness_matrix[0, 0] = len(self.wheels) * self.stiffness

        l = 2
        for i in range(len(self.wheels)):
            self.wheels[i].set_aux_stiffness_matrix()
            n_dof_wheel = self.wheels[i].aux_stiffness_matrix.shape[0]
            self.aux_stiffness_matrix[1, 1] += self.stiffness * self.wheel_distances[i] ** 2

            self.aux_stiffness_matrix[0, l] += -self.stiffness
            self.aux_stiffness_matrix[l, 0] += -self.stiffness

            self.aux_stiffness_matrix[1, l] += self.stiffness * self.wheel_distances[i]
            self.aux_stiffness_matrix[l, 1] += self.stiffness * self.wheel_distances[i]

            self.aux_stiffness_matrix[l, l] += self.stiffness

            for j in range(n_dof_wheel):
                for k in range(n_dof_wheel):
                    self.aux_stiffness_matrix[l + j, l + k] += self.wheels[i].aux_stiffness_matrix[j, k]

            l += n_dof_wheel

    def set_aux_damping_matrix(self):
        self.aux_damping_matrix = np.zeros((self.active_n_dof, self.active_n_dof))
        self.aux_damping_matrix[0, 0] = len(self.wheels) * self.damping

        l = 2
        for i in range(len(self.wheels)):
            self.wheels[i].set_aux_damping_matrix()
            n_dof_wheel = self.wheels[i].aux_damping_matrix.shape[0]
            self.aux_damping_matrix[1, 1] += self.damping * self.wheel_distances[i] ** 2

            self.aux_damping_matrix[0, l] += -self.damping
            self.aux_damping_matrix[l, 0] += -self.damping

            self.aux_damping_matrix[1, l] += self.damping * self.wheel_distances[i]
            self.aux_damping_matrix[l, 1] += self.damping * self.wheel_distances[i]

            self.aux_damping_matrix[l, l] += self.damping

            for j in range(n_dof_wheel):
                for k in range(n_dof_wheel):
                    self.aux_damping_matrix[l + j, l + k] += self.wheels[i].aux_damping_matrix[j, k]

            l += n_dof_wheel

    def set_static_force_vector(self):
        self.static_force_vector = np.zeros((self.active_n_dof, 1))
        self.static_force_vector[0, 0] = -self.mass * g

        l=2
        for i in range(len(self.wheels)):
            self.wheels[i].set_static_force_vector()
            n_dof_wheel = self.wheels[i].static_force_vector.shape[0]

            for j in range(n_dof_wheel):
                self.static_force_vector[l+j, 0] += self.wheels[i].static_force_vector[j, 0]
            l += n_dof_wheel


    def __trim_global_matrices_on_indices(self, row_indices: List, col_indices: List):
        """
        Removes items in global stiffness, mass, damping and force vector on row and column indices
        :param row_indices:
        :param col_indices:
        :return:
        """

        self.global_stiffness_matrix = utils.delete_from_lil(
            self.global_stiffness_matrix,
            row_indices=row_indices,
            col_indices=col_indices,
        )
        self.global_mass_matrix = utils.delete_from_lil(
            self.global_mass_matrix, row_indices=row_indices, col_indices=col_indices
        )
        self.global_damping_matrix = utils.delete_from_lil(
            self.global_damping_matrix, row_indices=row_indices, col_indices=col_indices
        )

        self.global_force_vector = utils.delete_from_lil(
            self.global_force_vector, row_indices=row_indices
        )


    def calculate_total_static_load(self, external_load):
        self.total_static_load = self.static_force_vector[0, 0] + external_load

        distributed_load = self.total_static_load / len(self.wheels)
        for wheel in self.wheels:
            wheel.calculate_total_static_load(distributed_load)


class Cart(ElementModelPart):
    def __init__(self):
        super().__init__()

        self.bogies = None
        self.bogie_distances = None  # including sign
        self.mass = 0
        self.inertia =0
        self.stiffness = 0  # stiffness cart - bogie
        self.damping = 0
        self.length = None

        self.__n_bogies = None

        self.total_static_load = None
        self.distances = None

    @property
    def y_disp_dof(self):
        return True

    @property
    def z_rot_dof(self):
        return True

    def set_mesh(self, mesh, y=2, z=0):
        self.nodes = [Node(self.distances[0], y, z)]

        mesh.add_unique_nodes_to_mesh(self.nodes)

        for bogie in self.bogies:
            bogie.set_mesh(mesh)

    def calculate_active_n_dof(self, index_dof):

        for node in self.nodes:
            node.index_dof[1] = index_dof
            index_dof += 1
            node.index_dof[2] = index_dof
            index_dof += 1

        for bogie in self.bogies:
            index_dof = bogie.calculate_active_n_dof(index_dof)

        self.active_n_dof = 2 + sum([bogie.active_n_dof for bogie in self.bogies])
        return index_dof


    def set_aux_mass_matrix(self):
        self.aux_mass_matrix = np.zeros((self.active_n_dof, self.active_n_dof))
        self.aux_mass_matrix[0, 0] = self.mass
        self.aux_mass_matrix[1, 1] = self.inertia

        l = 2
        for bogie in self.bogies:
            bogie.set_aux_mass_matrix()
            n_dof_bogie = bogie.aux_mass_matrix.shape[0]
            for j in range(n_dof_bogie):
                for k in range(n_dof_bogie):
                    self.aux_mass_matrix[l + j, l + k] += bogie.aux_mass_matrix[j, k]
            l += n_dof_bogie

    def set_aux_stiffness_matrix(self):
        self.aux_stiffness_matrix = np.zeros((self.active_n_dof, self.active_n_dof))

        self.aux_stiffness_matrix[0, 0] = len(self.bogies) * self.stiffness

        l = 2
        for i in range(len(self.bogies)):
            self.bogies[i].set_aux_stiffness_matrix()
            n_dof_bogie = self.bogies[i].aux_stiffness_matrix.shape[0]
            self.aux_stiffness_matrix[1, 1] += self.stiffness * self.bogie_distances[i] ** 2

            self.aux_stiffness_matrix[0, l] += -self.stiffness
            self.aux_stiffness_matrix[l, 0] += -self.stiffness

            self.aux_stiffness_matrix[1, l] += self.stiffness * self.bogie_distances[i]
            self.aux_stiffness_matrix[l, 1] += self.stiffness * self.bogie_distances[i]

            self.aux_stiffness_matrix[l, l] += self.stiffness

            for j in range(n_dof_bogie):
                for k in range(n_dof_bogie):
                    self.aux_stiffness_matrix[l + j, l + k] += self.bogies[i].aux_stiffness_matrix[j, k]
            l += n_dof_bogie

    def set_aux_damping_matrix(self):
        self.aux_damping_matrix = np.zeros((self.active_n_dof, self.active_n_dof))

        self.aux_damping_matrix[0, 0] = len(self.bogies) * self.damping

        l = 2
        for i in range(len(self.bogies)):
            self.bogies[i].set_aux_damping_matrix()
            n_dof_bogie = self.bogies[i].aux_damping_matrix.shape[0]
            self.aux_damping_matrix[1, 1] += self.damping * self.bogie_distances[i] ** 2

            self.aux_damping_matrix[0, l] += -self.damping
            self.aux_damping_matrix[l, 0] += -self.damping

            self.aux_damping_matrix[1, l] += self.damping * self.bogie_distances[i]
            self.aux_damping_matrix[l, 1] += self.damping * self.bogie_distances[i]

            self.aux_damping_matrix[l, l] += self.damping

            for j in range(n_dof_bogie):
                for k in range(n_dof_bogie):
                    self.aux_damping_matrix[l + j, l + k] += self.bogies[i].aux_damping_matrix[j, k]
            l += n_dof_bogie

    def set_static_force_vector(self):
        self.static_force_vector = np.zeros((self.active_n_dof, 1))
        self.static_force_vector[0, 0] = -self.mass * g

        l = 2
        for i in range(len(self.bogies)):
            self.bogies[i].set_static_force_vector()
            n_dof_bogie = self.bogies[i].static_force_vector.shape[0]

            for j in range(n_dof_bogie):
                self.static_force_vector[l + j, 0] += self.bogies[i].static_force_vector[j, 0]
            l += n_dof_bogie

    def calculate_total_static_load(self, external_load):
        self.total_static_load = self.static_force_vector[0,0] + external_load

        distributed_load = self.total_static_load / len(self.bogies)
        for bogie in self.bogies:
            bogie.calculate_total_static_load(distributed_load)


class TrainModel(GlobalSystem):
    def __init__(self):
        super().__init__()
        self.carts = None
        self.cart_distances = None

        self.__bogies = None
        self.__wheels = None

        self.herzian_contact_cof = None
        self.herzian_power = 3/2

        self.static_wheel_load = None
        self.static_wheel_deformation = None

        self.static_force_vector = None

        self.velocities = None
        self.time = None

        self.deformation_wheels = None
        self.irregularities_at_wheels = None
        self.total_static_load = None

        self.contact_dofs = None


    @property
    def bogies(self):
        return self.__bogies

    @property
    def wheels(self):
        return self.__wheels

    def set_mesh(self):

        for cart in self.carts:
            cart.set_mesh(self.mesh)

        self.nodes = list(self.mesh.nodes)

    def __get_bogies(self):
        self.__bogies = []
        for cart in self.carts:
            self.__bogies.extend(cart.bogies)

    def __get_wheels(self):
        self.__wheels = []
        for bogie in self.__bogies:
            self.__wheels.extend(bogie.wheels)

    def get_train_parts(self):
        self.__get_bogies()
        self.__get_wheels()


    def initialise_ndof(self):

        index_dof = 0
        for cart in self.carts:
            index_dof = cart.calculate_active_n_dof(index_dof)

        self.total_n_dof = sum([cart.active_n_dof for cart in self.carts])

    def set_global_mass_matrix(self):
        """
        Set mass matrix of train
        :return:
        """

        self.global_mass_matrix = np.zeros((self.total_n_dof, self.total_n_dof))
        l = 0
        for cart in self.carts:
            cart.set_aux_mass_matrix()
            n_dof_cart = cart.aux_mass_matrix.shape[0]
            for j in range(n_dof_cart):
                for k in range(n_dof_cart):
                    self.global_mass_matrix[l + j, l + k] += cart.aux_mass_matrix[j, k]
            l += n_dof_cart

    def set_global_stiffness_matrix(self):
        """
        Set stiffness matrix of train
        :return:
        """

        self.global_stiffness_matrix = np.zeros((self.total_n_dof, self.total_n_dof))
        l = 0
        for cart in self.carts:
            cart.set_aux_stiffness_matrix()
            n_dof_cart = cart.aux_stiffness_matrix.shape[0]
            for j in range(n_dof_cart):
                for k in range(n_dof_cart):
                    self.global_stiffness_matrix[l + j, l + k] += cart.aux_stiffness_matrix[j, k]

            l += n_dof_cart

    def set_global_damping_matrix(self):
        """
        Set damping matrix of train
        :return:
        """

        self.global_damping_matrix = np.zeros((self.total_n_dof, self.total_n_dof))
        l = 0
        for cart in self.carts:
            cart.set_aux_damping_matrix()
            n_dof_cart = cart.aux_damping_matrix.shape[0]
            for j in range(n_dof_cart):
                for k in range(n_dof_cart):
                    self.global_damping_matrix[l + j, l + k] += cart.aux_damping_matrix[j, k]

            l += n_dof_cart


    def set_static_force_vector(self):
        """
        Set static force vector of train
        :return:
        """

        self.static_force_vector = np.zeros((self.total_n_dof, 1))

        l = 0
        for cart in self.carts:

            # set static force vector of cart
            cart.set_static_force_vector()
            n_dof_cart = cart.static_force_vector.shape[0]

            # add static force vector of cart to train
            for j in range(n_dof_cart):
                self.static_force_vector[l + j, 0] += cart.static_force_vector[j, 0]
            l += n_dof_cart

    def calculate_total_static_load(self, external_load=0):

        self.total_static_load = external_load
        distributed_load = self.total_static_load / len(self.carts)
        for cart in self.carts:
            cart.calculate_total_static_load(distributed_load)

    def calculate_static_wheel_deformation(self):
        self.calculate_total_static_load()

        static_wheel_loads = np.array([wheel.total_static_load for wheel in self.wheels])
        self.static_wheel_deformation = self.herzian_contact_cof * np.sign(static_wheel_loads) * abs(static_wheel_loads) ** (1/self.herzian_power)

    def __calculate_elastic_wheel_deformation(self, t):

        elastic_wheel_deformation = (
            # self.static_wheel_deformation
                - self.irregularities_at_wheels[:, t]
        )

        return elastic_wheel_deformation


    def initialise_irregularities_at_wheels(self):
        if self.irregularities_at_wheels is None:
            self.irregularities_at_wheels =np.zeros((len(self.wheels), len(self.time)))

    def get_contact_dofs(self):
        wheel_dofs = []
        for wheel in self.wheels:
            for node in wheel.nodes:
                for idx_dof in node.index_dof:
                    if idx_dof is not None:
                        wheel_dofs.append(idx_dof)

        self.contact_dofs = wheel_dofs

    def initialize_force_vector(self):
        self.global_force_vector = np.zeros((self.total_n_dof, len(self.time)))
        self.set_static_force_vector()

        self.global_force_vector += self.static_force_vector

    def calculate_distances(self):
        """
        Calculate the distance of each element of the train for each time step.
        Set mesh of each element of the train.
        :return:
        """

        dt = np.diff(self.time)

        # calculate distance from velocity and time
        distances = np.cumsum(np.append(0, self.velocities[:-1] * dt))

        # calculated distance for each time step for each cart
        for i, cart in enumerate(self.carts):
            cart.distances = np.zeros(len(self.time))
            cart.distances = distances + self.cart_distances[i]

            # calculated distance for each time step for each bogie of cart
            for j, bogie in enumerate(cart.bogies):
                bogie.distances = np.zeros(len(self.time))
                bogie.distances = cart.distances + cart.bogie_distances[j]

                # calculated distance for each time step for each wheel of bogie
                for k, wheel in enumerate(bogie.wheels):
                    wheel.distances = np.zeros(len(self.time))
                    wheel.distances = bogie.distances + bogie.wheel_distances[k]

    def initialise_global_matrices(self):

        self.set_global_mass_matrix()
        self.set_global_damping_matrix()
        self.set_global_stiffness_matrix()
        self.initialize_force_vector()

        self.trim_global_matrices()

    def initialise(self):
        # Setup geometry
        self.calculate_distances()
        self.set_mesh()

        # Get bogies and wheels
        self.get_train_parts()

        self.initialise_irregularities_at_wheels()

        # setup number degree of freedom
        self.initialise_ndof()
        self.get_contact_dofs()

        self.initialise_global_matrices()
        self.calculate_total_static_load()

        self.set_stage_time_ids()

        self.solver.initialise(self.total_n_dof, self.time)

    def trim_global_matrices(self):
        """
        Removed obsolete indices from global matrices
        :return:
        """

        super().trim_all_global_matrices()
        self.get_contact_dofs()


    def calculate_initial_displacement(self, wheel_displacements, shift_in_ndof=0):
        """
        Calculates the initial displacement of the train
        :param wheel_displacements:
        :param shift_in_ndof: shift in number degree of freedom, relevant in coupled systems, default is set at 0
        :return:
        """

        # transfer matrices to compressed sparsed column matrices
        K = sparse.lil_matrix(deepcopy(self.global_stiffness_matrix))
        F = sparse.lil_matrix(deepcopy(self.global_force_vector))

        wheel_dofs = [wheel.nodes[0].index_dof[1] - shift_in_ndof for wheel in self.wheels]
        ini_solver = StaticSolver()
        ini_solver.initialise(self.total_n_dof - len(wheel_dofs), self.time)
        K = utils.delete_from_lil(
            K, row_indices=wheel_dofs, col_indices=wheel_dofs).tocsc()
        F = utils.delete_from_lil(
            F, row_indices=wheel_dofs).tocsc()
        ini_solver.calculate(K, F, 0, 1)

        # todo take into account initial differential settlements between wheels, for now max displacement of wheel is
        #  taken. This can improve numerical stability.
        mask = np.ones(self.solver.u[0,:].shape, bool)
        mask[wheel_dofs] = False
        self.solver.u[0, mask] = ini_solver.u[1, :] + max(wheel_displacements)
        self.solver.u[0, wheel_dofs] = wheel_displacements

    def calculate_stage(self, start_time_id, end_time_id):
        """
        Calculates the global system
        :return:
        """

        # transfer matrices to compressed sparsed column matrices
        M = sparse.csc_matrix(self.global_mass_matrix)
        C = sparse.csc_matrix(self.global_damping_matrix)
        K = sparse.csc_matrix(self.global_stiffness_matrix)
        F = sparse.csc_matrix(self.global_force_vector)

        # run_stages with Zhai solver
        if isinstance(self.solver, ZhaiSolver):
            self.solver.calculate(M, C, K, F, start_time_id, end_time_id)

        # run_stages with Newmark solver
        if isinstance(self.solver, NewmarkSolver):
            self.solver.calculate(M, C, K, F, start_time_id, end_time_id)

        # run_stages with Static solver
        if isinstance(self.solver, StaticSolver):
            self.solver.calculate(K, F, start_time_id, end_time_id)

    def set_stage_time_ids(self):
        """
        Find indices of unique time steps
        :return:
        """
        diff = np.diff(self.time)
        new_dt_idxs = sorted(np.unique(diff.round(decimals=15), return_index=True)[1])
        self.stage_time_ids = np.append(new_dt_idxs, len(self.time) - 1)

    def update_stage(self, start_time_id, end_time_id):
        """
        Updates model parts and solver
        :param start_time_id:
        :param end_time_id:
        :return:
        """
        self.solver.update(start_time_id)

    def main(self):

        self.initialise()

        # calculate stages
        for i in range(len(self.stage_time_ids) - 1):
            self.update_stage(self.stage_time_ids[i], self.stage_time_ids[i + 1])
            self.calculate_stage(self.stage_time_ids[i], self.stage_time_ids[i + 1])