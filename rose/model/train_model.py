import numpy as np
from scipy import sparse
from typing import List

from rose.model import utils
from rose.model.geometry import Node, Mesh
from rose.model.model_part import ElementModelPart
from rose.model.global_system import GlobalSystem
from solvers.newmark_solver import NewmarkSolver
from solvers.static_solver import StaticSolver
from solvers.zhai_solver import ZhaiSolver
from rose.model.irregularities import RailIrregularities

g = 9.81


class Wheel(ElementModelPart):
    """
    Wheel model part class.This class bases from :class:`~rose.model.model_part.ElementModelPart`.

    :Attributes:

        - :self.mass:                   wheel mass
        - :self.total_static_load:      total static load of the wheel
        - :self.distances:              Distance from the zero coordinate to the wheel at every time step
        - :self.active_n_dof:           Number of active degrees of freedom of the wheel
    """
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.total_static_load = None
        self.distances = None
        self.active_n_dof = None
        self.dofs = None

    @property
    def y_disp_dof(self):
        return True

    def fill_mass_matrix(self, mass_matrix):
        """
        Fills global mass matrix with the wheel contribution

        :param mass_matrix: global mass matrix
        :return:
        """
        wheel_dofs = self.nodes[0].index_dof
        mass_matrix[wheel_dofs[1], wheel_dofs[1]] = self.mass

    def fill_static_force_vector(self, static_force_vector):
        """
        Fills global static force vector with the wheel contribution

        :param static_force_vector: global static force vector
        :return:
        """

        wheel_dofs = self.nodes[0].index_dof

        static_force_vector[wheel_dofs[1],0] = -self.mass * g

    def set_static_force_vector(self):
        """
        Calculates the static load of the wheel and adds to the static force vector. Static load is calculated as
        -mass * gravity constant
        :return:
        """
        self.static_force_vector = np.zeros((1, 1))
        self.static_force_vector[0,0] = -self.mass * g

    def set_mesh(self,mesh, y=0, z=0):
        """
        Sets the initial mesh of the wheel and adds the mesh to the train mesh.

        :param mesh: train mesh
        :param y: initial y coordinate of the wheel
        :param z: initial z coordinate of the wheel
        :return:
        """
        self.nodes = [Node(self.distances[0], y, z)]
        mesh.add_unique_nodes_to_mesh(self.nodes)

    def reset_mesh(self,mesh):
        """
        Sets the initial mesh of the wheel and adds the mesh to the train mesh.

        :param mesh: train mesh
        :param y: initial y coordinate of the wheel
        :param z: initial z coordinate of the wheel
        :return:
        """
        mesh.add_unique_nodes_to_mesh(self.nodes)

    def calculate_active_n_dof(self, index_dof):
        """
        Calculates the amount and indices of active degrees of freedom of the wheel.

        :param index_dof: index of the degree of freedom in the global system
        :return:
        """
        self.active_n_dof = 1

        self.nodes[0].index_dof[1] = index_dof
        index_dof += 1

        return index_dof

    def calculate_total_static_load(self, external_load):
        """
        Calculates the total static load on the wheel, this includes the static load of the wheel itself + static
        external force
        :param external_load: external static load which works on the wheel
        :return:
        """
        # self.total_static_load = self.static_force_vector[0, 0] + external_load

        # internal force
        if self.total_static_load is None:
            self.total_static_load = -self.mass * g

        # add external force
        self.total_static_load = self.total_static_load + external_load


class Bogie(ElementModelPart):
    """
    Bogie model part class. This class bases from :class:`~rose.model.model_part.ElementModelPart`. A bogie is a mass
    which can translate and rotate. The bogie is connected to a number of wheels which are located at a certain distance
    from the middle of the bogie.

    :Attributes:
        - :self.wheels:                 all wheels which are connected to the bogie
        - :self.wheel_distances:        List of distance of each connected wheel to the centre of the bogie, including sign
        - :self.distribution_factor     List of factors which indicate how big of a part of stiffness and damping between
                                        wheels and bogie is taken into account. This is important in case of shared bogies
                                        default is 1
        - :self.mass:                   Mass of the bogie
        - :self.inertia:                Moment of inertia of the bogie
        - :self.stiffness:              Stiffness of the spring between the wheels and the bogie
        - :self.damping:                Damping of the dampers between the wheels and the bogie
        - :self.length:                 Length of the bogie
        - :self.total_static_load:      Total static load of the bogie plus static external load
        - :self.distances:              Distance from the zero coordinate to the bogie centre at every time step
        - :self.active_n_dof:           Number of active degrees of freedom of the bogie + connected wheels
    """
    def __init__(self):
        super().__init__()

        self.wheels: List = []
        self.wheel_distances: List = []  # including sign
        self.distribution_factor: List = []
        self.mass = 0
        self.inertia = 0
        self.stiffness = 0  # stiffness between bogie and wheels
        self.damping = 0
        self.length = 0

        self.total_static_load = None
        self.distances = None
        self.active_n_dof = None
        self.dofs = None

        self.__n_wheels = None

    @property
    def y_disp_dof(self):
        return True

    @property
    def z_rot_dof(self):
        return True

    def set_mesh(self, mesh, y=1, z=0):
        """
        Sets the initial mesh of the bogie and wheels and adds the mesh to the train mesh.

        :param mesh: train mesh
        :param y: initial y coordinate of the bogie centre
        :param z: initial z coordinate of the bogie centre
        :return:
        """

        # Create nodes of the bogie
        self.nodes = [Node(self.distances[0], y, z)]

        # add nodes to train mesh
        mesh.add_unique_nodes_to_mesh(self.nodes)

        # create mesh of the wheels which are connected to the bogie
        for wheel in self.wheels:
            wheel.set_mesh(mesh)

    def reset_mesh(self,mesh):
        """
        Resets mesh of the bogie and underlying wheels.
        :param mesh: global mesh
        :return:
        """
        # add nodes to train mesh
        mesh.add_unique_nodes_to_mesh(self.nodes)

        # create mesh of the wheels which are connected to the bogie
        for wheel in self.wheels:
            wheel.reset_mesh(mesh)

    def calculate_active_n_dof(self, index_dof):
        """
        Calculates the amount and indices of active degrees of freedom of the bogie. This includes the active degrees
        of freedom of the connected wheels.

        :param index_dof: index of the degree of freedom in the global system
        :return:
        """

        if all(self.nodes[0].index_dof == None):

            # set index of the degrees of freedom of the bogie
            for node in self.nodes:
                node.index_dof[1] = index_dof
                index_dof += 1
                node.index_dof[2] = index_dof
                index_dof += 1

            # calculate active degrees of freedom of the wheels and sets indices of the degrees of freedom of the wheels
            for idx, wheel in enumerate(self.wheels):
                index_dof = wheel.calculate_active_n_dof(index_dof)

            # calculate active number of degrees of freedom for only the bogie
            active_n_dof_bogie = 2 * len(self.nodes)

            # calculate active number of degrees of freedom of the bogie + connected wheels
            self.active_n_dof = active_n_dof_bogie + sum([wheel.active_n_dof for wheel in self.wheels])
        return index_dof

    def fill_mass_matrix(self, mass_matrix):
        """
        Fills global mass matrix with the bogie contribution

        :param mass_matrix: global mass matrix
        :return:
        """

        # get bogie degrees off freedom
        bogie_dofs = self.nodes[0].index_dof

        # adds bogie contribution
        mass_matrix[bogie_dofs[1], bogie_dofs[1]] = self.mass
        mass_matrix[bogie_dofs[2], bogie_dofs[2]] = self.inertia

        # set connected wheels part of the mass matrix
        for i in range(len(self.wheels)):
            self.wheels[i].fill_mass_matrix(mass_matrix)

    def fill_stiffness_matrix(self, stiffness_matrix):
        """
        Fills global stiffness matrix with the bogie contribution

        :param stiffness_matrix: global stiffness matrix
        :return:
        """

        # get bogie degrees off freedom
        bogie_dofs = self.nodes[0].index_dof

        # adds bogie contribution
        # stiffness_matrix[bogie_dofs[1],bogie_dofs[1]] += len(self.wheels) * self.stiffness
        stiffness_matrix[bogie_dofs[1], bogie_dofs[1]] +=  self.stiffness *sum(self.distribution_factor)

        # set connected wheels part of the stiffness matrix
        for i in range(len(self.wheels)):

            wheel_dofs = self.wheels[i].nodes[0].index_dof

            # add interaction between the bogie and wheels to the stiffness matrix
            stiffness_matrix[bogie_dofs[2], bogie_dofs[2]] += self.stiffness * self.wheel_distances[i] ** 2 * self.distribution_factor[i]

            stiffness_matrix[bogie_dofs[1], wheel_dofs[1]] += -self.stiffness * self.distribution_factor[i]
            stiffness_matrix[wheel_dofs[1], bogie_dofs[1]] += -self.stiffness * self.distribution_factor[i]

            stiffness_matrix[bogie_dofs[2], wheel_dofs[1]] += self.stiffness * self.wheel_distances[i] * self.distribution_factor[i]
            stiffness_matrix[wheel_dofs[1], bogie_dofs[2]] += self.stiffness * self.wheel_distances[i] * self.distribution_factor[i]

            stiffness_matrix[wheel_dofs[1], wheel_dofs[1]] += self.stiffness * self.distribution_factor[i]

    def fill_damping_matrix(self, damping_matrix):

        bogie_dofs = self.nodes[0].index_dof

        # damping_matrix[bogie_dofs[1], bogie_dofs[1]] += len(self.wheels) * self.damping
        damping_matrix[bogie_dofs[1], bogie_dofs[1]] += self.damping * sum(self.distribution_factor)
        # set connected bogies part of the local stiffness matrix
        for i in range(len(self.wheels)):
            wheel_dofs = self.wheels[i].nodes[0].index_dof

            # add interaction between the cart and bogies to the cart local damping matrix
            damping_matrix[bogie_dofs[2], bogie_dofs[2]] += self.damping * self.wheel_distances[i] ** 2 * self.distribution_factor[i]

            damping_matrix[bogie_dofs[1], wheel_dofs[1]] += -self.damping * self.distribution_factor[i]
            damping_matrix[wheel_dofs[1], bogie_dofs[1]] += -self.damping * self.distribution_factor[i]

            damping_matrix[bogie_dofs[2], wheel_dofs[1]] += self.damping * self.wheel_distances[i] * self.distribution_factor[i]
            damping_matrix[wheel_dofs[1], bogie_dofs[2]] += self.damping * self.wheel_distances[i] * self.distribution_factor[i]

            damping_matrix[wheel_dofs[1], wheel_dofs[1]] += self.damping * self.distribution_factor[i]

    def fill_static_force_vector(self, static_force_vector):
        """
        Calculates the static load of the bogie and wheels and adds to the static force vector. Static load is
        calculated as -mass * gravity constant
        :return:
        """

        # get degrees of freedom of the bogie
        bogie_dofs = self.nodes[0].index_dof

        static_force_vector[bogie_dofs[1],0] = -self.mass * g

        # set connected bogies part of the static force vector
        for i in range(len(self.wheels)):
            self.wheels[i].fill_static_force_vector(static_force_vector)

    def calculate_total_static_load(self, external_load, static_force_vector):
        """
        Calculates the total static load on the bogie and on the wheels, this includes the static load of the bogie
        itself + static external force
        :param external_load: external static load which works on the centre of the bogie
        :return:
        """

        # internal force
        bogie_dofs = self.nodes[0].index_dof
        if self.total_static_load is None:
            self.total_static_load = static_force_vector[bogie_dofs[1],0]

        # calculate static load on the bogie itself
        self.total_static_load = self.total_static_load + external_load

    def distribute_static_load(self):
        """
        # distribute the total static load on the bogie over the amount of connected wheels
        :return:
        """

        distributed_load = self.total_static_load / len(self.wheels)
        for wheel in self.wheels:
            wheel.calculate_total_static_load(distributed_load)


class Cart(ElementModelPart):
    """
    Cart model part class. This class bases from :class:`~rose.model.model_part.ElementModelPart`. A cart is a mass
    which can translate and rotate. The cart is connected to a number of bogies which are located at a certain distance
    from the middle of the cart.

    :Attributes:
        - :self.bogies:                 all bogies which are connected to the cart
        - :self.bogie_distances:        List of distance of each connected bogie to the centre of the cart, including sign
        - :self.distribution_factor:    List of factors which indicate how big of a part of stiffness and damping between
                                        carts and bogie is taken into account. This is important in case of shared carts
                                        default is 1
        - :self.mass:                   Mass of the cart
        - :self.inertia:                Moment of inertia of the cart
        - :self.stiffness:              Stiffness of the spring between the bogies and the cart
        - :self.damping:                Damping of the dampers between the bogies and the cart
        - :self.length:                 Length of the cart
        - :self.total_static_load:      Total static load of the cart plus static external load
        - :self.distances:              Distance from the zero coordinate to the cart centre at every time step
        - :self.active_n_dof:           Number of active degrees of freedom of the cart + connected bogies
    """
    def __init__(self):
        super().__init__()

        self.bogies: List = []
        self.bogie_distances: List = []  # including sign
        self.distribution_factor: List = []
        self.mass = 0
        self.inertia = 0
        self.stiffness = 0  # stiffness cart - bogie
        self.damping = 0
        self.length = None

        self.total_static_load = None
        self.distances = None
        self.active_n_dof = None
        self.dofs = None

        self.__n_bogies = None

    @property
    def y_disp_dof(self):
        return True

    @property
    def z_rot_dof(self):
        return True

    def set_mesh(self, mesh, y=2, z=0):
        """
        Sets the initial mesh of the cart and bogies and adds the mesh to the train mesh.

        :param mesh: train mesh
        :param y: initial y coordinate of the bogie centre
        :param z: initial z coordinate of the bogie centre
        :return:
        """
        # Create nodes of the cart
        self.nodes = [Node(self.distances[0], y, z)]

        # add nodes to train mesh
        mesh.add_unique_nodes_to_mesh(self.nodes)

        # create mesh of the bogies which are connected to the cart
        for bogie in self.bogies:
            bogie.set_mesh(mesh)

    def reset_mesh(self, mesh):
        """
        Resets mesh of the cart and underlying bogies.

        :param mesh: global mesh of the train
        :return:
        """

        # add nodes to train mesh
        mesh.add_unique_nodes_to_mesh(self.nodes)

        # create mesh of the bogies which are connected to the cart
        for bogie in self.bogies:
            bogie.reset_mesh(mesh)

    def calculate_active_n_dof(self, index_dof):
        """
        Calculates the amount and indices of active degrees of freedom of the cart. This includes the active degrees
        of freedom of the connected bogies.

        :param index_dof: index of the degree of freedom in the global system
        :return:
        """

        # set index of the degrees of freedom of the cart
        for node in self.nodes:
            node.index_dof[1] = index_dof
            index_dof += 1
            node.index_dof[2] = index_dof
            index_dof += 1

        # calculate active degrees of freedom of the bogies and sets indices of the degrees of freedom of the bogies
        for bogie in self.bogies:
            index_dof = bogie.calculate_active_n_dof(index_dof)

        # calculate active number of degrees of freedom for only the cart
        active_n_dof_cart = 2 * len(self.nodes)

        # calculate active number of degrees of freedom of the cart + connected bogies
        self.active_n_dof = active_n_dof_cart + sum([bogie.active_n_dof for bogie in self.bogies])
        return index_dof

    def fill_mass_matrix(self,mass_matrix):
        """
        Fills global mass matrix with cart contribution

        :param mass_matrix: global mass matrix of the train
        :return:
        """

        # get degrees of freedom of the cart
        cart_dofs = self.nodes[0].index_dof

        # set cart part of mass matrix
        mass_matrix[cart_dofs[1], cart_dofs[1]] = self.mass
        mass_matrix[cart_dofs[2], cart_dofs[2]] = self.inertia

        # set connected bogies part of the mass matrix
        for i in range(len(self.bogies)):
            self.bogies[i].fill_mass_matrix(mass_matrix)

    def fill_stiffness_matrix(self, stiffness_matrix):
        """
        Fills global stiffness matrix with cart contribution

        :param stiffness_matrix: global stiffness matrix of the train
        :return:
        """

        # get degrees of freedom of the cart
        cart_dofs = self.nodes[0].index_dof

        # stiffness_matrix[cart_dofs[1], cart_dofs[1]] += len(self.bogies) * self.stiffness
        stiffness_matrix[cart_dofs[1], cart_dofs[1]] += self.stiffness * sum(
            self.distribution_factor)

        # set connected bogies part of the stiffness matrix
        for i in range(len(self.bogies)):

            # get bogie degrees of freedom
            bogie_dofs = self.bogies[i].nodes[0].index_dof

            # add interaction between the cart and bogies to the stiffness matrix
            stiffness_matrix[cart_dofs[2], cart_dofs[2]] += self.stiffness * self.bogie_distances[i] ** 2 * self.distribution_factor[i]

            stiffness_matrix[cart_dofs[1], bogie_dofs[1]] += -self.stiffness * self.distribution_factor[i]
            stiffness_matrix[bogie_dofs[1], cart_dofs[1]] += -self.stiffness * self.distribution_factor[i]

            stiffness_matrix[cart_dofs[2], bogie_dofs[1]] += self.stiffness * self.bogie_distances[i] * self.distribution_factor[i]
            stiffness_matrix[bogie_dofs[1], cart_dofs[2]] += self.stiffness * self.bogie_distances[i] * self.distribution_factor[i]

            stiffness_matrix[bogie_dofs[1], bogie_dofs[1]] += self.stiffness * self.distribution_factor[i]

            # fill stiffness matrix with bogie contribution
            self.bogies[i].fill_stiffness_matrix(stiffness_matrix)

    def fill_damping_matrix(self, damping_matrix):
        """
        Fills global damping matrix with cart contribution

        :param damping_matrix: global damping matrix of the train
        :return:
        """

        # get cart degrees of freedom
        cart_dofs = self.nodes[0].index_dof

        # set cart part of damping matrix
        # damping_matrix[cart_dofs[1], cart_dofs[1]] += len(self.bogies) * self.damping
        damping_matrix[cart_dofs[1], cart_dofs[1]] += self.damping * sum(self.distribution_factor)

        # set connected bogies part of the local damping matrix
        for i in range(len(self.bogies)):
            bogie_dofs = self.bogies[i].nodes[0].index_dof

            # add interaction between the cart and bogies to the damping matrix
            damping_matrix[cart_dofs[2], cart_dofs[2]] += self.damping * self.bogie_distances[i] ** 2 * self.distribution_factor[i]

            damping_matrix[cart_dofs[1], bogie_dofs[1]] += -self.damping * self.distribution_factor[i]
            damping_matrix[bogie_dofs[1], cart_dofs[1]] += -self.damping * self.distribution_factor[i]

            damping_matrix[cart_dofs[2], bogie_dofs[1]] += self.damping * self.bogie_distances[i] * self.distribution_factor[i]
            damping_matrix[bogie_dofs[1], cart_dofs[2]] += self.damping * self.bogie_distances[i] * self.distribution_factor[i]

            damping_matrix[bogie_dofs[1], bogie_dofs[1]] += self.damping * self.distribution_factor[i]

            # fill damping matrix with bogie contribution
            self.bogies[i].fill_damping_matrix(damping_matrix)

    def fill_static_force_vector(self, static_force_vector):
        """
        Fills global static force vector with static force in carts

        :param static_force_vector: global vector of the static force in the train model
        :return:
        """

        # get degrees of freedom of current cart
        cart_dofs = self.nodes[0].index_dof

        # add static force of cart to the static force vector
        static_force_vector[cart_dofs[1],0] = -self.mass * g

        # fill static force vector for all bogies connected to the cart
        for i in range(len(self.bogies)):
            self.bogies[i].fill_static_force_vector(static_force_vector)

    def calculate_total_static_load(self, external_load, static_force_vector):
        """
        Calculates the total static load on the cart and on the bogies, this includes the static load of the cart
        itself + static external force
        :param external_load: external static load which works on the centre of the cart
        :return:
        """

        # calculate static load on the bogie itself
        cart_dofs = self.nodes[0].index_dof

        if not all(cart_dofs == None):
            self.total_static_load = static_force_vector[cart_dofs[1],0] + external_load
        else:
            self.total_static_load = external_load

        # firstly calculate static loads on bogies
        # distribute the total static load on the cart over the amount of connected bogies
        distributed_load = self.total_static_load / len(self.bogies)
        for bogie in self.bogies:
            bogie.calculate_total_static_load(distributed_load, static_force_vector)


class TrainModel(GlobalSystem):
    """
    Train model part class. This class bases from :class:`~rose.model.global_system.GlobalSystem`. This class contains
    all the attributes, and functions which are exclusively related to the train model. The class contains the mesh
    and model parts of all the train elements.

    :Attributes:
        - :self.carts:                      all carts which are connected to the train
        - :self.cart_distances:             list of dinstances between the [0,0] coordinate to the middle of each cart
        - :self.static_force_vector:        Global force vector of only the static load of the train.
        - :self.velocities:                 np array of the velocity of the train at each time step
        - :self.time:                       time discretisation
        - :self.irregularities_at_wheels:   np array of irregularities at the wheels at each time step
        - :self.total_static_load:          Total static load working on the carts
        - :self.contact_dofs:               global degree of freedom indices which are in contact with the surface (rail)
    """
    def __init__(self):
        super().__init__()
        self.carts: List = None
        self.cart_distances: List = None
        self.static_force_vector: np.ndarray = None

        self.velocities: np.ndarray = None
        self.time: np.ndarray = None

        self.irregularities_at_wheels: np.ndarray = None
        self.use_irregularities = False
        self.total_static_load: float = None

        self.contact_dofs: List = None
        self.nodes = None

        self.__bogies = None
        self.__wheels = None

    @property
    def bogies(self):
        """
        All bogies which are part of the train

        :return:
        """
        return self.__bogies

    @property
    def wheels(self):
        """
        All wheels which are part of the train

        :return:
        """
        return self.__wheels

    def set_mesh(self):
        """
        Sets the mesh of the complete train
        :return:
        """

        # set mesh for each cart
        for cart in self.carts:
            cart.set_mesh(self.mesh)

        # collect nodes
        self.nodes = list(self.mesh.nodes)

    def __get_bogies(self):
        self.__bogies = []
        bogie_nodes = []
        for cart in self.carts:
            for bogie in cart.bogies:
                # only add bogie if there is no bogie at the same location
                if bogie.nodes[0] not in bogie_nodes:
                    bogie_nodes.append(bogie.nodes[0])
                    self.__bogies.append(bogie)

    def __get_wheels(self):
        self.__wheels = []
        wheel_nodes = []
        for bogie in self.__bogies:
            for wheel in bogie.wheels:
                # only add wheel if there is no wheel at the same location
                if wheel.nodes[0] not in wheel_nodes:
                    wheel_nodes.append(wheel.nodes[0])
                    self.__wheels.append(wheel)

    def get_train_parts(self):
        """
        Collect all bogies and wheels which are connected to the train
        :return:
        """
        self.__get_bogies()
        self.__get_wheels()

    def initialise_ndof(self):
        """
        Initialise all indices of the degrees of freedom within the train. Also calculates total number of degrees of
        freedom
        :return:
        """

        # calculate active number of degree of freedom for each cart
        index_dof = 0
        for cart in self.carts:
            index_dof = cart.calculate_active_n_dof(index_dof)

        # calculate total number of degrees of freedom of thw whole train
        self.total_n_dof = sum([cart.active_n_dof for cart in self.carts])

    def set_global_mass_matrix(self):
        """
        Set global mass matrix of train
        :return:
        """

        # initialise global stiffness matrix
        self.global_mass_matrix = np.zeros((self.total_n_dof, self.total_n_dof))

        # set local stiffness matrices for each cart and add to global stiffness matrix
        for cart in self.carts:
            cart.fill_mass_matrix(self.global_mass_matrix)

    def set_global_stiffness_matrix(self):
        """
        Set global stiffness matrix of train
        :return:
        """

        # initialise global stiffness matrix
        self.global_stiffness_matrix = np.zeros((self.total_n_dof, self.total_n_dof))

        # set local stiffness matrices for each cart and add to global stiffness matrix
        for cart in self.carts:
            cart.fill_stiffness_matrix(self.global_stiffness_matrix)

    def set_global_damping_matrix(self):
        """
        Set global damping matrix of train
        :return:
        """

        # initialise global damping matrix
        self.global_damping_matrix = np.zeros((self.total_n_dof, self.total_n_dof))

        # set local stiffness matrices for each cart and add to global stiffness matrix
        for cart in self.carts:
            cart.fill_damping_matrix(self.global_damping_matrix)

    def set_static_force_vector(self):
        """
        Set static force vector of train
        :return:
        """

        # initialise global static force vector
        self.static_force_vector = np.zeros((self.total_n_dof, 1))

        # set static force vector for each cart and add to global static force vector
        for cart in self.carts:
            cart.fill_static_force_vector(self.static_force_vector)

    def calculate_total_static_load(self, external_load=0):
        """
        Calculates the total static load working on the complete train
        :param external_load: Optional external load working on the train
        :return:
        """

        # set static load on the train
        self.total_static_load = external_load

        # divide static load among carts
        distributed_load = self.total_static_load / len(self.carts)

        # firstly add static load on all the carts and bogies
        # todo note that static force vector is not used, instead first time index of global force vector is used for the
        # static force, because the obsolete indices are not reduced from the static force vector
        for cart in self.carts:
            cart.calculate_total_static_load(distributed_load, self.global_force_vector[:,0,None])

        # secondly distribute loads on wheels
        for bogie in self.bogies:
            bogie.distribute_static_load()

    def initialise_irregularities_at_wheels(self):
        """
        Initialise irregularities at the train wheels. If the irregularities are not initialised by the user, the
        irregularities at the wheels are set at 0.
        :return:
        """

        if self.irregularities_at_wheels is None:
            self.irregularities_at_wheels = np.zeros((len(self.wheels), len(self.time)))
            if self.use_irregularities is True:
                for idx, wheel in enumerate(self.wheels):
                    irregularities = RailIrregularities(wheel.distances)
                    self.irregularities_at_wheels[idx,:] = irregularities.irregularities

    def get_contact_dofs(self):
        """
        Gets the indices of the train degrees of freedom which are in contact with the surface (rail)
        :return:
        """

        wheel_dofs = []
        # loop over each wheel
        for wheel in self.wheels:
            for node in wheel.nodes:

                # get all active wheel degrees of freedom
                for idx_dof in node.index_dof:
                    if idx_dof is not None:
                        wheel_dofs.append(idx_dof)

        # append wheel degrees of freedom to contact degrees of freedom
        self.contact_dofs = wheel_dofs

    def initialize_force_vector(self):
        """
        Initialise global force vector of the train
        :return:
        """

        self.global_force_vector = np.zeros((self.total_n_dof, len(self.time)))

        # add static force vector to global force vector
        self.set_static_force_vector()
        self.global_force_vector += self.static_force_vector

    def calculate_distances(self):
        """
        Calculate the distance of each element of the train for each time step.
        :return:
        """

        # calculate time steps
        dt = np.diff(self.time)

        # calculate distance from velocity and time
        distances = np.cumsum(np.append(0, self.velocities[:-1] * dt))

        # calculate distance from the [0, 0] coordinate to the the middle of each cart at each time step
        for i, cart in enumerate(self.carts):
            cart.distances = np.zeros(len(self.time))
            cart.distances = distances + self.cart_distances[i]

            # calculate distance from the [0, 0] coordinate to the the middle of each bogie at each time step
            for j, bogie in enumerate(cart.bogies):
                bogie.distances = np.zeros(len(self.time))
                bogie.distances = cart.distances + cart.bogie_distances[j]

                # calculate distance from the [0, 0] coordinate to the the middle of each wheel at each time step
                for k, wheel in enumerate(bogie.wheels):
                    wheel.distances = np.zeros(len(self.time))
                    wheel.distances = bogie.distances + bogie.wheel_distances[k]

    def initialise_global_matrices(self):
        """
        Inititalise each global train matrix. I.e. mass matrix, damping matrix, stiffness matrix, force vector.
        :return:
        """

        # initialise global matrices
        self.set_global_mass_matrix()
        self.set_global_damping_matrix()
        self.set_global_stiffness_matrix()
        self.initialize_force_vector()

        # remove obsolete indices from global matrices
        self.trim_global_matrices()

    def reset_mesh(self):
        """
        Resets mesh of the train. A new instance of mesh is created and is filled with existing nodes of the carts, bogies and wheels
        :return:
        """
        self.mesh = Mesh()

        # reset mesh of the carts and underlying bogies and wheels
        for cart in self.carts:
            cart.reset_mesh(self.mesh)

        # collect nodes
        self.nodes = list(self.mesh.nodes)

    def check_distribution_factors(self):
        for cart in self.carts:
            if len(cart.distribution_factor) == 0:
                cart.distribution_factor = np.ones(len(cart.bogies))

        for bogie in self.bogies:
            if len(bogie.distribution_factor) == 0:
                bogie.distribution_factor = np.ones(len(bogie.wheels))

    def initialise(self):
        """
        Initialise train. Set geometry, set degrees of freedom, initialises global matrices and vectors, stages and
        solver.

        :return:
        """
        # Setup geometry
        self.calculate_distances()
        self.set_mesh()

        # Get bogies and wheels
        self.get_train_parts()

        # check distribution of bogie and wheel stiffnesses
        self.check_distribution_factors()

        # initialise irregularities at the wheels
        self.initialise_irregularities_at_wheels()

        # setup numbers of degree of freedom and get contact degrees of freedom
        self.initialise_ndof()
        self.get_contact_dofs()

        # resets mesh, this function should be obsolete but it isn't and I don't know why. Currently there are problems
        # when a bogie is connected to 2 carts, therefore this function is required for now.
        self.reset_mesh()

        # initialise global matrices and force vector
        self.initialise_global_matrices()
        self.calculate_total_static_load()

        # get stage divisions
        self.set_stage_time_ids()

        # initialise solver
        self.solver.initialise(self.total_n_dof, self.time)

    def trim_global_matrices(self):
        """
        Removed obsolete indices from global matrices
        :return:
        """

        # trim global matrices
        super().trim_all_global_matrices()

        # reset contact degrees of freedom
        self.get_contact_dofs()

    def calculate_initial_displacement(self, wheel_displacements, shift_in_ndof=0):
        """
        Calculates the initial displacement of the train

        :param wheel_displacements: displacement of track below initial location of the wheels
        :param shift_in_ndof: shift in number degree of freedom, relevant in coupled systems, default is set at 0
        :return:
        """

        # transform matrices to sparse lil matrices
        K = sparse.lil_matrix(np.copy(self.global_stiffness_matrix))
        F = sparse.lil_matrix(np.copy(self.global_force_vector))

        # get the vertical degree of freedom of the wheels
        wheel_dofs = [wheel.nodes[0].index_dof[1] - shift_in_ndof for wheel in self.wheels]

        # initialise static solver
        ini_solver = StaticSolver()
        ini_solver.initialise(self.total_n_dof - len(wheel_dofs), self.time)

        # remove wheel degrees of freedom from the system
        K = utils.delete_from_lil(
            K, row_indices=wheel_dofs, col_indices=wheel_dofs).tocsc()
        F = utils.delete_from_lil(
            F, row_indices=wheel_dofs).tocsc()

        # calculate initial displacement of the train, excluding the wheels
        ini_solver.calculate(K, F, 0, 1)

        # todo take into account initial differential settlements between wheels, for now max displacement of wheel is
        #  taken. This can improve numerical stability.

        # add wheel displacements to the initial displacement of the train system
        mask = np.ones(self.solver.u[0,:].shape, bool)
        mask[wheel_dofs] = False
        self.solver.u[0, mask] = ini_solver.u[1, :] + max(wheel_displacements)
        self.solver.u[0, wheel_dofs] = wheel_displacements

    def calculate_stage(self, start_time_id, end_time_id):
        """
        Calculates a stage of the train system

        :param start_time_id: first time index of the stage
        :param end_time_id: final time index of the stage
        :return:
        """

        # transform matrices to sparse column matrices
        M = sparse.csc_matrix(self.global_mass_matrix)
        C = sparse.csc_matrix(self.global_damping_matrix)
        K = sparse.csc_matrix(self.global_stiffness_matrix)
        F = sparse.csc_matrix(self.global_force_vector)

        # run_stage with Zhai solver
        if isinstance(self.solver, ZhaiSolver):
            self.solver.calculate(M, C, K, F, start_time_id, end_time_id)

        # run_stage with Newmark solver
        if isinstance(self.solver, NewmarkSolver):
            self.solver.calculate(M, C, K, F, start_time_id, end_time_id)

        # run_stage with Static solver
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
        Updates solver

        :param start_time_id: first time index of the stage
        :param end_time_id: final time index of the stage
        :return:
        """
        self.solver.update(start_time_id)

    def main(self):
        """
        Main function of the class. The system is initialised. Each stage is calculated.

        :return:
        """

        self.initialise()

        # calculate stages
        for i in range(len(self.stage_time_ids) - 1):
            self.update_stage(self.stage_time_ids[i], self.stage_time_ids[i + 1])
            self.calculate_stage(self.stage_time_ids[i], self.stage_time_ids[i + 1])