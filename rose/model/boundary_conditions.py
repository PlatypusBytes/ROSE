from rose.model.model_part import ConditionModelPart, ElementModelPart
import rose.model.utils as utils
import rose.pre_process.mesh_utils as mu

import numpy as np
from scipy import sparse
import math

# typing
from scipy.sparse.base import spmatrix
from typing import List
INTERSECTION_TOLERANCE = 1e-6


class SizeException(Exception):
    pass


class NoDispRotCondition(ConditionModelPart):
    """
    Class which contains a no rotation and no displacement boundary condition.

    """
    def __init__(self):
        super().__init__()


class LoadCondition(ConditionModelPart):
    """
    Class which contains a load boundary condition. This class bases from
    :class:`~rose.model.boundary_conditions.ConditionModelPart`.

    :Attributes:

        - :self.x_force:            Force in x direction
        - :self.y_force:            Force in y direction
        - :self.z_moment:           Moment around z-axis
        - :self.x_force_matrix:     Sparse matrix of force in x direction per time step
        - :self.y_force_matrix:     Sparse matrix of force in y direction per time step
        - :self.z_moment_matrix:    Sparse matrix of force in z direction per time step
        - :self.time:               array of each time step
        - :self.initalisation_time: array of each time step where the force is gradually increase from 0 to the final value
    """
    def __init__(self, x_disp_dof: bool = False, y_disp_dof: bool = False, z_rot_dof: bool = False):
        """
        :param x_disp_dof: true if x displacement degree of freedom is active
        :param y_disp_dof: true if y displacement degree of freedom is active
        :param z_rot_dof: true if z rotation degree of freedom is active
        """
        super().__init__()

        self.__x_disp_dof: bool = x_disp_dof
        self.__y_disp_dof: bool = y_disp_dof
        self.__z_rot_dof: bool = z_rot_dof

        self.x_force: float = None
        self.y_force: float = None
        self.z_moment: float = None

        self.x_force_matrix: spmatrix = None
        self.y_force_matrix: spmatrix = None
        self.z_moment_matrix: spmatrix = None

        self.time = []
        self.initialisation_time = []

    @property
    def x_disp_dof(self):
        """
        True if x displacement degree of freedom is active
        :return:
        """
        return self.__x_disp_dof

    @property
    def y_disp_dof(self):
        """
        True if y displacement degree of freedom is active
        :return:
        """
        return self.__y_disp_dof

    @property
    def z_rot_dof(self):
        """
        True if z rotation degree of freedom is active
        :return:
        """
        return self.__z_rot_dof

    def initialize_matrices(self):
        """
        Initialises force matrices as sparse lil matrices with dimension [number of nodes, number of time steps]

        :return:
        """
        super().initialize()

        if self.x_disp_dof:
            self.x_force_matrix = sparse.lil_matrix((len(self.nodes), len(self.time)))
            self.x_force_vector = np.zeros(len(self.nodes))
        if self.z_rot_dof:
            self.z_moment_matrix = sparse.lil_matrix((len(self.nodes), len(self.time)))
            self.z_moment_vector = np.zeros(len(self.nodes))
        if self.y_disp_dof:
            self.y_force_matrix = sparse.lil_matrix((len(self.nodes), len(self.time)))
            self.y_force_vector = np.zeros(len(self.nodes))

    def set_load_vector_as_function_of_time(self, load: float, build_up_idxs: int) -> np.ndarray:
        """
        Creates a load as a function of time. During the first time steps with a length of the build_up_idxs
        the load is linearly increased from 0.

        :param load: load value
        :param build_up_idxs: number of indices where load is gradually increased from zero
        :return:
        """

        time_load = np.ones(len(self.time)) * load
        #
        time_load[0:build_up_idxs] = np.linspace(
            0, load, len(self.initialisation_time)
        )
        return time_load


class LineLoadCondition(LoadCondition):
    """
    Class which contains a line load boundary condition. This class bases from
    :class:`~rose.model.boundary_conditions.LoadCondition`.

    :Attributes:

        - :self.active_elements:    Numpy array which contains non-zero values if line load condition is active at a
                                        certain node at a certain time [number of nodes, number of time steps]
        - :self.contact_model_part: Element model part which is in contact with the line load condition

    """
    def __init__(self, x_disp_dof=False, y_disp_dof=False, z_rot_dof=False):
        """
        :param x_disp_dof: true if x displacement degree of freedom is active
        :param y_disp_dof: true if y displacement degree of freedom is active
        :param z_rot_dof: true if z rotation degree of freedom is active
        """
        super().__init__(x_disp_dof, y_disp_dof, z_rot_dof)

        self.active_elements: np.ndarray = None
        self.contact_model_part: ElementModelPart = None
        self.contact_model_parts: List[ElementModelPart] = None

    def validate(self):
        """
        Validates if each element in the line load condition contains 2 nodes.
        :return:
        """
        for element in self.elements:
            if len(element.nodes) != 2:
                raise SizeException("Elements with this condition require 2 nodes")


class MovingPointLoad(LineLoadCondition):
    """
    Class which contains a moving point load boundary condition. This class bases from
    :class:`~rose.model.boundary_conditions.LineLoadCondition`.

    :Attributes:

        - :self.velocities:         Numpy array of velocity per time step
        - :self.time:               Numpy array of each time step
        - :self.start_distance:     Initial distance of the moving point load relative to the first node in the condition
        - :self.start_element_idx:  Index of first element which is in contact with the moving load
        - :self.cum_distances_force: Numpy array of cumulative distance of moving point load
        - :self.cum_distances_nodes: Numpy array of cumulative distance of the nodes in current condition
        - :self.moving_coords:      Numpy array of the coordinates at each time step of the moving load
        - :self.moving_x_force:     Numpy array of the force in x direction at each time step
        - :self.moving_y_force:     Numpy array of the force in y direction at each time step
        - :self.moving_z_moment:    Numpy array of the moment around the z-axis at each time step

    """
    def __init__(self, x_disp_dof=False, y_disp_dof=False, z_rot_dof=False, start_distance=None):
        """
        :param x_disp_dof: true if x displacement degree of freedom is active
        :param y_disp_dof: true if y displacement degree of freedom is active
        :param z_rot_dof: true if z rotation degree of freedom is active
        :param start_distance: initial distance of the moving point load relative to the first node in the condition
        """
        super().__init__(x_disp_dof, y_disp_dof, z_rot_dof)

        # input
        self.velocities: np.ndarray = None
        self.time: np.array = None

        self.start_distance: float = start_distance

        # calculated
        self.start_element_idx: int = None

        self.cum_distances_force = None
        self.cum_distances_nodes = None

        self.moving_coords = None
        self.moving_x_force = None
        self.moving_y_force = None
        self.moving_z_moment = None

        self.model_part_at_t = None
    @property
    def moving_force_vector(self):
        """
        Numpy array of moving x force, moving y force and moving z moment
        :return:
        """
        return np.array([self.moving_x_force, self.moving_y_force, self.moving_z_moment])

    def initialize(self):
        """
        Initialises moving point load
        :return:
        """

        # initialise the force matrices
        self.initialize_matrices()

        # calculate cumulative distance of the nodes and the moving load
        self.calculate_cumulative_distance_contact_nodes()
        self.calculate_cumulative_distance_moving_load()

        # find the index of the first contact element
        self.get_first_element_idx()

        # Indicate which elements are active per time step
        self.set_active_elements()

        # Sets the load vector as a function of time
        self.set_load_vectors_as_function_of_time()

        # filters data which is outside of the geometry
        self.filter_load_outside_range()

        # calculate the moving coordinates of the load
        self.calculate_moving_coords()

        self.set_contact_model_part_as_function_of_time()

        self.np_nodes = np.array(self.nodes)

        # set the moving load
        self.set_moving_point_load()



    def initialize_matrices(self):
        """
        Initialises force vectors as sparse as numpy arrays with dimension [number of time steps]

        :return:
        """
        super(MovingPointLoad, self).initialize_matrices()

        if self.moving_x_force is None:
            self.moving_x_force = np.zeros(len(self.time))
        if self.moving_y_force is None:
            self.moving_y_force = np.zeros(len(self.time))
        if self.moving_z_moment is None:
            self.moving_z_moment = np.zeros(len(self.time))

    def set_contact_model_part_as_function_of_time(self):
        """
        Sets the contact model part as a function of time. It is assumed that the contact model parts are spatially
        ordered.
        :return:
        """

        if self.contact_model_parts is not None:

            self.model_part_at_t = []

            # get first contact model part
            mp_idx = 0
            current_model_part = self.contact_model_parts[mp_idx]
            current_max_idx = len(current_model_part.elements)

            # loop over time
            for t in range(len(self.time)):

                # get index of contact element at time t
                active_element = self.active_elements[:,t]
                active_el_idx = np.where(active_element)[0]

                # check if active element is outside current model part, if it is outside, increment contact model part
                if active_el_idx >= current_max_idx:
                    mp_idx += 1
                    current_model_part = self.contact_model_parts[mp_idx]
                    current_max_idx += len(current_model_part.elements)

                # add contact model part to list
                self.model_part_at_t.append(current_model_part)
        else:
            self.model_part_at_t = [self.contact_model_part for t in range(len(self.time))]

    def set_load_vectors_as_function_of_time(self):
        """
        Sets load vectors as a function of time. During initialisation time, the force is increased from 0 to the final
        value. The remaining time, the load vectors remain at a constant value.
        :return:
        """
        # set normal force vector as a function of time
        if self.x_force is not None:
            self.moving_x_force = self.set_load_vector_as_function_of_time(
                self.x_force, len(self.initialisation_time))
        else:
            self.moving_x_force = np.zeros(len(self.time))

        # set y force vector as a function of time
        if self.y_force is not None:
            self.moving_y_force = self.set_load_vector_as_function_of_time(
                 self.y_force, len(self.initialisation_time))
        else:
            self.moving_y_force = np.zeros(len(self.time))

        # set z moment vector as a function of time
        if self.z_moment is not None:
            self.moving_z_moment= self.set_load_vector_as_function_of_time(
                 self.z_moment, len(self.initialisation_time))
        else:
            self.moving_z_moment = np.zeros(len(self.time))

    def set_active_elements(self):
        """
        Gets element indices where point load is located for each time step.

        :return:
        """

        # set_active_elements
        self.active_elements = np.zeros((len(self.elements), len(self.cum_distances_force)))

        # start searching at the initial element index
        i = self.start_element_idx
        for idx, distance in enumerate(self.cum_distances_force):
            if i < len(self.cum_distances_nodes) - 2:

                # if distance of force at current time step is past the distance of the current node, increment the node
                if distance > self.cum_distances_nodes[i + 1]:
                    i += 1
            # fill active elements matrix
            self.active_elements[i, idx] = True

    def filter_load_outside_range(self):
        """
        Filter normal load, vertical load an z rotation moment outside the geometry
        :return:
        """

        # Filter the forces
        if self.moving_x_force is not None:
            self.moving_x_force = utils.filter_data_outside_range(
                self.moving_x_force,
                self.cum_distances_force,
                self.cum_distances_nodes[0],
                self.cum_distances_nodes[-1]
            )

        if self.moving_y_force is not None:
            self.moving_y_force = utils.filter_data_outside_range(
                self.moving_y_force,
                self.cum_distances_force,
                self.cum_distances_nodes[0],
                self.cum_distances_nodes[-1]
            )
        if self.moving_z_moment is not None:
            self.moving_z_moment = utils.filter_data_outside_range(
                self.moving_x_force,
                self.cum_distances_force,
                self.cum_distances_nodes[0],
                self.cum_distances_nodes[-1]
            )

        # filter the cumulative distance of the force
        self.cum_distances_force = utils.filer_location(
            self.cum_distances_force, self.cum_distances_nodes[0], self.cum_distances_nodes[-1])

    def calculate_moving_coords(self):
        """
        Calculates the coordinates of the moving load at each time step.

        :return:
        """
        nodal_coordinates = np.array([node.coordinates for node in self.nodes])
        self.moving_coords = utils.interpolate_cumulative_distance_on_nodes(
            self.cum_distances_nodes, nodal_coordinates, self.cum_distances_force
        )

    def get_first_element_idx(self):
        """
        Gets first contact element index. This function assumes a sorted list of elements
        :return:
        """

        # calculate distance between cumulative distance nodes and the initial force position
        ini_diff_node_force = self.cum_distances_nodes - self.cum_distances_force[0]

        # check if all elements have 2 nodes
        if np.all([len(element.nodes) == 2 for element in self.elements]):
            # get starting element index, it is the index of the first positive value -1 of the ini_diff_node_force
            self.start_element_idx = np.where(ini_diff_node_force > 0, ini_diff_node_force, np.inf).argmin() - 1
        else:
            self.start_element_idx = None

    def calculate_cumulative_distance_contact_nodes(self):
        """
        Calculate the cumulative distance of the contact nodes.

        :return:
        """
        # get numpy array of nodal coordinates
        nodal_coordinates = np.array([node.coordinates for node in self.nodes])

        # calculate cumulative distances between nodes and point load locations
        self.cum_distances_nodes = mu.calculate_cum_distances_coordinate_array(nodal_coordinates)

    def calculate_cumulative_distance_moving_load(self):
        """
        Calculate the cumulative distance of the moving load
        :return:
        """

        # if start coords are not given, set the first node as start coordinates
        if self.start_distance is None:
            self.start_distance = 0

        # calculate distance from force
        self.cum_distances_force = utils.calculate_cum_distance_from_velocity(self.time, self.velocities)
        self.cum_distances_force += self.start_distance

    def __distribute_normal_force(self, distance, force):
        """
        Distributes normal force on relevant nodes

        :param distance: distance of load from first node
        :param force: load vector
        :return:
        """

        # add normal_load_to_nodes
        self.contact_model_part.set_normal_shape_functions(distance)
        normal_force_vector = np.array([
            force[0] * self.contact_model_part.normal_shape_functions[0],
            force[0] * self.contact_model_part.normal_shape_functions[1],
        ])

        return normal_force_vector

    def __distribute_shear_force(self, distance, force):
        """
        Distributes shear force on relevant nodes

        :param distance: distance of load from first node
        :param force: load vector
        :return:
        """

        self.contact_model_part.set_y_shape_functions(distance)

        shear_force_vector = np.array([
            force[1] * self.contact_model_part.y_shape_functions[0],
            force[1] * self.contact_model_part.y_shape_functions[2],
        ])

        z_mom_vector = np.array([
            force[1] * self.contact_model_part.y_shape_functions[1],
            force[1] * self.contact_model_part.y_shape_functions[3],
        ])

        return shear_force_vector, z_mom_vector

    def __distribute_z_moment(self, distance, force):
        """
        Distributes moment around z-axis on relevant nodes

        :param distance: distance of load from first node
        :param force: load vector
        :return:
        """

        self.contact_model_part.set_z_rot_shape_functions(distance)
        shear_force_vector = np.array([
            force[2] * self.contact_model_part.z_rot_shape_functions[0],
            force[2] * self.contact_model_part.z_rot_shape_functions[2],
        ])
        z_mom_vector = np.array([
            force[2] * self.contact_model_part.z_rot_shape_functions[1],
            force[2] * self.contact_model_part.z_rot_shape_functions[3],
        ])

        return shear_force_vector, z_mom_vector

    def distribute_point_load_on_nodes(
        self,
        node_indices: np.ndarray,
        time_idx: int,
        distance: float,
        rotated_force: np.ndarray,
        element_rot: float,
    ):
        """
        Distribute point load on surrounding nodes at timestep t

        :param node_indices: indices of surrounding nodes at time t
        :param time_idx: idx of time step
        :param distance: distance point load to first node of element at time t
        :param rotated_force: rotated force vector at time t
        :param element_rot: rotation of contact element at time t
        :return:
        """

        # get contact_model part
        self.contact_model_part = self.model_part_at_t[time_idx]

        # todo make calling of shapefunctions more general, for now it only works on a beam with normal, y and z-rot dof
        # get nodal normal force vector
        normal_force_vector = self.__distribute_normal_force(distance, rotated_force)

        # get nodal shear force and z-moment force vectors
        shear_force_vector_v, z_mom_vector_v = self.__distribute_shear_force(distance, rotated_force)
        shear_force_vector_z, z_mom_vector_z = self.__distribute_z_moment(distance, rotated_force)

        shear_force_vector = shear_force_vector_v + shear_force_vector_z
        z_mom_vector = z_mom_vector_v + z_mom_vector_z

        # combine all local forces in array
        local_force_matrix = np.array([normal_force_vector,shear_force_vector,z_mom_vector])

        # calculate global forces at a single timestep
        global_force_matrix = utils.rotate_point_around_z_axis(np.array([-element_rot]), local_force_matrix[None,:,:])[0]
        for idx, node_idx in enumerate(node_indices):
            self.x_force_matrix[node_idx, time_idx] += global_force_matrix[0, idx]
            self.y_force_matrix[node_idx, time_idx] += global_force_matrix[1, idx]
            self.z_moment_matrix[node_idx, time_idx] += global_force_matrix[2, idx]

    def set_moving_point_load(self):
        """
        Sets a moving point load on the condition elements.
        :return:
        """
        t=0
        self.update_force(t)
        # # find contact element indices
        # element_idxs = self.active_elements.nonzero()[0].astype(int)
        #
        # # get contact elements
        # np_elements = np.array(self.elements)
        # unique_contact_elements = np_elements[list((dict.fromkeys(element_idxs)))]
        # contact_elements = np_elements[element_idxs]
        #
        # # calculate distances between first element coord and moving load at time t
        # coordinates = np.array([np.array(element.nodes[0].coordinates) for element in contact_elements])
        # sq_diff_coords = np.power(coordinates - self.moving_coords, 2)
        # distances = np.sqrt(np.sum(sq_diff_coords, axis=1))
        #
        # # find first and last index of node in nodes list for efficiency
        # first_idx = self.nodes.index(np_elements[element_idxs][0].nodes[0])
        # last_idx = self.nodes.index(np_elements[element_idxs][-1].nodes[-1])
        #
        # # find indices of element nodes in node list
        # if first_idx < last_idx:
        #     node_indices = np.array([np.array([self.nodes.index(node, first_idx, last_idx+1)
        #                                        for node in element.nodes]) for element in unique_contact_elements])
        #
        # else:
        #     node_indices = np.array([np.array([self.nodes.index(node, first_idx, last_idx + 1)
        #                                        for node in element.nodes]) for element in unique_contact_elements])
        #
        #
        # # get node indices of contact element at every time step
        # i = 0
        # new_node_indices = []
        # for element in contact_elements:
        #     if i < len(unique_contact_elements):
        #         if element == unique_contact_elements[i]:
        #             i += 1
        #     new_node_indices.append(node_indices[i-1])
        # node_indices = np.array(new_node_indices)
        #
        # # get all nodal coordinates
        # np_nodes = np.array(self.nodes)
        # # nodal_coordinates = np.array([[np_nodes[node_indices[time_idx,0]].coordinates,
        # #                                np_nodes[node_indices[time_idx,1]].coordinates] for time_idx in range(len(self.time))])
        #
        # # nodal_coordinates = np.array([np_nodes[node_indices[t,0]].coordinates,
        # #                                np_nodes[node_indices[t,1]].coordinates])
        #
        # # calculate rotation for each element
        # element_rot = utils.calculate_point_rotation(np_nodes[node_indices[t,0]].coordinates, np_nodes[node_indices[t,1]].coordinates)
        #
        # # calculate rotated force vector at each time step
        # moving_force_vector = np.array([self.moving_x_force[t], self.moving_y_force[t], self.moving_z_moment[t]])
        #
        # rotated_force = utils.rotate_point_around_z_axis(element_rot, moving_force_vector)
        #
        # # transform matrices to nd arrays for efficiency
        # self.x_force_matrix = self.x_force_matrix.toarray()
        # self.y_force_matrix = self.y_force_matrix.toarray()
        # self.z_moment_matrix = self.z_moment_matrix.toarray()
        #
        # #distribute rotated forces on nodes, vectorizing this method might result in an overflow error
        # for time_idx in range(len(self.time)):
        #     self.distribute_point_load_on_nodes(
        #         node_indices[time_idx, :],
        #         time_idx,
        #         distances[time_idx],
        #         rotated_force,
        #         element_rot
        #     )

    def update_force(self, t):

        self.x_force_vector = np.zeros(len(self.nodes))
        self.z_moment_vector = np.zeros(len(self.nodes))
        self.y_force_vector = np.zeros(len(self.nodes))

        if t == 0:

            # find contact element indices
            element_idxs = self.active_elements.nonzero()[0].astype(int)

            # get contact elements
            np_elements = np.array(self.elements)
            unique_contact_elements = np_elements[list((dict.fromkeys(element_idxs)))]
            contact_elements = np_elements[element_idxs]

            # calculate distances between first element coord and moving load at time t
            coordinates = np.array([np.array(element.nodes[0].coordinates) for element in contact_elements])
            sq_diff_coords = np.power(coordinates - self.moving_coords, 2)

            self.distances = np.sqrt(np.sum(sq_diff_coords, axis=1))

            # find first and last index of node in nodes list for efficiency
            first_idx = self.nodes.index(np_elements[element_idxs][0].nodes[0])
            last_idx = self.nodes.index(np_elements[element_idxs][-1].nodes[-1])

            # find indices of element nodes in node list
            if first_idx < last_idx:
                node_indices = np.array([np.array([self.nodes.index(node, first_idx, last_idx+1)
                                                   for node in element.nodes]) for element in unique_contact_elements])

            else:
                node_indices = np.array([np.array([self.nodes.index(node, first_idx, last_idx + 1)
                                                   for node in element.nodes]) for element in unique_contact_elements])

            # get node indices of contact element at every time step
            i = 0
            new_node_indices = []
            for element in contact_elements:
                if i < len(unique_contact_elements):
                    if element == unique_contact_elements[i]:
                        i += 1
                new_node_indices.append(node_indices[i-1])
            self.node_indices = np.array(new_node_indices)

            # get all nodal coordinates
            self.np_nodes = np.array(self.nodes)

        np_nodes = self.np_nodes

        # calculate rotation for each element
        element_rot = utils.calculate_point_rotation(np_nodes[self.node_indices[t,0]].coordinates, np_nodes[self.node_indices[t,1]].coordinates)

        # calculate rotated force vector at each time step
        moving_force_vector = np.array([self.moving_x_force[t], self.moving_y_force[t], self.moving_z_moment[t]])

        rotated_force = utils.rotate_point_around_z_axis(element_rot, moving_force_vector)

        # get contact_model part
        self.contact_model_part = self.model_part_at_t[t]

        # todo make calling of shapefunctions more general, for now it only works on a beam with normal, y and z-rot dof
        # get nodal normal force vector
        if math.isclose(rotated_force[0],0):
            normal_force_vector = np.zeros(len(self.contact_model_part.normal_shape_functions))
        else:
            normal_force_vector = self.__distribute_normal_force(self.distances[t], rotated_force)

        # get nodal shear force and z-moment force vectors
        if math.isclose(rotated_force[1], 0):
            shear_force_vector_v = z_mom_vector_v = np.zeros(int(len(self.contact_model_part.y_shape_functions)/2))
        else:
            shear_force_vector_v, z_mom_vector_v = self.__distribute_shear_force(self.distances[t], rotated_force)

        if math.isclose(rotated_force[2], 0):
            shear_force_vector_z = z_mom_vector_z = np.zeros(int(len(self.contact_model_part.z_rot_shape_functions)/2))
        else:
            shear_force_vector_z, z_mom_vector_z = self.__distribute_z_moment(self.distances[t], rotated_force)

        shear_force_vector = shear_force_vector_v + shear_force_vector_z
        z_mom_vector = z_mom_vector_v + z_mom_vector_z

        # combine all local forces in array
        local_force_matrix = np.array([normal_force_vector, shear_force_vector, z_mom_vector])

        # calculate global forces at a single timestep
        global_force_vector = utils.rotate_point_around_z_axis(-element_rot, local_force_matrix[:, :])

        for idx, node_idx in enumerate(self.node_indices[t,:]):

            self.x_force_vector[node_idx] += global_force_vector[0, idx]
            self.y_force_vector[node_idx] += global_force_vector[1, idx]
            self.z_moment_vector[node_idx] += global_force_vector[2, idx]




