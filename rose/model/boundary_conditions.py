from rose.model.model_part import ConditionModelPart, ElementModelPart
import rose.model.utils as utils
import rose.pre_process.mesh_utils as mu

import numpy as np
from scipy import sparse

INTERSECTION_TOLERANCE = 1e-6

class SizeException(Exception):
    pass

class NoDispRotCondition(ConditionModelPart):
    def __init__(self):
        super().__init__()

class LoadCondition(ConditionModelPart):
    def __init__(self, x_disp_dof=False, y_disp_dof=False, z_rot_dof=False):
        super().__init__()

        self.__x_disp_dof = x_disp_dof
        self.__y_disp_dof = y_disp_dof
        self.__z_rot_dof = z_rot_dof

        # self.x_force = None

        self.x_force = None
        self.z_moment = None
        self.y_force = None

        self.x_force_matrix = None
        self.z_moment_matrix = None
        self.y_force_matrix = None

        self.time = []
        self.initialisation_time = []

    @property
    def x_disp_dof(self):
        return self.__x_disp_dof

    @property
    def y_disp_dof(self):
        return self.__y_disp_dof

    @property
    def z_rot_dof(self):
        return self.__z_rot_dof

    def initialize_matrices(self):
        super().initialize()

        if self.x_disp_dof:
            self.x_force_matrix = sparse.lil_matrix((len(self.nodes), len(self.time)))
        if self.z_rot_dof:
            self.z_moment_matrix = sparse.lil_matrix((len(self.nodes), len(self.time)))
        if self.y_disp_dof:
            self.y_force_matrix = sparse.lil_matrix((len(self.nodes), len(self.time)))


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
    def __init__(self, x_disp_dof=False, y_disp_dof=False, z_rot_dof=False):
        super().__init__(x_disp_dof, y_disp_dof, z_rot_dof)

        self.nodal_ndof = 3
        self.active_elements = None

        self.contact_model_part: ElementModelPart = None

    def validate(self):
        for element in self.elements:
            if len(element.nodes) != 2:
                raise SizeException("Elements with this condition require 2 nodes")


class MovingPointLoad(LineLoadCondition):
    def __init__(self, x_disp_dof=False, y_disp_dof=False, z_rot_dof=False, start_coord=None):
        super().__init__(x_disp_dof, y_disp_dof, z_rot_dof)

        # input
        self.velocities = None
        self.time = None

        self.start_coord = start_coord

        # calculated
        self.start_element_idx = None

        self.cum_distances_force = None
        self.cum_distances_nodes = None

        self.moving_coords = None
        self.moving_x_force = None
        self.moving_y_force = None
        self.moving_z_moment = None

    @property
    def moving_force_vector(self):
        return np.array([self.moving_x_force, self.moving_y_force, self.moving_z_moment])

    def initialize(self):

        self.initialize_matrices()

        self.calculate_cumulative_distance_contact_nodes()
        self.calculate_cumulative_distance_moving_load()

        self.get_first_element_idx()

        self.set_active_elements()
        self.set_load_vectors_as_function_of_time()
        self.filter_load_outside_range()
        self.calculate_moving_coords()
        self.set_moving_point_load()

    def initialize_matrices(self):
        super(MovingPointLoad, self).initialize_matrices()

        if self.moving_x_force is None:
            self.moving_x_force = np.zeros(len(self.time))
        if self.moving_y_force is None:
            self.moving_y_force = np.zeros(len(self.time))
        if self.moving_z_moment is None:
            self.moving_z_moment = np.zeros(len(self.time))
        # if self.moving_force_vector is None:
        #     self.moving_force_vector = np.array([self.moving_x_force, self.moving_y_force, self.moving_z_moment])


    def set_load_vectors_as_function_of_time(self):
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
        # get element idx where point load is located for each time step
        # set_active_elements
        self.active_elements = np.zeros((len(self.elements), len(self.cum_distances_force)))
        i = self.start_element_idx
        for idx, distance in enumerate(self.cum_distances_force):
            if i < len(self.cum_distances_nodes) - 2:
                if distance > self.cum_distances_nodes[i + 1]:
                    i += 1
            self.active_elements[i, idx] = True


    def filter_load_outside_range(self):
        """
        Filter normal load, vertical load an z rotation moment outside range
        :return:
        """
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

        self.cum_distances_force = utils.filer_location(
            self.cum_distances_force, self.cum_distances_nodes[0], self.cum_distances_nodes[-1])


    def calculate_moving_coords(self):
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
        # get numpy array of nodal coordinates
        nodal_coordinates = np.array([node.coordinates for node in self.nodes])

        # calculate cumulative distances between nodes and point load locations
        self.cum_distances_nodes = mu.calculate_cum_distances_coordinate_array(nodal_coordinates)

    def calculate_cumulative_distance_moving_load(self):

        # if start coords are not given, set the first node as start coordinates
        if self.start_coord is None:
            self.start_coord = 0

        # calculate distance from force
        self.cum_distances_force = utils.calculate_cum_distance_from_velocity(self.time, self.velocities)
        self.cum_distances_force += self.start_coord


    def __distribute_normal_force(self, distance, force):
        """
        Distributes normal force on relevant nodes

        :param distance: distance of load from first node
        :param force: load vector
        :return:
        """

        # add normal_load_to_nodes
        self.contact_model_part.set_normal_shape_functions(distance)
        normal_interp_factors = np.array([
            self.contact_model_part.normal_shape_functions[0],
            self.contact_model_part.normal_shape_functions[1],
        ])

        return force[0] * normal_interp_factors


    def __distribute_shear_force(self, distance, force):
        """
        Distributes shear force on relevant nodes

        :param distance: distance of load from first node
        :param force: load vector
        :return:
        """

        self.contact_model_part.set_y_shape_functions(distance)
        shear_interp_factors = np.array([
            self.contact_model_part.y_shape_functions[0],
            self.contact_model_part.y_shape_functions[2],
        ])

        z_mom_interp_factors = np.array([
            self.contact_model_part.y_shape_functions[1],
            self.contact_model_part.y_shape_functions[3],
        ])

        shear_force_vector = force[1] * shear_interp_factors
        z_mom_vector = force[1] * z_mom_interp_factors

        return shear_force_vector, z_mom_vector

    def __distribute_z_moment(self, distance, force):
        """
        Distributes moment around z-axis on relevant nodes

        :param distance: distance of load from first node
        :param force: load vector
        :return:
        """

        self.contact_model_part.set_z_rot_shape_functions(distance)
        shear_interp_factors = np.array([
            self.contact_model_part.z_rot_shape_functions[0],
            self.contact_model_part.z_rot_shape_functions[2],
        ])
        z_mom_interp_factors = np.array([
            self.contact_model_part.z_rot_shape_functions[1],
            self.contact_model_part.z_rot_shape_functions[3],
        ])

        shear_force_vector = force[2] * shear_interp_factors
        z_mom_vector = force[2] * z_mom_interp_factors

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
        global_force_matrix = utils.rotate_point_around_z_axis([-element_rot], local_force_matrix[None,:,:])[0]
        for idx, node_idx in enumerate(node_indices):
            self.x_force_matrix[node_idx, time_idx] += global_force_matrix[0, idx]
            self.y_force_matrix[node_idx, time_idx] += global_force_matrix[1, idx]
            self.z_moment_matrix[node_idx, time_idx] += global_force_matrix[2, idx]

    def set_moving_point_load(self):
        """
        Sets a moving point load on the condition elements.
        :return:
        """

        # find contact element indices
        element_idxs = self.active_elements.nonzero()[0].astype(int)

        # get contact elements
        np_elements = np.array(self.elements)
        contact_elements = np_elements[element_idxs]

        # calculate distances between first element coord and moving load at time t
        coordinates = np.array([np.array(element.nodes[0].coordinates) for element in contact_elements])
        sq_diff_coords = np.power(coordinates - self.moving_coords, 2)
        distances = np.sqrt(np.sum(sq_diff_coords, axis=1))

        # find first and last index of node in nodes list for efficiency
        first_idx = self.nodes.index(np_elements[element_idxs][0].nodes[0])
        last_idx = self.nodes.index(np_elements[element_idxs][-1].nodes[-1])

        # find indices of element nodes in node list
        if first_idx < last_idx:
            node_indices = np.array([np.array([self.nodes.index(node, first_idx, last_idx+1)
                                               for node in element.nodes]) for element in contact_elements])
        else:
            node_indices = np.array([np.array([self.nodes.index(node, last_idx, first_idx+1)
                                               for node in element.nodes]) for element in contact_elements])

        # distribute point load on each time step, vectorizing this method might result in an overflow error if too many



        # get all nodal coordinates
        np_nodes = np.array(self.nodes)
        nodal_coordinates = np.array([[np_nodes[node_indices[time_idx,0]].coordinates,
                                       np_nodes[node_indices[time_idx,1]].coordinates] for time_idx in range(len(self.time))])

        # calculate rotation for each element
        element_rots = utils.calculate_rotation(nodal_coordinates[:,0,:], nodal_coordinates[:,1,:])

        # calculate rotated force vector at each time step
        rotated_force = utils.rotate_point_around_z_axis(element_rots, self.moving_force_vector[:,:].T)

        # distribute rotated forces on nodes
        for time_idx in range(len(self.time)):
            self.distribute_point_load_on_nodes(
                node_indices[time_idx, :],
                time_idx,
                distances[time_idx],
                rotated_force[time_idx,:],
                element_rots[time_idx]
            )


