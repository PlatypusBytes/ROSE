from rose.base.model_part import ConditionModelPart, ElementModelPart
import rose.utils.utils as utils
import rose.utils.mesh_utils as mu

import numpy as np
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy import sparse

INTERSECTION_TOLERANCE = 1e-6

class SizeException(Exception):
    pass

class NoDispRotCondition(ConditionModelPart):
    def __init__(self):
        super().__init__()

    # def normal_dof(self):
    #     return
    #     self.normal_dof = False
    #     self.z_rot_dof = False
    #     self.y_disp_dof = False


class LoadCondition(ConditionModelPart):
    def __init__(self, normal_dof=False, y_disp_dof=False, z_rot_dof=False):
        super().__init__()

        self.__normal_dof = normal_dof
        self.__y_disp_dof = y_disp_dof
        self.__z_rot_dof = z_rot_dof

        self.normal_force = None
        self.z_moment = None
        self.y_force = None

        self.normal_force_matrix = None
        self.z_moment_matrix = None
        self.y_force_matrix = None


        self.time = []
        self.initialisation_time = []

    @property
    def normal_dof(self):
        return self.__normal_dof

    @property
    def y_disp_dof(self):
        return self.__y_disp_dof

    @property
    def z_rot_dof(self):
        return self.__z_rot_dof

    def initialize_matrices(self):
        super().initialize()

        if self.normal_dof:
            self.normal_force_matrix = sparse.lil_matrix((len(self.nodes), len(self.time)))
        if self.z_rot_dof:
            self.z_moment_matrix = sparse.lil_matrix((len(self.nodes), len(self.time)))
        if self.y_disp_dof:
            self.y_force_matrix = sparse.lil_matrix((len(self.nodes), len(self.time)))


    def set_load_vector_as_function_of_time(self, load, build_up_idxs):

        time_load = np.ones(len(self.time)) * load
        #
        time_load[0:build_up_idxs] = np.linspace(
            0, load, len(self.initialisation_time)
        )
        return time_load


class LineLoadCondition(LoadCondition):
    def __init__(self, normal_dof=False, y_disp_dof=False, z_rot_dof=False):
        super().__init__(normal_dof, y_disp_dof, z_rot_dof)

        self.nodal_ndof = 3
        self.active_elements = None

        self.contact_model_part: ElementModelPart = None

    def validate(self):
        for element in self.elements:
            if len(element.nodes) != 2:
                raise SizeException("Elements with this condition require 2 nodes")


class MovingPointLoad(LineLoadCondition):
    def __init__(self, normal_dof=False, y_disp_dof=False, z_rot_dof=False, start_coord=None):
        super().__init__(normal_dof, y_disp_dof, z_rot_dof)

        # input
        self.velocities = None
        self.time = None

        self.start_coord = start_coord

        # calculated
        self.start_element_idx = None

        self.cum_distances_force = None
        self.cum_distances_nodes = None

        self.moving_coords = None
        self.moving_normal_force = None
        self.moving_y_force = None
        self.moving_z_moment = None


    def initialize(self):

        self.initialize_matrices()

        self.calculate_cumulative_distance_contact_nodes()
        self.calculate_cumulative_distance_moving_load()

        self.set_active_elements()
        self.set_load_vectors_as_function_of_time()
        self.filter_load_outside_range()
        self.calculate_moving_coords()
        self.set_moving_point_load()


    def set_load_vectors_as_function_of_time(self):
        # set normal force vector as a function of time
        if self.normal_force is not None:
            self.moving_normal_force = self.set_load_vector_as_function_of_time(
                self.normal_force, len(self.initialisation_time))

        # set y force vector as a function of time
        if self.y_force is not None:
            self.moving_y_force = self.set_load_vector_as_function_of_time(
                 self.y_force, len(self.initialisation_time))

        # set z moment vector as a function of time
        if self.z_moment is not None:
            self.moving_z_moment= self.set_load_vector_as_function_of_time(
                 self.z_moment, len(self.initialisation_time))


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
        if self.moving_normal_force is not None:
            self.moving_normal_force = utils.filter_data_outside_range(
                self.moving_normal_force,
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
                self.moving_normal_force,
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

    def calculate_cumulative_distance_contact_nodes(self):
        # get numpy array of nodal coordinates
        nodal_coordinates = np.array([node.coordinates for node in self.nodes])

        # calculate cumulative distances between nodes and point load locations
        self.cum_distances_nodes = mu.calculate_cum_distances_coordinate_array(nodal_coordinates)

    def calculate_cumulative_distance_moving_load(self):

        # if start coords are not given, set the first node as start coordinates
        if self.start_coord is None:
            self.start_coord = np.array(self.nodes[0].coordinates)

        # find element in which the start coordinates are located
        self.start_element_idx = utils.find_intersecting_point_element(self.elements, self.start_coord)

        # calculate distance from force
        self.cum_distances_force = utils.calculate_cum_distance_from_velocity(self.time, self.velocities)

        # add distance force to first node in first active element
        self.cum_distances_force += utils.distance_np(
            self.start_coord, np.array(self.elements[self.start_element_idx].nodes[0].coordinates)
        )
        # ???
        self.cum_distances_force += self.cum_distances_nodes[self.nodes.index(
            self.elements[self.start_element_idx].nodes[0])]

    def distribute_point_load_on_nodes(
        self,
        element_idx,
        node_indices,
        time_idx,
        distance,
        normal_force,
        z_moment,
        y_force,
    ):
        """

        :param element_idx: idx of intersected element
        :param time_idx: idx of time step
        :param intersection_point: intersected point
        :param normal_force:
        :param z_moment:
        :param y_force:
        :return:
        """
        # determine interpolation factors
        # calculate distance between first point in element and intersection point
        # distance = np.sqrt(
        #     np.sum(
        #         (
        #             (self.elements[element_idx].nodes[0].coordinates)
        #             - np.array(intersection_point)
        #         )
        #         ** 2
        #     )
        # )

        # todo make calling of shapefunctions more general, for now it only works on a beam with normal, y and z-rot dof
        # add normal_load_to_nodes
        if normal_force is not None:
            self.contact_model_part.set_normal_shape_functions(distance)
            normal_interp_factors = [
                self.contact_model_part.normal_shape_functions[0],
                self.contact_model_part.normal_shape_functions[1],
            ]

            for idx, node_idx in enumerate(node_indices):
                self.normal_force_matrix[node_idx, time_idx] += (
                    normal_force[time_idx] * normal_interp_factors[idx]
                )

        # add y_load_to_nodes
        if y_force is not None:
            self.contact_model_part.set_y_shape_functions(distance)

            y_interp_factors = [
                self.contact_model_part.y_shape_functions[0],
                self.contact_model_part.y_shape_functions[2],
            ]
            z_mom_interp_factors = [
                self.contact_model_part.y_shape_functions[1],
                self.contact_model_part.y_shape_functions[3],
            ]

            for idx, node_idx in enumerate(node_indices):
                self.z_moment_matrix[node_idx, time_idx] += (
                    y_force[time_idx] * z_mom_interp_factors[idx]
                )
                self.y_force_matrix[node_idx, time_idx] += (
                    y_force[time_idx] * y_interp_factors[idx]
                )

        # add z_moment
        if z_moment is not None:
            self.contact_model_part.set_z_rot_shape_functions(distance)
            y_interp_factors = [
                self.contact_model_part.z_rot_shape_functions[0],
                self.contact_model_part.z_rot_shape_functions[2],
            ]
            z_mom_interp_factors = [
                self.contact_model_part.z_rot_shape_functions[1],
                self.contact_model_part.z_rot_shape_functions[3],
            ]

            for idx, node_idx in enumerate(node_indices):
                self.z_moment_matrix[node_idx, time_idx] += (
                    z_moment[time_idx] * z_mom_interp_factors[idx]
                )
                self.y_force_matrix[node_idx, time_idx] += (
                    z_moment[time_idx] * y_interp_factors[idx]
                )


    def distribute_point_load_on_nodes_2(
            self,
            element_idxs,
            intersection_points,
            normal_force,
            z_moment,
            y_force,
    ):
        """

        :param element_idx: idx of intersected element
        :param time_idx: idx of time step
        :param intersection_point: intersected point
        :param normal_force:
        :param z_moment:
        :param y_force:
        :return:
        """
        #todo correct this vectorized method

        # determine interpolation factors
        # calculate distance between first point in element and intersection point
        np_elements = np.array(self.elements)

        coordinates = np.array([np.array(element.nodes[0].coordinates) for element in np_elements[element_idxs]])
        sq_diff_coords = np.power(coordinates - intersection_points, 2)
        distances = np.sqrt(np.sum(sq_diff_coords, axis=1))



        node_indices = np.array([np.array([self.nodes.index(node) for node in element.nodes]) for element in np_elements[element_idxs]])
        # todo make calling of shapefunctions more general, for now it only works on a beam with normal, y and z-rot dof
        # add normal_load_to_nodes
        if normal_force is not None:
            normal_interp_factors = np.zeros((distances.size,2))
            for idx, distance in enumerate(distances):
                self.contact_model_part.set_normal_shape_functions(distance)
                normal_interp_factors[idx,:] = self.contact_model_part.normal_shape_functions

            # self.contact_model_part.set_normal_shape_functions(distance)
            # normal_interp_factors = [
            #     self.contact_model_part.normal_shape_functions[0],
            #     self.contact_model_part.normal_shape_functions[1],
            # ]

            # node_indices = np.array([np.array([self.nodes.index(node) for node in element]) for element in np_elements[element_idxs]])

            for idx, node in enumerate(self.elements[element_idx].nodes):
                self.normal_force_matrix[self.nodes.index(node), time_idx] += (
                        normal_force[time_idx] * normal_interp_factors[idx]
                )

        # add y_load_to_nodes
        if y_force is not None:

            y_interp_factors = np.zeros((distances.size,2))
            z_mom_interp_factors = np.zeros((distances.size,2))
            for idx, distance in enumerate(distances):
                self.contact_model_part.set_y_shape_functions(distance)
                y_interp_factors[idx,:] = self.contact_model_part.y_shape_functions[0], self.contact_model_part.y_shape_functions[2]
                z_mom_interp_factors[idx,:] = self.contact_model_part.y_shape_functions[1], self.contact_model_part.y_shape_functions[3]



            z_moment_matrix = self.z_moment_matrix.toarray()
            y_force_matrix = self.y_force_matrix.toarray()

            z_moment_matrix[node_indices[:,0], :] += (
                    y_force * z_mom_interp_factors[:,0]
            )
            z_moment_matrix[node_indices[:,1], :] += (
                    y_force * z_mom_interp_factors[:,1]
            )
            self.z_moment_matrix = sparse.lil_matrix(z_moment_matrix)


            y_force_matrix[node_indices[:,0], :] += (
                    y_force * y_interp_factors[:,0]
            )
            y_force_matrix[node_indices[:,1], :] += (
                    y_force * y_interp_factors[:,1]
            )
            self.y_force_matrix = sparse.lil_matrix(y_force_matrix)

        # add z_moment
        if z_moment is not None:

            y_interp_factors = np.zeros((distances.size,2))
            z_mom_interp_factors = np.zeros((distances.size,2))
            for idx, distance in enumerate(distances):
                self.contact_model_part.set_z_rot_shape_functions(distance)
                y_interp_factors[idx,:] = self.contact_model_part.z_rot_shape_functions[0], self.contact_model_part.z_rot_shape_functions[2]
                z_mom_interp_factors[idx,:] = self.contact_model_part.z_rot_shape_functions[1], self.contact_model_part.z_rot_shape_functions[3]



            z_moment_matrix = self.z_moment_matrix.toarray()
            y_force_matrix = self.y_force_matrix.toarray()

            z_moment_matrix[node_indices[:,0], :] += (
                    y_force * z_mom_interp_factors[:,0]
            )
            z_moment_matrix[node_indices[:,1], :] += (
                    y_force * z_mom_interp_factors[:,1]
            )
            self.z_moment_matrix = sparse.lil_matrix(z_moment_matrix)


            y_force_matrix[node_indices[:,0], :] += (
                    y_force * y_interp_factors[:,0]
            )
            y_force_matrix[node_indices[:,1], :] += (
                    y_force * y_interp_factors[:,1]
            )
            self.y_force_matrix = sparse.lil_matrix(y_force_matrix)


            # self.contact_model_part.set_z_rot_shape_functions(distance)
            # y_interp_factors = [
            #     self.contact_model_part.z_rot_shape_functions[0],
            #     self.contact_model_part.z_rot_shape_functions[2],
            # ]
            # z_mom_interp_factors = [
            #     self.contact_model_part.z_rot_shape_functions[1],
            #     self.contact_model_part.z_rot_shape_functions[3],
            # ]
            #
            # for idx, node in enumerate(self.elements[element_idx].nodes):
            #     self.z_moment_matrix[self.nodes.index(node), time_idx] += (
            #             z_moment[time_idx] * z_mom_interp_factors[idx]
            #     )
            #     self.y_force_matrix[self.nodes.index(node), time_idx] += (
            #             z_moment[time_idx] * y_interp_factors[idx]
            #     )

    def set_moving_point_load(self):
        """
        Sets a moving point load on the condition elements.
        :param coordinates:
        :param time:
        :param normal_force:
        :param z_moment:
        :param y_force:
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
        # time steps or nodes are present
        for time_idx in range(len(self.time)):
            self.distribute_point_load_on_nodes(
                int(element_idxs[time_idx]),
                node_indices[time_idx, :],
                time_idx,
                distances[time_idx],
                self.moving_normal_force,
                self.moving_z_moment,
                self.moving_y_force,
            )


