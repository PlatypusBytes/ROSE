from src.model_part import ConditionModelPart
import src.utils as utils

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

        self.normal_force = np.array([])
        self.z_moment = np.array([])
        self.y_force = np.array([])

        self.time = []

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
            self.normal_force = sparse.lil_matrix((len(self.nodes), len(self.time)))
        if self.z_rot_dof:
            self.z_moment = sparse.lil_matrix((len(self.nodes), len(self.time)))
        if self.y_disp_dof:
            self.y_force = sparse.lil_matrix((len(self.nodes), len(self.time)))


class LineLoadCondition(LoadCondition):
    def __init__(self, normal_dof=False, y_disp_dof=False, z_rot_dof=False):
        super().__init__(normal_dof, y_disp_dof, z_rot_dof)

        self.nodal_ndof = 3

    def validate(self):
        for element in self.elements:
            if len(element.nodes) != 2:
                raise SizeException("Elements with this condition require 2 nodes")

    def __distribute_point_load_on_nodes(
        self,
        model_part,
        element_idx,
        time_idx,
        intersection_point,
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
        distance = np.sqrt(
            np.sum(
                (
                    (self.elements[element_idx].nodes[0].coordinates)
                    - np.array(intersection_point)
                )
                ** 2
            )
        )

        # todo make calling of shapefunctions more general, for now it only works on a beam with normal, y and z-rot dof
        # add normal_load_to_nodes
        if normal_force is not None:
            model_part.set_normal_shape_functions(distance)
            normal_interp_factors = [
                model_part.normal_shape_functions[0],
                model_part.normal_shape_functions[1],
            ]

            for idx, node in enumerate(self.elements[element_idx].nodes):
                self.normal_force[self.nodes.index(node), time_idx] += (
                    normal_force[time_idx] * normal_interp_factors[idx]
                )

        # add y_load_to_nodes
        if y_force is not None:
            model_part.set_y_shape_functions(distance)

            y_interp_factors = [
                model_part.y_shape_functions[0],
                model_part.y_shape_functions[2],
            ]
            z_mom_interp_factors = [
                model_part.y_shape_functions[1],
                model_part.y_shape_functions[3],
            ]

            for idx, node in enumerate(self.elements[element_idx].nodes):
                self.z_moment[self.nodes.index(node), time_idx] += (
                    y_force[time_idx] * z_mom_interp_factors[idx]
                )
                self.y_force[self.nodes.index(node), time_idx] += (
                    y_force[time_idx] * y_interp_factors[idx]
                )

        # add z_moment
        if z_moment is not None:
            model_part.set_z_rot_shape_functions(distance)
            y_interp_factors = [
                model_part.z_rot_shape_functions[0],
                model_part.z_rot_shape_functions[2],
            ]
            z_mom_interp_factors = [
                model_part.z_rot_shape_functions[1],
                model_part.z_rot_shape_functions[3],
            ]

            for idx, node in enumerate(self.elements[element_idx].nodes):
                self.z_moment[self.nodes.index(node), time_idx] += (
                    z_moment[time_idx] * z_mom_interp_factors[idx]
                )
                self.y_force[self.nodes.index(node), time_idx] += (
                    z_moment[time_idx] * y_interp_factors[idx]
                )

    def set_moving_point_load(
        self,
        model_part,
        coordinates,
        time,
        element_idxs=None,
        normal_force=None,
        y_force=None,
        z_moment=None,
    ):
        """
        Sets a moving point load on the condition elements.
        :param coordinates:
        :param time:
        :param normal_force:
        :param z_moment:
        :param y_force:
        :return:
        """

        # if it is known on which element the point load intersects, do not search the element
        if element_idxs is not None:
            for time_idx in range(len(time)):
                self.__distribute_point_load_on_nodes(
                    model_part,
                    int(element_idxs[time_idx]),
                    time_idx,
                    coordinates[time_idx],
                    normal_force,
                    z_moment,
                    y_force,
                )
            return

        # else search for each timestep on which element in the mesh the point lies
        # convert elements to shapely elements for intersection
        shapely_elements = utils.get_shapely_elements(self.elements)

        # calculate centroids
        centroids = np.array(
            [
                utils.centeroid_np(
                    np.array([node.coordinates for node in element.nodes])
                )
                for element in self.elements
            ]
        )

        tree = KDTree(centroids)
        for time_idx in range(len(time)):
            # convert point to shapely point for intersection
            point = Point(coordinates[time_idx])

            for i in range(len(self.elements)):
                nr_nearest_neighbours = i + 1

                # find nearest neighbour element of point coordinates
                nearest_neighbours = tree.query(
                    coordinates[time_idx], k=nr_nearest_neighbours
                )
                element_idx = (
                    nearest_neighbours[1]
                    if isinstance(nearest_neighbours[1], (np.int32, np.int64))
                    else nearest_neighbours[1][-1]
                )

                # check if coordinate is in element
                if (
                    shapely_elements[element_idx]
                    .buffer(INTERSECTION_TOLERANCE)
                    .intersection(point)
                ):
                    self.__distribute_point_load_on_nodes(
                        model_part,
                        element_idx,
                        time_idx,
                        coordinates[time_idx],
                        normal_force,
                        z_moment,
                        y_force,
                    )
                    break
