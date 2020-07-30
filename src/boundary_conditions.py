from src.model_part import ConditionModelPart
import src.utils as utils

import numpy as np
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

INTERSECTION_TOLERANCE = 1e-6

class NoDispRotCondition(ConditionModelPart):
    def __init__(self):
        super().__init__()
        self.normal_dof = False
        self.z_rot_dof = False
        self.y_disp_dof = False


class LoadCondition(ConditionModelPart):
    def __init__(self, normal_dof=False, y_disp_dof=False, z_rot_dof=False):
        super().__init__()
        self.normal_dof = normal_dof
        self.z_rot_dof = z_rot_dof
        self.y_disp_dof = y_disp_dof

        self.normal_force = []
        self.z_moment = []
        self.y_force = []

    def __distribute_load_on_nodes(self, element_idx, time_idx, intersection_point, normal_force, z_moment, y_force):
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
        distances = cdist(np.array([node.coordinates for node in self.elements[element_idx].nodes]),
                          np.array([[intersection_point.x, intersection_point.y, intersection_point.z]]), 'euclidean')
        sum_distances = sum(distances)
        interp_factors = [1 - distance / sum_distances for distance in distances]

        # interpolate given value to nearby nodes
        for idx, node in enumerate(self.elements[element_idx].nodes):
            if normal_force is not None:
                self.normal_force[self.nodes.index(node), time_idx] = normal_force[time_idx] * interp_factors[idx]
            if z_moment is not None:
                self.z_moment[self.nodes.index(node), time_idx] = z_moment[time_idx] * interp_factors[idx]
            if y_force is not None:
                self.y_force[self.nodes.index(node), time_idx] = y_force[time_idx] * interp_factors[idx]

    def set_moving_point_load(self, coordinates, time, normal_force=None, z_moment=None, y_force=None):
        """
        Sets a moving point load on the condition elements.
        :param coordinates:
        :param time:
        :param normal_force:
        :param z_moment:
        :param y_force:
        :return:
        """
        # convert elements to shapely elements for intersection
        shapely_elements = utils.get_shapely_elements(self.elements)

        # calculate centroids
        centroids = np.array([utils.centeroidnp(np.array([node.coordinates for node in element.nodes]))
                              for element in self.elements])

        tree = KDTree(centroids)
        for time_idx in range(len(time)):
            # convert point to shapely point for intersection
            point = Point(coordinates[time_idx].coordinates)

            for i in range(len(self.elements)):
                nr_nearest_neighbours = i+1

                # find nearest neighbour element of point coordinates
                nearest_neighbours = tree.query(coordinates[time_idx].coordinates, k=nr_nearest_neighbours)
                element_idx = nearest_neighbours[1] if isinstance(nearest_neighbours[1], np.int32) \
                    else nearest_neighbours[1][-1]

                # check if coordinate is in element
                if shapely_elements[element_idx].buffer(INTERSECTION_TOLERANCE).intersection(point):
                    self.__distribute_load_on_nodes(element_idx, time_idx, point, normal_force, z_moment, y_force)
                    break