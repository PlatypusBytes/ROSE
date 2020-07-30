from src.model_part import ConditionModelPart
from src.utils import centeroidnp, get_shapely_elements

import numpy as np
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

INTERSECTION_TOLERANCE = 1e-6

class NoDispRotCondition(ConditionModelPart):
    def __init__(self):
        super().__init__()
        self.rotation_dof = False
        self.x_disp_dof = False
        self.y_disp_dof = False


class CauchyCondition(ConditionModelPart):
    def __init__(self, rotation_dof=False, y_disp_dof=False, x_disp_dof=False):
        super(ConditionModelPart).__init__()
        self.rotation_dof = rotation_dof
        self.x_disp_dof = x_disp_dof
        self.y_disp_dof = y_disp_dof

        self.moment = []
        self.x_force = []
        self.y_force = []

    def __distribute_load_on_nodes(self, element_idx, time_idx, intersection_point, moment, x_force, y_force):
        """

        :param element_idx: idx of intersected element
        :param time_idx: idx of time step
        :param intersection_point: intersected point
        :param moment:
        :param x_force:
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
            if moment is not None:
                self.moment[self.nodes.index(node), time_idx] = moment[time_idx] * interp_factors[idx]
            if x_force is not None:
                self.x_force[self.nodes.index(node), time_idx] = x_force[time_idx] * interp_factors[idx]
            if y_force is not None:
                self.y_force[self.nodes.index(node), time_idx] = y_force[time_idx] * interp_factors[idx]

    def set_moving_point_load(self, coordinates, time, moment=None, x_force=None, y_force=None):
        """
        Sets a moving point load on the condition elements.
        :param coordinates:
        :param time:
        :param moment:
        :param x_force:
        :param y_force:
        :return:
        """
        # convert elements to shapely elements for intersection
        shapely_elements = get_shapely_elements(self.elements)

        # calculate centroids
        centroids = np.array([centeroidnp(np.array([node.coordinates for node in element.nodes]))
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
                    self.__distribute_load_on_nodes(element_idx, time_idx, point, moment, x_force, y_force)
                    break