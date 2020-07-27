from src.model_part import ConditionModelPart
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from scipy.spatial.distance import cdist

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

    def set_moving_point_load(self, coordinates, time, moment=None, x_force=None, y_force=None):

        for time_idx in range(len(time)):
            point = Point(coordinates[time_idx].coordinates)
            for element in self.elements:
                # convert element to shapely object
                shapely_el = []
                if len(element.nodes) == 2:
                    shapely_el = LineString(
                        [node.coordinates for node in element.nodes])
                elif len(element.nodes) > 2:
                    shapely_el = Polygon([node.coordinates for node in element.nodes])

                if shapely_el:
                    # check if coordinate is in element
                    if shapely_el.buffer(1e-6).intersection(point):

                        # determine interpolation factors
                        distances = cdist(np.array([node.coordinates for node in element.nodes]),
                                          np.array([[point.x, point.y, point.z]]), 'euclidean')
                        sum_distances = sum(distances)
                        interp_factors = [1 - distance/sum_distances for distance in distances]

                        # interpolate given value to nearby nodes
                        for idx, node in enumerate(element.nodes):
                            if moment is not None:
                                self.moment[self.nodes.index(node), time_idx] = moment[time_idx] * interp_factors[idx]
                            if x_force is not None:
                                self.x_force[self.nodes.index(node), time_idx] = x_force[time_idx] * interp_factors[idx]
                            if y_force is not None:
                                self.y_force[self.nodes.index(node), time_idx] = y_force[time_idx] * interp_factors[idx]
                        break
