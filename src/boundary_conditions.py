from src.model_part import ConditionModelPart
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import pandas as pd

import time as timet

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

    def centeroidnp(self, arr):
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        sum_z = np.sum(arr[:, 2])
        return sum_x / length, sum_y / length, sum_z / length


    def set_moving_point_load(self, coordinates, time, moment=None, x_force=None, y_force=None):

        t = timet.time()

        # convert elements to shapely elements for intersection
        shapely_elements = []
        for element in self.elements:
            if len(element.nodes) == 2:
                shapely_elements.append(LineString(
                    [node.coordinates for node in element.nodes]))
            elif len(element.nodes) > 2:
                shapely_elements.append(Polygon([node.coordinates for node in element.nodes]))

        centroids = np.array([self.centeroidnp(np.array([node.coordinates for node in element.nodes]))
                              for element in self.elements])

        tree = KDTree(centroids)
        for time_idx in range(len(time)):
            # convert point to shapely point for intersecion
            point = Point(coordinates[time_idx].coordinates)

            for i in range(len(self.elements)):
                nr_nearest_neighbours = i+1

                # find nearest neighbour element of point coordinates
                nearest_neighbours = tree.query(coordinates[time_idx].coordinates, k=nr_nearest_neighbours)
                element_idx = nearest_neighbours[1] if isinstance(nearest_neighbours[1], np.int32) \
                    else nearest_neighbours[1][-1]

                # check if coordinate is in element
                if shapely_elements[element_idx].buffer(1e-6).intersection(point):
                    # determine interpolation factors
                    distances = cdist(np.array([node.coordinates for node in self.elements[element_idx].nodes]),
                                      np.array([[point.x, point.y, point.z]]), 'euclidean')
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
                    break


        print(timet.time()-t)
            #
            # for element in self.elements:
            #     # convert element to shapely object
            #     shapely_el = []
            #     if len(element.nodes) == 2:
            #         shapely_el = LineString(
            #             [node.coordinates for node in element.nodes])
            #     elif len(element.nodes) > 2:
            #         shapely_el = Polygon([node.coordinates for node in element.nodes])
            #
            #     if shapely_el:
            #         # check if coordinate is in element
            #         if shapely_el.buffer(1e-6).intersection(point):
            #
            #             # determine interpolation factors
            #             distances = cdist(np.array([node.coordinates for node in element.nodes]),
            #                               np.array([[point.x, point.y, point.z]]), 'euclidean')
            #             sum_distances = sum(distances)
            #             interp_factors = [1 - distance/sum_distances for distance in distances]
            #
            #             # interpolate given value to nearby nodes
            #             for idx, node in enumerate(element.nodes):
            #                 if moment is not None:
            #                     self.moment[self.nodes.index(node), time_idx] = moment[time_idx] * interp_factors[idx]
            #                 if x_force is not None:
            #                     self.x_force[self.nodes.index(node), time_idx] = x_force[time_idx] * interp_factors[idx]
            #                 if y_force is not None:
            #                     self.y_force[self.nodes.index(node), time_idx] = y_force[time_idx] * interp_factors[idx]
            #             break



    #
    #
    # def set_moving_point_load(self, coordinates, time, moment=None, x_force=None, y_force=None):
    #     g_elements = []
    #     for element in self.elements:
    #         # convert element to shapely object
    #         if len(element.nodes) == 2:
    #             g_elements.append(LineString(
    #                 [node.coordinates for node in element.nodes]))
    #         elif len(element.nodes) > 2:
    #             g_elements.append(Polygon([node.coordinates for node in element.nodes]))
    #         else:
    #             g_elements.append(None)
    #     df = pd.DataFrame({
    #         "element": g_elements,
    #         "original_elements": self.elements
    #     })
    #
    #     # t = timet.time()
    #     # df_time_int = pd.DataFrame({
    #     #     "time": range(len(time))
    #     # })
    #     # df_time_int = df_time_int.assign(
    #     #     point=list(map(lambda x: Point(coordinates[x].coordinates), df_time_int["time"])))
    #     # df_time_int = df_time_int.assign(intersected_element= list(map(lambda y:
    #     #                                                                df.assign(intersection = list(map(lambda x: True if x.buffer(1e-6).intersection(point) else False, df["element"]))).loc[df["intersection"] == True ].reset_index(drop=True)["original_elements"][0],
    #     #                                                                df_time_int["point"]))
    #     for time_idx in range(len(time)):
    #         point = Point(coordinates[time_idx].coordinates)
    #         df = df.assign(intersection = list(map(lambda x: True if x.buffer(1e-6).intersection(point) else False, df["element"])))
    #         intersected_element = df.loc[df["intersection"] == True ].reset_index(drop=True)["original_elements"][0]
    #
    #         distances = cdist(np.array([node.coordinates for node in intersected_element.nodes]),
    #                           np.array([[point.x, point.y, point.z]]), 'euclidean')
    #         sum_distances = sum(distances)
    #         interp_factors = [1 - distance/sum_distances for distance in distances]
    #
    #         # interpolate given value to nearby nodes
    #         for idx, node in enumerate(intersected_element.nodes):
    #             if moment is not None:
    #                 self.moment[self.nodes.index(node), time_idx] = moment[time_idx] * interp_factors[idx]
    #             if x_force is not None:
    #                 self.x_force[self.nodes.index(node), time_idx] = x_force[time_idx] * interp_factors[idx]
    #             if y_force is not None:
    #                 self.y_force[self.nodes.index(node), time_idx] = y_force[time_idx] * interp_factors[idx]
    #
    #     print(timet.time() - t)
