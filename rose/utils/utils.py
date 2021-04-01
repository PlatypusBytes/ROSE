from scipy import sparse
import numpy as np
from shapely.geometry import LineString, Polygon

from typing import List

# from src.boundary_conditions import LoadCondition
from rose.model.geometry import Node, Element
from scipy import sparse
from scipy.spatial.distance import cdist

import copy

from shapely.geometry import Point
from scipy.spatial import KDTree
from scipy import sparse
from scipy import interpolate

def filer_location(locations: np.array, lower_limit: float, upper_limit: float):
    locations[locations > upper_limit] = upper_limit
    locations[locations < lower_limit] = lower_limit

    return locations

def filter_data_outside_range(
    data: np.array, locations: np.array, lower_limit: float, upper_limit: float
):
    # Set force at 0 when its located outside the range
    data[locations > upper_limit] = 0
    data[locations < lower_limit] = 0

    # new_locations = copy.deepcopy(locations)
    # new_locations[locations > upper_limit] = upper_limit
    # new_locations[locations < lower_limit] = lower_limit

    return data #, new_locations


def calculate_cumulative_distance_of_coordinates(coordinates: np.array):
    """
    Calculates cumulative distance between a sorted coordinate array
    :param coordinates:
    :return:
    """
    return np.append(
        0,
        np.cumsum(
            distance_np(
                coordinates[:-1, :], coordinates[1:, :], axis=1
            )
        ),
    )


def interpolate_cumulative_distance_on_nodes(
    cumulative_distances: np.array, coordinates: np.array, new_cumulative_distances: np.array
):
    """
    Interpolate cumulative distances on coordinates

    :param cumulative_distances:
    :param coordinates:
    :param new_cumulative_distances:
    :return:
    """

    # set interpolation functions
    fx = interpolate.interp1d(cumulative_distances, coordinates[:, 0])
    fy = interpolate.interp1d(cumulative_distances, coordinates[:, 1])
    fz = interpolate.interp1d(cumulative_distances, coordinates[:, 2])

    # interpolate
    new_x_coords = fx(new_cumulative_distances)
    new_y_coords = fy(new_cumulative_distances)
    new_z_coords = fz(new_cumulative_distances)

    return np.array([new_x_coords, new_y_coords, new_z_coords]).transpose()

def calculate_cum_distance_from_velocity(time, velocities):
    cum_distances = np.append(
                    0,
                    np.cumsum((time[1:] - time[:-1]) * velocities[:-1])
                    )
    return cum_distances


def reshape_force_vector(n_nodes_element, dofs, force_vector):
    """
    Reshapes force vector of the model part based on the total possible degrees of freedom (active and unactive). This
    function is required to easily fill the global force vector

    :param n_nodes: number of nodes per element
    :param dofs: list of degrees of freedoms
    :param force_vector: current force vector
    :return:
    """

    new_force_vector = np.zeros((n_nodes_element * len(dofs), 1))

    dofs = np.asarray(dofs)
    for i in range(n_nodes_element):
        for k in range(len(dofs)):
            if dofs[k]:
                new_force_vector[
                    i * len(dofs) + k, 0
                ] = force_vector[
                    i * sum(dofs) + sum(dofs[0:k]),
                    0,
                ]

    return new_force_vector


def reshape_aux_matrix(n_nodes_element, dofs, aux_matrix):
    """
    Reshapes aux matrix of the model part based on the total possible degrees of freedom (active and unactive). This
    function is required to easily fill the global matrices

    :param n_nodes: number of nodes per element
    :param dofs: list of degrees of freedoms
    :param aux_matrix: current auxiliar matrix
    :return:
    """
    new_aux_matrix = np.zeros((n_nodes_element * len(dofs), n_nodes_element * len(dofs)))

    dofs = np.asarray(dofs)
    for i in range(n_nodes_element):
        for j in range(n_nodes_element):
            for k in range(len(dofs)):
                if dofs[k]:
                    for l in range(len(dofs)):
                        if dofs[l]:
                            new_aux_matrix[
                                i * len(dofs) + k, j * len(dofs) + l
                            ] = aux_matrix[
                                i * sum(dofs) + sum(dofs[0:k]),
                                j * sum(dofs) + sum(dofs[0:l]),
                            ]

    return new_aux_matrix


def delete_from_lil(mat: sparse.lil_matrix, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the lil sparse matrix
    ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:, col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat

# def calculate_rotation(node1: Node, node2: Node):
#     """
#     Calculates rotation between 2 nodes in a 2d space
#     :param node1: first node
#     :param node2: second node
#     :return:
#     """
#     #todo make general, now it works for 2 nodes in a 2d space
#     if np.isclose(node2.coordinates[0] - node1.coordinates[0], 0):
#         return np.pi/2 * np.sign((node2.coordinates[1] - node1.coordinates[1]))
#
#     rot = 0
#     if node2.coordinates[0] < node1.coordinates[0]:
#         rot += np.pi
#
#     rot += np.arctan((node2.coordinates[1] - node1.coordinates[1])/ (node2.coordinates[0] - node1.coordinates[0]))
#     return rot

def calculate_rotation(coord1: np.ndarray, coord2: np.ndarray):
    """
    Calculates rotation between 2 nodes in a 2d space
    :param node1: first node
    :param node2: second node
    :return:
    """
    #todo make general, now it works for 2 nodes in a 2d space
    rot = np.zeros(coord1.shape[0])
    is_x_equal = np.isclose(coord2[:,0] - coord1[:,0], 0)
    rot[is_x_equal] = np.pi/2 * np.sign((coord2[is_x_equal, 1] - coord1[is_x_equal, 1]))
        # return np.pi/2 * np.sign((coord2[1] - coord1[1]))
    rot[coord2[:,0] < coord1[:,0]] += np.pi

    rot[~is_x_equal] += np.arctan((coord2[:,1] - coord1[:,1])/ (coord2[:,0] - coord1[:,0]))
    return rot


def rotate_point_around_z_axis(rotation: np.ndarray, point_vector: np.ndarray):
    """
    Rotates a point around the z-axis
    :param rotation: rotation in radians
    :param point_vector: vector of global values [x-direction, y-direction, z-rotation]
    :return:
    """

    rot_matrix = np.zeros((len(rotation),3, 3))
    rot_matrix[:, 0, 0] = np.cos(rotation)
    rot_matrix[:, 1, 1] = np.cos(rotation)
    rot_matrix[:, 0, 1] = np.sin(rotation)
    rot_matrix[:,1, 0] = -rot_matrix[:,0, 1]
    rot_matrix[:,2, 2] = 1

    rotated_point = np.zeros(point_vector.shape)

    # rotate each time step
    for idx,(mat, point) in enumerate(zip(rot_matrix,point_vector)):
        rotated_point[idx] = mat.dot(point)
    return rotated_point



def rotate_force_vector(element: Element, contact_model_part, force_vector: np.array):
    """
    Rotates force vector based on rotation of element
    :param element:
    :param contact_model_part:
    :param force_vector:
    :return:
    """
    # todo make general, now it works for 2 nodes in a 2d space
    if len(element.nodes) == 2:
        rot = calculate_rotation(element.nodes[0], element.nodes[1])
        contact_model_part.set_rotation_matrix(rot, 2)
        rot_matrix = contact_model_part.rotation_matrix[:3,:3]

        if rot_matrix is not None:
            return rot_matrix.dot(force_vector)

    return force_vector

def rotate_aux_matrix(element: Element, model_part, aux_matrix: np.array):
    """
    Rotates aux matrix based on rotation of element
    :param element:
    :param model_part:
    :param aux_matrix:
    :return:
    """
    # todo make general, now it works for 2 nodes in a 2d space
    if len(element.nodes) == 2:
        rot = calculate_rotation(element.nodes[0].coordinates[None,:], element.nodes[1].coordinates[None,:])
        model_part.set_rotation_matrix(rot, 2)
        rot_matrix = model_part.rotation_matrix

        if rot_matrix is not None:
            return rot_matrix.dot(aux_matrix.dot(rot_matrix.transpose()))

    return aux_matrix


def add_aux_matrix_to_global(
    global_matrix: sparse.lil_matrix,
    aux_matrix,
    elements: List[Element],
    model_part,
    nodes: List[Node] = None,
):

    if elements:
        original_aux_matrix = copy.copy(aux_matrix)
        for element in elements:

            # rotate aux matrix
            aux_matrix = rotate_aux_matrix(element, model_part, original_aux_matrix)

            # loop over each node in modelpart element except the last node
            for node_nr in range(len(element.nodes) - 1):

                # add diagonal part of aux matrix to the global matrix
                for j, id_1 in enumerate(element.nodes[node_nr].index_dof):
                    for k, id_2 in enumerate(element.nodes[node_nr].index_dof):
                        row_index = len(element.nodes[node_nr].index_dof) * node_nr + j
                        col_index = len(element.nodes[node_nr].index_dof) * node_nr + k
                        global_matrix[id_1, id_2] += aux_matrix[row_index, col_index]

                for node_nr_2 in range(node_nr + 1, len(element.nodes)):
                    # add top right part of aux matrix to the global matrix
                    for j, id_1 in enumerate(element.nodes[node_nr].index_dof):
                        for k, id_2 in enumerate(element.nodes[node_nr_2].index_dof):
                            row_index = len(element.nodes[node_nr].index_dof) * node_nr + j
                            col_index = len(element.nodes[node_nr].index_dof) * (node_nr + node_nr_2) + k
                            global_matrix[id_1, id_2] += aux_matrix[row_index, col_index]

                    # add bottom left part of the aux matrix to the global matrix
                    for j, id_1 in enumerate(element.nodes[node_nr_2].index_dof):
                        for k, id_2 in enumerate(element.nodes[node_nr].index_dof):
                            row_index = len(element.nodes[node_nr].index_dof) * (node_nr + node_nr_2) + j
                            col_index = len(element.nodes[node_nr].index_dof) * node_nr + k
                            global_matrix[id_1, id_2] += aux_matrix[row_index, col_index]

            # add last node of the aux matrix to the global matrix
            for j, id_1 in enumerate(element.nodes[-1].index_dof):
                for k, id_2 in enumerate(element.nodes[-1].index_dof):
                    row_index = len(element.nodes[-1].index_dof) * (len(element.nodes) - 1) + j
                    col_index = len(element.nodes[-1].index_dof) * (len(element.nodes) - 1) + k
                    global_matrix[id_1, id_2] += aux_matrix[row_index, col_index]
        return global_matrix

    # add single nodes to the global matrix (for model parts without elements)
    if nodes is not None:
        for node in nodes:
            for j, id_1 in enumerate(node.index_dof):
                for k, id_2 in enumerate(node.index_dof):
                    global_matrix[id_1, id_2] += aux_matrix[j, k]

        return global_matrix


def distance_np(coordinates_array1: np.array, coordinates_array2: np.array, axis=0):
    """
    Calculate distance between 2 coordinate numpy arrays

    :param coordinates_array1:
    :param coordinates_array2:
    :param axis:
    :return:
    """
    return np.sqrt(np.sum((coordinates_array1 - coordinates_array2) ** 2, axis=axis))


def centeroid_np(arr):
    """
    Calculate centroid of numpy array
    :param arr: numpy array
    :return: centroid
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return sum_x / length, sum_y / length, sum_z / length


def find_intersecting_point_element(
    elements, point_coordinates, intersection_tolerance=1e-6
):
    """
    Finds index of the element in an element array which intersects with a given point

    :param elements:
    :param point_coordinates:
    :param intersection_tolerance:
    :return:
    """

    # convert elements to shapely elements for intersection
    shapely_elements = get_shapely_elements(elements)

    # calculate centroids
    centroids = np.array(
        [
            centeroid_np(np.array([node.coordinates for node in element.nodes]))
            for element in elements
        ]
    )

    # set kdtree to quickly search nearest element of the point
    tree = KDTree(centroids)

    # set shapely point for intersection
    point = Point(point_coordinates)
    element_idx = None

    # loop is required because the closest element centroid to the point is not always the centroid of the
    # intersecting element
    for i in range(len(elements)):
        nr_nearest_neighbours = i + 1
        # find nearest neighbour element of point coordinates
        nearest_neighbours = tree.query(point_coordinates, k=nr_nearest_neighbours)
        element_idx = (
            nearest_neighbours[1]
            if isinstance(nearest_neighbours[1], (np.int32, np.int64))
            else nearest_neighbours[1][-1]
        )

        # check if coordinate is in element
        if (
            shapely_elements[element_idx]
            .buffer(intersection_tolerance)
            .intersection(point)
        ):
            return element_idx

    return element_idx


def __create_shapely_element(element: Element):
    """
    Convert element to shapely element
    :param element:
    :return:
    """
    if len(element.nodes) == 2:
        return LineString([node.coordinates for node in element.nodes])
    elif len(element.nodes) > 2:
        return Polygon([node.coordinates for node in element.nodes])


def get_shapely_elements(elements: List[Element]):
    """
    Convert elements to shapely elements for intersection
    :param elements:
    :return:
    """
    return [__create_shapely_element(element) for element in elements]
