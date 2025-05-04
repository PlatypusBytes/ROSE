from rose.model.geometry import Node, Element
from rose.model.irregularities import RailIrregularities

import numpy as np
from scipy import sparse, interpolate
import math

from typing import List
import copy


def filer_location(locations: np.array, lower_limit: float, upper_limit: float):
    """
    Filter location array which is outside lower and upper limit. If location is outside limits, it is set to the
    corresponding limit

    :param locations: location array to be filtered
    :param lower_limit: lower limit of location array
    :param upper_limit: upper limit of location array
    :return:
    """

    # filter location array outside limits and set location to limit
    locations[locations > upper_limit] = upper_limit
    locations[locations < lower_limit] = lower_limit

    return locations


def filter_data_outside_range(data: np.array, locations: np.array, lower_limit: float, upper_limit: float):
    """
    Filters data outside upper and lower location limits

    :param data: data array to be filtered
    :param locations: location array to ben checked for limits
    :param lower_limit: lower limit of location array
    :param upper_limit: upper limit of location array
    :return:
    """

    # Set data at 0 when its located outside the limits
    data[locations > upper_limit] = 0
    data[locations < lower_limit] = 0

    return data


def interpolate_cumulative_distance_on_nodes(
    cumulative_distances: np.array, coordinates: np.array, new_cumulative_distances: np.array
):
    """
    Interpolate cumulative distances on coordinates

    :param cumulative_distances: cumulative distance array of the nodes
    :param coordinates: nodal coordinates
    :param new_cumulative_distances: new cumulative distance array to be interpolated with
    :return:
    """

    # set interpolation functions in x,y,z direction
    fx = interpolate.interp1d(cumulative_distances, coordinates[:, 0])
    fy = interpolate.interp1d(cumulative_distances, coordinates[:, 1])
    fz = interpolate.interp1d(cumulative_distances, coordinates[:, 2])

    # interpolate in x,y,z direction
    new_x_coords = fx(new_cumulative_distances)
    new_y_coords = fy(new_cumulative_distances)
    new_z_coords = fz(new_cumulative_distances)

    return np.array([new_x_coords, new_y_coords, new_z_coords]).transpose()


def calculate_cum_distance_from_velocity(time, velocities):
    """
    Calculate cumulative distance by multiplying the time steps by the velocities

    :param time: time discretisation
    :param velocities: array of velocity each time step
    :return:
    """
    cum_distances = np.append(
                    0,
                    np.cumsum((time[1:] - time[:-1]) * velocities[:-1])
                    )
    return cum_distances


def reshape_aux_matrix(n_nodes_element: int, dofs: list, aux_matrix: np.ndarray) -> np.ndarray:
    """
    Reshapes aux matrix of the model part based on the total possible degrees of freedom (active and inactive). This
    function is required to easily fill the global matrices

    :param n_nodes_element: number of nodes per element
    :param dofs: list of all possible degrees of freedoms per element
    :param aux_matrix: current auxiliary matrix
    :return:
    """

    # inititalise reshaped auxiliary matrix
    new_aux_matrix = np.zeros((n_nodes_element * len(dofs), n_nodes_element * len(dofs)))

    # loop over nodes per element
    for i in range(n_nodes_element):
        for j in range(n_nodes_element):
            # loop over possible degrees of freedom
            for k in range(len(dofs)):
                if dofs[k]:
                    for l in range(len(dofs)):
                        if dofs[l]:
                            # if degree of freedom is active, add data from original aux matrix to reshaped aux matrix
                            new_aux_matrix[
                                i * len(dofs) + k, j * len(dofs) + l
                            ] = aux_matrix[
                                i * sum(dofs) + sum(dofs[0:k]),
                                j * sum(dofs) + sum(dofs[0:l]),
                            ]

    return new_aux_matrix


def delete_from_lil(mat: sparse.lil_matrix, row_indices=[], col_indices=[]):
    """
    Remove the rows  and columns  from the lil sparse matrix

    WARNING: Indices of altered axes are reset in the returned matrix

    :param mat: matrix to be trimmed
    :param row_indices: row indices to be removed
    :param col_indices: column indices to be removed
    :return:
    """

    # fill rows and cols if row_indices and col_indices are to be removed
    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    # remove both rows and columns from matrix
    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:, col_mask]

    # remove only rows from matrix
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]

    # remove only columns from matrix
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat


def calculate_point_rotation(coord1: np.ndarray, coord2: np.ndarray):
    """
    Calculates rotation between 2 points

    :param coord1: coordinates point 1
    :param coord2: coordinates point 2
    :return:
    """

    # initialise rotation
    rot = 0

    # check if both x coordinates are equal
    is_x_equal = math.isclose(coord2[0],coord1[0])

    # apply a 90 degree rotation if x coordinates are equal
    if is_x_equal:
        return np.pi / 2 * np.sign((coord2[1] - coord1[1]))

    # apply a 180 degree rotation if first x coord is smaller than the second
    if coord2[0] < coord1[0]:
        rot += np.pi

    # calculate rotation between the coordinates if the x coordinates are not equal
    rot += math.atan((coord2[1] - coord1[1]) / (coord2[0] - coord1[0]))

    return rot


def calculate_rotation(coord1: np.ndarray, coord2: np.ndarray):
    """
    Calculates rotation between 2 coordinate arrays in a 2d space.

    :param coord1: first coordinate array
    :param coord2: second coordinate array
    :return:
    """
    # todo make general, now it works for 2 nodes in a 2d space

    # initialise rotation
    rot = np.zeros(coord1.shape[0])

    # check if both x coordinates are equal
    is_x_equal = np.isclose(coord2[:,0] - coord1[:,0], 0)

    # apply a 90 degree rotation if x coordinates are equal
    rot[is_x_equal] = np.pi/2 * np.sign((coord2[is_x_equal, 1] - coord1[is_x_equal, 1]))

    # if second x coordinate is smaller than the first x coordinate, add pi to the rotation
    rot[coord2[:, 0] < coord1[:, 0]] += np.pi

    # calculate rotation between the coordinates if the x coordinates are not equal
    rot[~is_x_equal] += np.arctan((coord2[:, 1] - coord1[:, 1])[~is_x_equal] /
                                  (coord2[:, 0] - coord1[:, 0])[~is_x_equal])

    return rot


def rotate_point_around_z_axis(rotation: np.ndarray, point_vector: np.ndarray):
    """
    Rotates a point around the z-axis

    :param rotation: rotation array in radians
    :param point_vector: vector of global values [x-direction, y-direction, z-rotation] to be rotated
    :return:
    """

    if isinstance(rotation, float):
        rot_matrix = np.zeros((3, 3))
        rot_matrix[0, 0] = math.cos(rotation)
        rot_matrix[1, 1] = rot_matrix[0, 0]
        rot_matrix[0, 1] = math.sin(rotation)
        rot_matrix[1, 0] = -rot_matrix[0, 1]
        rot_matrix[2, 2] = 1

        return rot_matrix.dot(point_vector)

    # set rotation matrix
    rot_matrix = np.zeros((len(rotation), 3, 3))
    rot_matrix[:, 0, 0] = np.cos(rotation)
    rot_matrix[:, 1, 1] = np.cos(rotation)
    rot_matrix[:, 0, 1] = np.sin(rotation)
    rot_matrix[:, 1, 0] = -rot_matrix[:, 0, 1]
    rot_matrix[:, 2, 2] = 1

    rotated_point = np.zeros(point_vector.shape)

    # rotate each time step
    for idx, (mat, point) in enumerate(zip(rot_matrix,point_vector)):
        rotated_point[idx] = mat.dot(point)

    return rotated_point


def rotate_force_vector(element: Element, contact_model_part, force_vector: np.array):
    """
    Rotates force vector based on rotation of element, currently only works on 2 noded elements.

    :param element: elements within current model part
    :param contact_model_part: model part on which the force vector is located
    :param force_vector: force vector to be rotated
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

    :param element: elements within current model part
    :param model_part: current model part of which the auxiliary matrix is to be rotated
    :param aux_matrix: auxiliary matrix to be rotated
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
    """
    Adds auxiliary matrix to the global matrix

    :param global_matrix: sparse global matrix
    :param aux_matrix: auxiliary matrix to be added
    :param elements: list of elements in current model part
    :param model_part: current model part of which the auxiliary matrix is to be added to the global matrix
    :param nodes: list of nodes in current model part
    :return:
    """

    global_matrix = global_matrix.toarray()

    if len(elements) > 0:
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

        global_matrix = sparse.lil_matrix(global_matrix)
        return global_matrix

    # add single nodes to the global matrix (for model parts without elements)
    if nodes is not None:
        for node in nodes:
            for j, id_1 in enumerate(node.index_dof):
                for k, id_2 in enumerate(node.index_dof):
                    global_matrix[id_1, id_2] += aux_matrix[j, k]

        global_matrix = sparse.lil_matrix(global_matrix)
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


def generate_rail_irregularities(wheels: List, time, **kwargs):
    """
    Generates rail irregularities at all wheels in a list

    :param wheels: list of wheels if the train
    :param time: all time steps
    :param kwargs: key word arguments for RailIrregularities
    :return:
    """

    # initialise irregularity matrix
    irregularities_at_wheels = np.zeros((len(wheels), len(time)))

    # generate rail irregularities for each wheel
    for idx, wheel in enumerate(wheels):
        irregularities = RailIrregularities(wheel.distances,**kwargs)
        irregularities_at_wheels[idx, :] = irregularities.irregularities

    return irregularities_at_wheels

