from scipy import sparse
import numpy as np
from shapely.geometry import LineString, Polygon

from typing import List
# from src.boundary_conditions import LoadCondition
from src.geometry import Node, Element

def init_aux_matrix(n_nodes, dofs):
    return np.zeros((n_nodes*sum(dofs), n_nodes*sum(dofs)))

def reshape_aux_matrix(n_nodes, dofs, aux_matrix):
    new_aux_matrix = np.zeros((n_nodes*len(dofs), n_nodes*len(dofs)))

    dofs = np.asarray(dofs)
    for i in range(n_nodes):
        for j in range(n_nodes):
            for k in range(len(dofs)):
                if dofs[k]:
                    for l in range(len(dofs)):
                        if dofs[l]:
                            new_aux_matrix[i * len(dofs) + k, j * len(dofs) + l] = \
                                aux_matrix[i * sum(dofs) + sum(dofs[0:k]), j * sum(dofs) + sum(dofs[0:l])]

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


def add_aux_matrix_to_global(global_matrix: sparse.lil_matrix, aux_matrix, elements: List[Element],
                             nodes: List[Node] = None):
    """

    :param global_matrix:
    :param aux_matrix:
    :param elements:
    :return:
    """
    if nodes is None:
        nodes = []
    if elements:
        for element in elements:
            for node_nr in range(len(element.nodes) - 1):
                # add diagonal
                for j, id_1 in enumerate(element.nodes[node_nr].index_dof):
                    for k, id_2 in enumerate(element.nodes[node_nr].index_dof):
                        global_matrix[id_1, id_2] += aux_matrix[len(element.nodes[node_nr].index_dof) * node_nr + j,
                                                                len(element.nodes[node_nr].index_dof) * node_nr + k]
                # add interaction
                for node_nr_2 in range(node_nr+1, len(element.nodes)):
                    for j, id_1 in enumerate(element.nodes[node_nr].index_dof):
                        for k, id_2 in enumerate(element.nodes[node_nr_2].index_dof):
                            global_matrix[id_1, id_2] += aux_matrix[len(element.nodes[node_nr].index_dof) * node_nr + j,
                                                                    len(element.nodes[node_nr].index_dof) * (node_nr + node_nr_2) + k]
                    for j, id_1 in enumerate(element.nodes[node_nr_2].index_dof):
                        for k, id_2 in enumerate(element.nodes[node_nr].index_dof):
                            global_matrix[id_1, id_2] += aux_matrix[len(element.nodes[node_nr].index_dof) * (node_nr + node_nr_2) + j,
                                                                    len(element.nodes[node_nr].index_dof) * node_nr + k]

            # add last node
            for j, id_1 in enumerate(element.nodes[-1].index_dof):
                for k, id_2 in enumerate(element.nodes[-1].index_dof):
                    global_matrix[id_1, id_2] += aux_matrix[len(element.nodes[-1].index_dof) * (len(element.nodes)-1) + j,
                                                            len(element.nodes[-1].index_dof) * (len(element.nodes)-1) + k]
        return global_matrix

    elif nodes is not None:
        for node in nodes:
            for j, id_1 in enumerate(node.index_dof):
                for k, id_2 in enumerate(node.index_dof):
                    global_matrix[id_1, id_2] += aux_matrix[j, k]

        return global_matrix

def centeroidnp(arr):
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


def get_shapely_elements(elements: List[Element]):
    # convert elements to shapely elements for intersection
    shapely_elements = []
    for element in elements:
        if len(element.nodes) == 2:
            shapely_elements.append(LineString(
                [node.coordinates for node in element.nodes]))
        elif len(element.nodes) > 2:
            shapely_elements.append(Polygon([node.coordinates for node in element.nodes]))

    return shapely_elements
