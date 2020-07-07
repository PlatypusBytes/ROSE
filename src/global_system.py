from src import track, soil, utils, geometry

from src.model_part import ElementModelPart, ConstraintModelPart, ModelPart
from src.boundary_conditions import CauchyCondition
from src.geometry import Mesh, Node, Element
from scipy import sparse
import numpy as np
from typing import List

class GlobalSystem:
    def __init__(self):

        self.mesh = Mesh()

        self.global_mass_matrix = None
        self.global_stiffness_matrix = None
        self.global_damping_matrix = None
        self.global_force_vector = None

        self.solver = None

        self.time = None

        self.model_parts = []  # type: List[ModelPart]

        self.total_n_dof = None

    def __update_nodal_dofs(self, node, model_part):
        node.rotation_dof = model_part.rotation_dof if model_part.rotation_dof else node.rotation_dof
        node.x_disp_dof = model_part.x_disp_dof if model_part.x_disp_dof else node.x_disp_dof
        node.y_disp_dof = model_part.y_disp_dof if model_part.y_disp_dof else node.y_disp_dof

    def initialise_model_parts(self):

        for model_part in self.model_parts:
            model_part.initialize()
            if isinstance(model_part, ElementModelPart):
                for node in model_part.nodes:
                    self.__update_nodal_dofs(node, model_part)


    def __add_aux_matrices_to_global(self, model_part):
        """
        Add aux matrices of model part to the global matrices. If the model part does not have elements, the model part
        nodes are used as reference

        :param model_part:
        :return:
        """
        if model_part.elements:
            node_references = None
        else:
            node_references = model_part.nodes

        if model_part.aux_stiffness_matrix is not None:
            self.global_stiffness_matrix = utils.add_aux_matrix_to_global(
                self.global_stiffness_matrix, model_part.aux_stiffness_matrix, model_part.elements, node_references)

        if model_part.aux_mass_matrix is not None:
            self.global_mass_matrix = utils.add_aux_matrix_to_global(
                self.global_mass_matrix, model_part.aux_mass_matrix, model_part.elements, node_references)

        if model_part.aux_damping_matrix is not None:
            self.global_damping_matrix = utils.add_aux_matrix_to_global(
                self.global_damping_matrix, model_part.aux_damping_matrix, model_part.elements, node_references)

    def __reshape_aux_matrices(self, model_part: ElementModelPart):
        """
        Reshape aux matrix of model part with the same dimensions as the dof in the corresponding node

        :param model_part:
        :return:
        """

        if model_part.elements:
            n_nodes_element = len(model_part.elements[0].nodes)
        else:
            n_nodes_element = 1

        if model_part.aux_stiffness_matrix is not None:
            model_part.aux_stiffness_matrix = utils.reshape_aux_matrix(n_nodes_element, [model_part.x_disp_dof,
                                                                                         model_part.y_disp_dof,
                                                                                         model_part.rotation_dof],
                                                                       model_part.aux_stiffness_matrix)

        if model_part.aux_mass_matrix is not None:
            model_part.aux_mass_matrix = utils.reshape_aux_matrix(n_nodes_element,
                                                                  [model_part.x_disp_dof, model_part.y_disp_dof,
                                                                   model_part.rotation_dof],
                                                                  model_part.aux_mass_matrix)

        if model_part.aux_damping_matrix is not None:
            model_part.aux_damping_matrix = utils.reshape_aux_matrix(n_nodes_element,
                                                                     [model_part.x_disp_dof, model_part.y_disp_dof,
                                                                      model_part.rotation_dof],
                                                                     model_part.aux_damping_matrix)

    def __add_condition_to_global(self, condition):

        for i, node in enumerate(condition.nodes):
            if condition.rotation_dof:
                self.global_force_vector[node.index_dof[0], :] += condition.moment[i, :]
            if condition.y_disp_dof:
                self.global_force_vector[node.index_dof[1], :] += condition.y_force[i, :]
            if condition.x_disp_dof:
                self.global_force_vector[node.index_dof[2], :] += condition.x_force[i, :]


    def __trim_global_matrices_on_indices(self, row_indices, col_indices):

        self.global_stiffness_matrix = utils.delete_from_csr(self.global_stiffness_matrix,
                                                        row_indices=row_indices,
                                                        col_indices=col_indices)
        self.global_mass_matrix = utils.delete_from_csr(self.global_mass_matrix,
                                                   row_indices=row_indices,
                                                   col_indices=col_indices)
        self.global_damping_matrix = utils.delete_from_csr(self.global_damping_matrix,
                                                      row_indices=row_indices,
                                                      col_indices=col_indices)

        self.global_force_vector = utils.delete_from_csr(self.global_force_vector, row_indices=row_indices)


    def __recalculate_dof(self):
        i = 0
        for node in self.mesh.nodes:
            for idx, index_dof in enumerate(node.index_dof):
                if index_dof is None:
                    i -= 1
                else:
                    node.index_dof[idx] = index_dof + i

        self.total_n_dof = self.total_n_dof + i

    def trim_all_global_matrices(self):
        for idx in range(len(self.mesh.nodes) - 1, -1, -1):
            if not self.mesh.nodes[idx].x_disp_dof:
                row_indices = [self.mesh.nodes[idx].index_dof[2]]
                col_indices = [self.mesh.nodes[idx].index_dof[2]]

                self.mesh.nodes[idx].index_dof[2] = None

                self.__trim_global_matrices_on_indices(row_indices, col_indices)

            if not self.mesh.nodes[idx].y_disp_dof:
                row_indices = [self.mesh.nodes[idx].index_dof[1]]
                col_indices = [self.mesh.nodes[idx].index_dof[1]]

                self.mesh.nodes[idx].index_dof[1] = None

                self.__trim_global_matrices_on_indices(row_indices, col_indices)

            if not self.mesh.nodes[idx].rotation_dof:
                row_indices = [self.mesh.nodes[idx].index_dof[0]]
                col_indices = [self.mesh.nodes[idx].index_dof[0]]

                self.mesh.nodes[idx].index_dof[0] = None

                self.__trim_global_matrices_on_indices(row_indices, col_indices)

        self.__recalculate_dof()

    def initialise_global_matrices(self):

        self.global_stiffness_matrix = sparse.csr_matrix((self.total_n_dof, self.total_n_dof))
        self.global_damping_matrix = sparse.csr_matrix((self.total_n_dof, self.total_n_dof))
        self.global_mass_matrix = sparse.csr_matrix((self.total_n_dof, self.total_n_dof))
        self.global_force_vector = sparse.csr_matrix((self.total_n_dof, len(self.time)))

        for model_part in self.model_parts:
            if isinstance(model_part, ElementModelPart):
                self.__reshape_aux_matrices(model_part)
                self.__add_aux_matrices_to_global(model_part)

            if isinstance(model_part, CauchyCondition):
                self.__add_condition_to_global(model_part)

        for model_part in self.model_parts:
            if isinstance(model_part, ConstraintModelPart):
                model_part.set_constraint_condition()

        self.trim_all_global_matrices()

    def initialise_ndof(self):
        ndof = 0
        index_dof = 0
        for node in self.mesh.nodes:
            node.index_dof[0] = index_dof
            index_dof += 1
            node.index_dof[1] = index_dof
            index_dof += 1
            node.index_dof[2] = index_dof
            index_dof += 1
            ndof = ndof + len(node.index_dof)

        self.total_n_dof = ndof


    def calculate(self):
        self.solver.initialise(self.total_n_dof, self.time)
        self.solver.calculate(self.global_mass_matrix, self.global_damping_matrix,
               self.global_stiffness_matrix, self.global_force_vector,
               self.time[1] - self.time[0], self.time[-1], t_start=self.time[0])



    def main(self):
        self.initialise_model_parts()
        self.initialise_ndof()
        self.initialise_global_matrices()
        self.calculate()



















    # todo check what is required below
    #
    #
    #
    #
    # def __add_track_to_geometry(self):
    #     track_nodes = self.track.nodes
    #     track_elements = self.track.elements
    #     self.nodes = np.append(self.nodes, track_nodes)
    #     self.elements = np.append(self.elements, track_elements)
    #
    #     return track_nodes, track_elements
    #
    # def __add_soil_to_geometry(self):
    #     soil_nodes = self.soil.nodes
    #     soil_elements = self.soil.elements
    #
    #     self.nodes = np.append(self.nodes, soil_nodes)
    #     self.elements = np.append(self.elements, soil_elements)
    #
    # def _add_model_part_to_geometry(self, model_part):
    #     model_part_nodes = model_part.mesh.nodes
    #     model_part_elements = model_part.mesh.elements
    #
    #     self.nodes = np.append(self.nodes, model_part_nodes)
    #     self.elements = np.append(self.elements, model_part_elements)
    #
    #
    # def __get_unique_items(self, L):
    #     seen = set()
    #     seen2 = set()
    #     seen_add = seen.add
    #     seen2_add = seen2.add
    #     for item in L:
    #         if item in seen:
    #             seen2_add(item)
    #         else:
    #             seen_add(item)
    #     return np.array(seen2)
    #
    # def __merge_geometry(self):
    #     self.nodes = self.__get_unique_items(self.nodes)
    #     self.elements = self.__get_unique_items(self.elements)
    #
    # def set_geometry(self):
    #
    #
    #     for model_part in self.model_parts:
    #         model_part.set_geometry()
    #         self._add_model_part_to_geometry(model_part)
    #
    #     # self.track.set_geometry()
    #
    #     # track_nodes, _ = self.__add_track_to_geometry()
    #     # self.__add_soil_to_geometry()
    #
    #     self.__merge_geometry()




    # def __add_soil_to_global_stiffness_matrix(self):
    #     for i in range(self.__n_sleepers):
    #         self.global_stiffness_matrix[i + self.rail.ndof, i + self.rail.ndof] \
    #             += self.soil.aux_stiffness_matrix[0, 0]
    #         self.global_stiffness_matrix[i + self.rail.ndof + self.__n_sleepers, i + self.rail.ndof] \
    #             += self.soil.aux_stiffness_matrix[1, 0]
    #         self.global_stiffness_matrix[i + self.rail.ndof, i + self.rail.ndof + self.__n_sleepers] \
    #             += self.soil.aux_stiffness_matrix[0, 1]
    #         self.global_stiffness_matrix[i + self.rail.ndof + self.__n_sleepers, i + self.rail.ndof + self.__n_sleepers] \
    #             += self.soil.aux_stiffness_matrix[1, 1]

    # def set_global_stiffness_matrix(self):
    #     """
    #
    #     :return:
    #     """
    #     self.global_stiffness_matrix = sparse.csr_matrix((self.total_n_dof, self.total_n_dof))
    #
    #
    #
    #     self.track.set_global_stiffness_matrix()
    #     self.soil.set_global_stiffness_matrix()








        # self.soil.set_aux_stiffness_matrix()
        # self.__add_soil_to_global_stiffness_matrix()


    # def apply_no_disp_boundary_condition(self):
    #     bottom_node_idxs = np.arange(self.rail.ndof + self.__n_sleepers,
    #                                  self.rail.ndof + 2 * self.__n_sleepers).tolist()
    #
    #     self.force = utils.delete_from_csr(self.force, row_indices=bottom_node_idxs)
    #     self.global_mass_matrix = utils.delete_from_csr(self.global_mass_matrix, row_indices=bottom_node_idxs,
    #                                                     col_indices=bottom_node_idxs)
    #     self.global_stiffness_matrix = utils.delete_from_csr(self.global_stiffness_matrix, row_indices=bottom_node_idxs,
    #                                                          col_indices=bottom_node_idxs)
    #     self.global_damping_matrix = utils.delete_from_csr(self.global_damping_matrix, row_indices=bottom_node_idxs,
    #                                                        col_indices=bottom_node_idxs)

    # def initialise_track(self, rail, sleeper, rail_pads):
    #     self.track.rail = rail
    #     self.track.sleeper = sleeper
    #     self.track.rail_pads = rail_pads
    #
    #     self.track.initialise_track()



# def delete_from_global_matrices(global_stiffness_matrix, global_mass_matrix, global_damping_matrix, global_b_matrix,
#                                 row_indices, col_indices):
#
#     global_stiffness_matrix = utils.delete_from_csr(global_stiffness_matrix,
#                                                     row_indices=row_indices,
#                                                     col_indices=col_indices)
#     global_mass_matrix = utils.delete_from_csr(global_mass_matrix,
#                                                     row_indices=row_indices,
#                                                     col_indices=col_indices)
#     global_damping_matrix = utils.delete_from_csr(global_damping_matrix,
#                                                     row_indices=row_indices,
#                                                     col_indices=col_indices)
#
#     global_b_matrix = utils.delete_from_csr(global_b_matrix, row_indices=row_indices)
#
#     return global_stiffness_matrix, global_mass_matrix, global_damping_matrix, global_b_matrix




