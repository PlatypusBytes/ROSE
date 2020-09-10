from src import utils

from src.model_part import ElementModelPart, ConstraintModelPart, ModelPart
from src.boundary_conditions import LoadCondition
from src.geometry import Mesh

from scipy import sparse
import numpy as np
import logging
from typing import List

from src.solver import NewmarkSolver, StaticSolver
from src.exceptions import *

class GlobalSystem:
    def __init__(self):

        self.mesh = Mesh()

        self.global_mass_matrix = None
        self.global_stiffness_matrix = None
        self.global_damping_matrix = None
        self.global_force_vector = None

        self.solver = None
        self.time = None
        self.stage_time_ids = None

        self.model_parts = []  # type: List[ModelPart]

        self.total_n_dof = None

        self.displacements = None
        self.velocities = None
        self.accelerations = None

    def validate_input(self):
        for model_part in self.model_parts:
            model_part.validate_input()

        if logging.getLogger()._cache.__contains__(40) or logging.getLogger()._cache.__contains__(50):
            raise ParameterNotDefinedException(Exception)

    def initialise_model_parts(self):
        """
        Initialises all model parts
        :return:
        """

        for model_part in self.model_parts:
            # initialise model part
            model_part.initialize()

    def __add_aux_matrices_to_global(self, model_part: ElementModelPart):
        """
        Add aux matrices of model part to the global matrices. If the model part does not have elements, the model part
        nodes are used as reference

        :param model_part:
        :return:
        """

        # use nodes as reference if model part does not contain elements
        if model_part.elements:
            node_references = None
        else:
            node_references = model_part.nodes

        # add aux stiffness matrix to global stiffness matrix
        if model_part.aux_stiffness_matrix is not None:
            self.global_stiffness_matrix = utils.add_aux_matrix_to_global(
                self.global_stiffness_matrix,
                model_part.aux_stiffness_matrix,
                model_part.elements,
                model_part, node_references,
            )

        # add aux mass matrix to global mass matrix
        if model_part.aux_mass_matrix is not None:
            self.global_mass_matrix = utils.add_aux_matrix_to_global(
                self.global_mass_matrix,
                model_part.aux_mass_matrix,
                model_part.elements,
                model_part, node_references,
            )

        # add aux damping matrix to global damping matrix
        if model_part.aux_damping_matrix is not None:
            self.global_damping_matrix = utils.add_aux_matrix_to_global(
                self.global_damping_matrix,
                model_part.aux_damping_matrix,
                model_part.elements,
                model_part, node_references,
            )

    def __reshape_aux_matrices(self, model_part: ElementModelPart):
        """
        Reshape aux matrix of model part with the same dimensions as the active dof in the corresponding node

        :param model_part:
        :return:
        """

        # gets number of nodes per element if element exist, else nnodes = 1
        if model_part.elements:
            n_nodes_element = len(model_part.elements[0].nodes)
        else:
            n_nodes_element = 1

        # reshape aux stiffness matrix if exists
        if model_part.aux_stiffness_matrix is not None:
            model_part.aux_stiffness_matrix = utils.reshape_aux_matrix(
                n_nodes_element,
                [model_part.normal_dof, model_part.y_disp_dof, model_part.z_rot_dof],
                model_part.aux_stiffness_matrix,
            )

        # reshape aux mass matrix if exists
        if model_part.aux_mass_matrix is not None:
            model_part.aux_mass_matrix = utils.reshape_aux_matrix(
                n_nodes_element,
                [model_part.normal_dof, model_part.y_disp_dof, model_part.z_rot_dof],
                model_part.aux_mass_matrix,
            )

        # reshape aux damping matrix if exists
        if model_part.aux_damping_matrix is not None:
            model_part.aux_damping_matrix = utils.reshape_aux_matrix(
                n_nodes_element,
                [model_part.normal_dof, model_part.y_disp_dof, model_part.z_rot_dof],
                model_part.aux_damping_matrix,
            )

    def __add_condition_to_global(self, condition: LoadCondition):
        """
        Adds load condition to the global force vector

        :param condition:
        :return:
        """

        for i, node in enumerate(condition.nodes):
            # add load condition on normal displacement dof
            if condition.normal_dof:
                self.global_force_vector[
                    node.index_dof[0], :
                ] += condition.normal_force[i, :]

            # add load condition on y displacement dof
            if condition.y_disp_dof:
                self.global_force_vector[node.index_dof[1], :] += condition.y_force[
                    i, :
                ]

            # add load condition on z rotation dof
            if condition.z_rot_dof:
                self.global_force_vector[node.index_dof[2], :] += condition.z_moment[
                    i, :
                ]

    def __trim_global_matrices_on_indices(self, row_indices: List, col_indices: List):
        """
        Removes items in global stiffness, mass, damping and force vector on row and column indices
        :param row_indices:
        :param col_indices:
        :return:
        """

        self.global_stiffness_matrix = utils.delete_from_lil(
            self.global_stiffness_matrix,
            row_indices=row_indices,
            col_indices=col_indices,
        )
        self.global_mass_matrix = utils.delete_from_lil(
            self.global_mass_matrix, row_indices=row_indices, col_indices=col_indices
        )
        self.global_damping_matrix = utils.delete_from_lil(
            self.global_damping_matrix, row_indices=row_indices, col_indices=col_indices
        )

        self.global_force_vector = utils.delete_from_lil(
            self.global_force_vector, row_indices=row_indices
        )

    def __recalculate_dof(self, removed_indices: np.array):
        """
        Recalculates the total number of degree of freedoms and the index of the nodal dof in the global matrices
        :return:
        """
        i = 0
        for node in self.mesh.nodes:
            for idx, index_dof in enumerate(node.index_dof):
                if index_dof in removed_indices:
                    i -= 1
                    node.index_dof[idx] = None
                    node.set_dof(idx, False)
                else:
                    node.index_dof[idx] = index_dof + i

        self.total_n_dof = self.total_n_dof + i

    def __get_constrained_indices(self):
        """
        Gets indices of constrained dofs in global matrices. A dof is defined as constrained when the dof == False
        :return:
        """

        constrained_indices = []
        # check which nodes are constrained
        for idx in range(len(self.mesh.nodes) - 1, -1, -1):

            # check if normal displacement dof is obsolete
            if not self.mesh.nodes[idx].normal_dof:
                constrained_indices.append(self.mesh.nodes[idx].index_dof[0])

            # check if y displacement dof is obsolete
            if not self.mesh.nodes[idx].y_disp_dof:
                constrained_indices.append(self.mesh.nodes[idx].index_dof[1])

            # check if z rotation dof is obsolete
            if not self.mesh.nodes[idx].z_rot_dof:
                constrained_indices.append(self.mesh.nodes[idx].index_dof[2])

        return constrained_indices

    def trim_all_global_matrices(self):
        """
        Checks which degrees of freedom are obsolete and removes from global matrices

        :return:
        """

        # gets indices without mass
        massless_indices = list(np.flip(np.where(np.isclose(self.global_mass_matrix.diagonal(), 0))[0]))

        # gets indices which are constrained
        constrained_row_indices = self.__get_constrained_indices()

        # combine massless and constrained indices
        obsolete_indices = sorted(list(np.unique(constrained_row_indices + massless_indices)), reverse=True)

        # remove obsolete rows and columns from global matrices
        self.__trim_global_matrices_on_indices(list(obsolete_indices), list(obsolete_indices))

        # recalculate dof numbering
        if len(obsolete_indices) > 0:
            self.__recalculate_dof(np.array(obsolete_indices))

    def initialise_global_matrices(self):
        """
        Inititialises all the global matrices with all element model parts and conditions and constraints

        :return:
        """

        # initialise global lil matrices
        self.global_stiffness_matrix = sparse.lil_matrix(
            (self.total_n_dof, self.total_n_dof)
        )
        self.global_damping_matrix = sparse.lil_matrix(
            (self.total_n_dof, self.total_n_dof)
        )
        self.global_mass_matrix = sparse.lil_matrix(
            (self.total_n_dof, self.total_n_dof)
        )
        self.global_force_vector = sparse.lil_matrix((self.total_n_dof, len(self.time)))

        # add aux matrices to global matrices for each element model part and load model part
        for model_part in self.model_parts:
            if isinstance(model_part, ElementModelPart):
                self.__reshape_aux_matrices(model_part)
                self.__add_aux_matrices_to_global(model_part)
            if isinstance(model_part, LoadCondition):
                self.__add_condition_to_global(model_part)

        # sets constraint conditions, it is important that this is done after initialising the global matrices with the
        # element and load model parts
        for model_part in self.model_parts:
            if isinstance(model_part, ConstraintModelPart):
                model_part.set_constraint_condition()

        # removes obsolete rows and columns from the global matrices
        self.trim_all_global_matrices()

    def initialise_ndof(self):
        """
        Initialise total number of degrees of freedom in the global system and sets nodal dof index in the global
        matrices
        :return:
        """
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

    def set_stage_time_ids(self):
        """
        Find indices of unique time steps
        :return:
        """
        diff = np.diff(self.time)
        new_dt_idxs = sorted(np.unique(diff.round(decimals=7), return_index=True)[1])
        self.stage_time_ids = np.append(new_dt_idxs, len(self.time) - 1)

    def update_model_parts(self):
        """
        Updates all model parts
        :return:
        """

        for model_part in self.model_parts:
            model_part.update()

    def initialise(self):
        """
        Initialises model parts, degrees of freedom, global matrices and solver

        :return:
        """
        self.initialise_model_parts()
        self.initialise_ndof()
        self.initialise_global_matrices()

        self.set_stage_time_ids()

        self.solver.initialise(self.total_n_dof, self.time)

    def update(self, start_time_id, end_time_id):
        """
        Updates model parts and solver
        :param start_time_id:
        :param end_time_id:
        :return:
        """
        self.update_model_parts()
        self.solver.update(start_time_id)

    def calculate_stage(self, start_time_id, end_time_id):
        """
        Calculates the global system
        :return:
        """

        # transfer matrices to compressed sparsed column matrices
        M = self.global_mass_matrix.tocsc()
        C = self.global_damping_matrix.tocsc()
        K = self.global_stiffness_matrix.tocsc()
        F = self.global_force_vector.tocsc()

        # run_stages with Newmark solver
        if isinstance(self.solver, NewmarkSolver):
            self.solver.calculate(M, C, K, F, start_time_id, end_time_id)

        # run_stages with Static solver
        if isinstance(self.solver, StaticSolver):
            self.solver.calculate(K, F, start_time_id, end_time_id)

        self.assign_results_to_nodes()

    def finalise(self):
        """
        Finalises calculation
        :return:
        """

        self.displacements = self.solver.u
        self.velocities = self.solver.v
        self.accelerations = self.solver.a

        self.time = self.solver.time

    def _assign_result_to_node(self, node):
        """
        Assigns solver results to a node
        :param node:
        :return:
        """
        node_ids_dofs = list(node.index_dof[node.index_dof != None])
        node.assign_result(
            self.solver.u[:, node_ids_dofs],
            self.solver.v[:, node_ids_dofs],
            self.solver.a[:, node_ids_dofs],
        )
        return node

    def assign_results_to_nodes(self):
        """
        Assigns all solver results to all nodes in the mesh
        :return:
        """
        self.mesh.nodes = list(
            map(lambda node: self._assign_result_to_node(node), self.mesh.nodes)
        )

    def main(self):

        self.validate_input()
        self.initialise()

        # calculate stages
        for i in range(len(self.stage_time_ids) - 1):
            self.update(self.stage_time_ids[i], self.stage_time_ids[i + 1])
            self.calculate_stage(self.stage_time_ids[i], self.stage_time_ids[i + 1])

        self.finalise()
