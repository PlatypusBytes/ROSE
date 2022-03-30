from rose.model.model_part import ElementModelPart, ConstraintModelPart, ModelPart
from rose.model.boundary_conditions import LoadCondition
from rose.model.geometry import Mesh
from rose.model.exceptions import *

from solvers.newmark_solver import NewmarkSolver
from solvers.static_solver import StaticSolver
from rose.model.solver import ZhaiSolver, Solver
from rose.model import utils

from scipy import sparse
import numpy as np
import logging
from typing import List


class GlobalSystem:
    """
       Class which the global system, with this system the following formula van be calculated: K·u + C·v + M·a = F.

       The class includes the mesh, the global matrices and vectors, the solver, the time discretisation, the model
       parts and the calculated displacements, velocities and accelerations of the system

       :Attributes:

           - :self.mesh:                        The mesh within the system
           - :self.global_mass_matrix:          The global mass matrix M of the system
           - :self.global_stiffness_matrix:     The global stiffness matrix K of the system
           - :self.global_damping_matrix:       The global damping matrix C of the system
           - :self.global_force_vector:         The global force vector F of the system
           - :self.solver:                      The solver which is used to solve the system
           - :self.model_parts:                 List of model parts which are present within the model
           - :self.total_n_dof:                 Total number of degrees of freedom in the while system
           - :self.time:                        The time discretisation
           - :self.initialisation_time:         The time discretisation, in which the external force is gradually applied
           - :self.stage_time_ids:              The time indices in which the time step size changes
           - :self.time_out:                    The time discretisation which is used for the output
           - :self.displacements_out:           The output displacement results for each node in the system
           - :self.velocities_out:              The output velocity results for each node in the system
           - :self.accelerations_out:           The output acceleration results for each node in the system
           - :self.g:                           Gravity constant
           - :self.is_rayleigh_damping:         Boolean which is true if rayleigh damping is taken into account
           - :self.damping_ratio:               Rayleigh damping ratio
           - :self.radial_frequency_one:        First rayleigh radial frequency
           - :self.radial_frequency_two:        Second rayleigh radial frequency

    """
    def __init__(self):

        self.mesh: Mesh = Mesh()

        self.global_mass_matrix: sparse = None
        self.global_stiffness_matrix: sparse = None
        self.global_damping_matrix: sparse = None
        self.global_force_vector: sparse = None

        self.solver: Solver = None
        self.model_parts: List[ModelPart] = []
        self.total_n_dof: int = None

        self.time: np.ndarray = None
        self.initialisation_time: np.ndarray = None
        self.stage_time_ids: np.ndarray = None
        self.time_out: np.ndarray = None

        self.displacements_out: np.ndarray = None
        self.velocities_out: np.ndarray = None
        self.accelerations_out: np.ndarray = None

        self.g: float = 9.81

        self.is_rayleigh_damping: bool = False
        self.damping_ratio: float = 0
        self.radial_frequency_one: float = 0
        self.radial_frequency_two: float = 0

    def validate_input(self):
        """
        Validates input and adds errors to the logger
        :return:
        """
        for model_part in self.model_parts:
            model_part.validate_input()

        if logging.getLogger()._cache.__contains__(40) or logging.getLogger()._cache.__contains__(50):
            raise ParameterNotDefinedException(Exception)

    def initialise_model_parts(self):
        """
        Initialises all model parts
        :return:
        """

        print("Initialising model parts")
        for model_part in self.model_parts:
            # initialise model part
            model_part.initialize()

    def __add_aux_matrices_to_global(self, model_part: ElementModelPart):
        """
        Add aux matrices of model part to the global matrices. If the model part does not have elements, the model part
        nodes are used as reference

        :param model_part: Element model part
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

    @staticmethod
    def __reshape_aux_matrices(model_part: ElementModelPart):
        """
        Reshape aux matrix of model part with the same dimensions as the active dof in the corresponding node

        :param model_part: Element model part
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

        :param condition: Load condition
        :return:
        """
        self.global_force_vector = self.global_force_vector.toarray()
        for i, node in enumerate(condition.nodes):
            # add load condition on normal displacement dof
            if condition.x_disp_dof:
                self.global_force_vector[
                    node.index_dof[0], :
                ] += condition.x_force_matrix[i, :]

            # add load condition on y displacement dof
            if condition.y_disp_dof:
                self.global_force_vector[node.index_dof[1], :] += condition.y_force_matrix[
                    i, :
                ]

            # add load condition on z rotation dof
            if condition.z_rot_dof:
                self.global_force_vector[node.index_dof[2], :] += condition.z_moment_matrix[
                    i, :
                ]
        self.global_force_vector = sparse.lil_matrix(self.global_force_vector)

    def trim_global_matrices_on_indices(self, row_indices: List, col_indices: List):
        """
        Removes items in global stiffness, mass, damping and force vector on row and column indices

        :param row_indices: row indices which are to be removed
        :param col_indices: column indices which are to be removed
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

    def recalculate_dof(self, removed_indices: np.array):
        """
        Recalculates the total number of degree of freedoms and the index of the nodal dof in the global matrices

        :param removed_indices: indices which are removed from the global system
        :return:
        """

        i = 0  # Negative counter which keeps track of the amount of removed indices
        for node in self.mesh.nodes:
            for idx, index_dof in enumerate(node.index_dof):

                # if degree of freedom index is removed from the global system, also remove the index from the node
                if index_dof in removed_indices:
                    i -= 1
                    node.index_dof[idx] = None
                    node.set_dof(idx, False)
                elif index_dof is None:
                    node.set_dof(idx, False)

                # if degree of freedom index is still present, reset the index value
                else:
                    node.index_dof[idx] = index_dof + i

        # recalculate the total amount of active degrees of freedom in the whole system
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
            if not self.mesh.nodes[idx].x_disp_dof:
                dof_idx = self.mesh.nodes[idx].index_dof[0]
                if dof_idx is not None:
                    constrained_indices.append(dof_idx)

            # check if y displacement dof is obsolete
            if not self.mesh.nodes[idx].y_disp_dof:
                dof_idx = self.mesh.nodes[idx].index_dof[1]
                if dof_idx is not None:
                    constrained_indices.append(dof_idx)

            # check if z rotation dof is obsolete
            if not self.mesh.nodes[idx].z_rot_dof:
                dof_idx = self.mesh.nodes[idx].index_dof[2]
                if dof_idx is not None:
                    constrained_indices.append(dof_idx)

        return constrained_indices

    def trim_all_global_matrices(self):
        """
        Checks which degrees of freedom are obsolete and removes from global matrices

        :return:
        """

        # gets indices without mass
        massless_indices = list(np.flip(np.where(np.isclose(self.global_mass_matrix.diagonal(), 0))[0]))

        #gets indices without stiffness
        stiffnessless_indices = list(np.flip(np.where(np.isclose(self.global_stiffness_matrix.diagonal(), 0))[0]))

        # gets indices which are constrained
        constrained_row_indices = self.__get_constrained_indices()

        # combine massless and constrained indices
        obsolete_indices = sorted(list(np.unique(constrained_row_indices + massless_indices + stiffnessless_indices)), reverse=True)

        # remove obsolete rows and columns from global matrices
        self.trim_global_matrices_on_indices(list(obsolete_indices), list(obsolete_indices))

        # recalculate dof numbering
        self.recalculate_dof(np.array(obsolete_indices))

    def add_model_parts_to_global_matrices(self):
        """
        Adds data from model parts to the global matrices

        :return:
        """
        # add aux matrices to global matrices for each element model part and load model part
        for model_part in self.model_parts:
            if isinstance(model_part, ElementModelPart):
                self.__reshape_aux_matrices(model_part)
                self.__add_aux_matrices_to_global(model_part)
            if isinstance(model_part, LoadCondition):
                self.__add_condition_to_global(model_part)

        # sets constraint conditions, it is important that this is done after initialising the global matrices with the
        # element and load model parts.
        for model_part in self.model_parts:
            if isinstance(model_part, ConstraintModelPart):
                model_part.set_constraint_condition()

        # removes obsolete rows and columns from the global matrices
        self.trim_all_global_matrices()

    def calculate_initial_displacement(self):
        """
        Calculates initial displacement of the system
        :return:
        """

        # transfer matrices to compressed sparse column matrices
        K = sparse.csc_matrix(self.global_stiffness_matrix.copy(), dtype=float)
        F = sparse.csc_matrix(self.global_force_vector[:, :3].copy(), dtype=float)

        # calculate system with static solver
        ini_solver = StaticSolver()
        ini_solver.initialise(self.total_n_dof, self.time[:3])
        ini_solver.calculate(K, F, 0, 2)

        # transfer displacements result to the solver of the main calculation
        self.solver.u[0, :] = ini_solver.u[1, :]

    def initialise_global_matrices(self):
        """
        Inititialises all the global matrices and global force vector as empty sparse matrices

        :return:
        """
        print("Initialising global matrices")

        # initialise global lil matrices
        self.global_stiffness_matrix = sparse.lil_matrix(
            (self.total_n_dof, self.total_n_dof), dtype=float
        )
        self.global_damping_matrix = sparse.lil_matrix(
            (self.total_n_dof, self.total_n_dof), dtype=float
        )
        self.global_mass_matrix = sparse.lil_matrix(
            (self.total_n_dof, self.total_n_dof), dtype=float
        )

        self.global_force_vector = sparse.lil_matrix((self.total_n_dof, len(self.time)),dtype=float)

    def __calculate_rayleigh_damping_factors(self):
        """
        Calculate rayleigh damping coefficients
        :return:
        """
        constant = (
                2
                * self.damping_ratio
                / (self.radial_frequency_one + self.radial_frequency_two)
        )
        a0 = self.radial_frequency_one * self.radial_frequency_two * constant
        a1 = constant
        return a0, a1

    def calculate_rayleigh_damping(self):
        """
        Calculates the rayleigh damping matrix and adds it to the global damping matrix. Rayleigh damping is calculated
        with the following formula:

        .. math::
            C = a0 \cdot M + a1 \cdot K  \n
            a0 = \\omega 1 * \\omega 2 * a1 \n
            a1 = 2 * \\xi / (\\omega 1 * \\omega 2) \n

        Where C is the rayleigh damping matrix \n
        M is the global mass matrix \n
        K is the global stiffness matrix \n
        :math:`{\\omega}1` and :math:`{\\omega}2` are the first and second radial frequency \n
        :math:`{\\xi}` is the rayleigh damping factor \n

        :return:
        """

        # calculate rayleigh damping if required
        if self.is_rayleigh_damping:
            a0, a1 = self.__calculate_rayleigh_damping_factors()
            rayleigh_damping_matrix = self.global_mass_matrix * a0 + self.global_stiffness_matrix * a1

            # add rayleigh damping matrix to the global damping matrix
            self.global_damping_matrix += rayleigh_damping_matrix

    def initialise_ndof(self):
        """
        Initialise total number of degrees of freedom in the global system and sets nodal dof index in the global
        matrices.

        :return:
        """
        ndof = 0 # number of degrees of freedom
        index_dof = 0 # index of the degree of freedom in the global system
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
        Find indices of unique time step sizes, this indices represent the division between different stages.

        :return:
        """
        diff = np.diff(self.time)
        new_dt_idxs = sorted(np.unique(diff.round(decimals=15), return_index=True)[1])
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
        Initialises model parts, degrees of freedom, global matrices, stages and solver.

        :return:
        """

        self.mesh.reorder_element_ids()
        self.mesh.reorder_node_ids()

        self.initialise_model_parts()
        self.initialise_ndof()
        self.initialise_global_matrices()
        self.add_model_parts_to_global_matrices()
        self.calculate_rayleigh_damping()

        self.set_stage_time_ids()

        self.solver.initialise(self.total_n_dof, self.time)

    def update(self, start_time_id, end_time_id):
        """
        Updates model parts and solver

        :param start_time_id: first time index of the stage
        :param end_time_id: final time index of the stage
        :return:
        """
        # update model parts
        self.update_model_parts()

        # update solver
        self.solver.update(start_time_id)

    def calculate_stage(self, start_time_id, end_time_id):
        """
        Calculates a stage of the global system

        :param start_time_id: first time index of the stage
        :param end_time_id: final time index of the stage
        :return:
        """
        print("Calculating calculation phase")

        # transfer matrices to compressed sparse column matrices
        M = self.global_mass_matrix.tocsc()
        C = self.global_damping_matrix.tocsc()
        K = self.global_stiffness_matrix.tocsc()
        self.global_force_vector = self.global_force_vector.tocsc()
        F = self.global_force_vector

        # run_stages with Zhai solver if required
        if isinstance(self.solver, ZhaiSolver):
            self.solver.calculate(M, C, K, F, start_time_id, end_time_id)

        # run_stages with Newmark solver if required
        if isinstance(self.solver, NewmarkSolver):
            self.solver.calculate(M, C, K, F, start_time_id, end_time_id)

        # run_stages with Static solver if required
        if isinstance(self.solver, StaticSolver):
            self.solver.calculate(K, F, start_time_id, end_time_id)

        # assign the solver results to the nodes.
        self.assign_results_to_nodes()

    def calculate_force_in_elements(self):
        """
        Calculates force in each element. The force is calculate by multiplying the local stiffness matrix of the
        element with the displacements in the nodes of the element. The result on the nodes is then averaged to
        calculate the force in the element.

        :return:
        """

        # loop over elements
        for element in self.mesh.elements:

            # get auxiliary stiffness matrix of the element
            for model_part in element.model_parts:
                if isinstance(model_part, ElementModelPart):
                    aux_matrix = model_part.aux_stiffness_matrix
                    mask = [model_part.normal_dof, model_part.y_disp_dof, model_part.z_rot_dof]
                    break
            else:
                model_part = None
                aux_matrix = None
                mask = None

            # if element has 2 nodes, calculate force in element
            #todo make general for 2 or 3D elements
            if len(element.nodes) == 2 and model_part is not None and aux_matrix is not None and mask is not None:
                # get nodal displacements and stack them
                displacements = np.concatenate([node.displacements for node in element.nodes], axis=1)

                # get element rotation matrix
                rot_matrix = model_part.rotation_matrix

                # rotate displacements if rotate matrix is present
                # calculate nodal force
                if rot_matrix is not None:
                    inv_rot_matrix = np.linalg.inv(model_part.rotation_matrix)
                    force = aux_matrix.dot(inv_rot_matrix.dot(displacements.T))
                else:
                    force = aux_matrix.dot(displacements.T)

                # calculate mean of nodal forces, taking into account direction and assign to element
                nodal_force = np.array_split(force, len(element.nodes), axis=0)
                element_force = (nodal_force[0] - nodal_force[1]) /2
                element_force = element_force.T[:, mask]
                element.assign_force(element_force, mask)

    def finalise(self):
        """
        Finalises calculation. Forces in the elements are calculated, furthermore,  output displacements, velocities
        and accelerations are assigned.
        :return:
        """

        print("Finalising calculation")

        self.solver.finalise()
        self.calculate_force_in_elements()
        self.displacements_out = self.solver.u
        self.velocities_out = self.solver.v
        self.accelerations_out = self.solver.a

        self.time_out = self.solver.time_out

    def _assign_result_to_node(self, node):
        """
        Assigns solver results to a node. Displacents, velocities, accelerations and forces are assigned to the node.

        :param node: Node to be filled
        :return:
        """

        node_ids_dofs = list(node.index_dof[node.index_dof != None])
        node.assign_result(
            self.solver.u[:, node_ids_dofs],
            self.solver.v[:, node_ids_dofs],
            self.solver.a[:, node_ids_dofs],
            force = self.solver.f[:, node_ids_dofs]
        )
        return node

    def assign_results_to_nodes(self):
        """
        Assigns all solver results to all nodes in the mesh

        :return:
        """
        vec_f = np.vectorize(self._assign_result_to_node)
        self.mesh.nodes = vec_f(self.mesh.nodes)

    def main(self):
        """
        Main function of the class. Input is validated, then the system is initialised. Each stage is calculated and the
        system is finalised.

        :return:
        """

        self.validate_input()
        self.initialise()

        # calculate stages
        for i in range(len(self.stage_time_ids) - 1):
            self.update(self.stage_time_ids[i], self.stage_time_ids[i + 1])
            self.calculate_stage(self.stage_time_ids[i], self.stage_time_ids[i + 1])

        self.finalise()
