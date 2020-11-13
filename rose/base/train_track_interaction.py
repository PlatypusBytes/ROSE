from rose.train_model.train_model import TrainModel
from rose.base.global_system import GlobalSystem
from rose.base.model_part import ElementModelPart, TimoshenkoBeamElementModelPart
from rose.utils.utils import *
from rose.utils.mesh_utils import *
from rose.solver.solver import NewmarkSolver, StaticSolver, ZhaiSolver

import numpy as np
from scipy import sparse
from copy import deepcopy

class CoupledTrainTrack():
    def __init__(self):
        self.train = None
        self.track = None
        self.rail = None

        self.time = None
        self.initialisation_time = None

        self.herzian_contact_coef = None
        self.herzian_power = None

        self.irregularities_at_wheels = None

        self.global_mass_matrix = None
        self.global_damping_matrix = None
        self.global_stiffness_matrix = None
        self.global_force_vector = None

        self.total_n_dof = None

        self.solver = None
        self.velocities = None
        self.wheel_loads = None

        self.force_vector = None
        self.static_force_vector = None

        self.g = 9.81
        self.velocities = None
        self.time = None

        self.deformation_wheels = None  # np.zeros((4, 1))
        self.deformation_track_at_wheels = None  # np.zeros((4, 1))
        self.irregularities_at_wheels = None  # np.zeros(4)
        self.total_static_load = None

        self.contact_dofs = None

        self.track_elements = None
        self.wheel_distances = None
        self.wheel_node_dist = None
        self.y_shape_factors_track = None

    def get_position_wheels(self, t):
        position_wheels = [wheel.distances for wheel in self.train.wheels]
        position_wheels_at_t = [position_wheel[t] for position_wheel in position_wheels]
        return position_wheels_at_t

    def set_wheel_load_on_track(self, track_elements, wheel_loads,t):
        for idx, track_element in enumerate(track_elements):
            global_indices = np.array([track_element.nodes[0].index_dof[1], track_element.nodes[0].index_dof[2],
                                       track_element.nodes[1].index_dof[1], track_element.nodes[1].index_dof[2]])

            force_vector = np.array([self.y_shape_factors[idx, t, 0] * wheel_loads[idx],
                                     self.y_shape_factors[idx, t, 1] * wheel_loads[idx],
                                     self.y_shape_factors[idx, t, 2] * wheel_loads[idx],
                                     self.y_shape_factors[idx, t, 3] * wheel_loads[idx]])

            mask = [i for i in range(len(global_indices)) if global_indices[i] is not None]
            t_idxs = np.ones(len(mask)) * t

            self.track.global_force_vector[global_indices[mask],t_idxs] = force_vector[mask]


    def initialize_track_elements(self):
        self.track_elements =np.empty((len(self.wheel_loads), len(self.time)))
        self.track_elements = []
        for idx, wheel_load in enumerate(self.wheel_loads):
            wheel_load_np = np.array(wheel_load.elements)
            self.track_elements.append(wheel_load_np[wheel_load.active_elements.nonzero()[0]])

        self.track_elements = np.array(self.track_elements)


    def calculate_distance_wheels_track_nodes(self):
        self.wheel_distances = np.array([wheel.distances for wheel in self.train.wheels])

        nodal_coordinates = np.array([np.array([element.nodes[0].coordinates for element in wheel])
                                      for wheel in self.track_elements])

        cumulative_dist_nodes = np.array([calculate_cum_distances_coordinate_array(wheel_coords)
                                          for wheel_coords in nodal_coordinates])

        self.wheel_node_dist = np.array([wheel - cumulative_dist_nodes[idx] for idx, wheel in enumerate(self.wheel_distances)])



    def calculate_y_shape_factors(self):

        self.y_shape_factors = np.zeros((len(self.wheel_node_dist),len(self.wheel_node_dist[0]), 4))
        for w_idx , w_track_elements in enumerate(self.track_elements):
            for idx, element in enumerate(w_track_elements):
                for model_part in element.model_parts:
                    if isinstance(model_part, TimoshenkoBeamElementModelPart):
                        model_part.set_y_shape_functions(self.wheel_node_dist[0, idx])
                        self.y_shape_factors[w_idx,idx, :] = copy.deepcopy(model_part.y_shape_functions)



    def get_track_element_at_wheels(self, t):
        return self.track_elements[:,t]

    def __get_wheel_dofs(self):
        return [wheel.nodes[0].index_dof[1] for wheel in self.train.wheels]

    def calculate_active_n_dof(self):
        self.train.calculate_active_n_dof()


    def __calculate_elastic_wheel_deformation(self, t):

        elastic_wheel_deformation = (
            # self.static_wheel_deformation
                - self.train.irregularities_at_wheels[:, t]
        )

        return elastic_wheel_deformation

    def calculate_static_contact_deformation(self):
        G = self.herzian_contact_coef
        pow = self.herzian_power

        self.static_contact_deformation = np.array([np.sign(wheel.total_static_load)* G * abs(wheel.total_static_load)
                                                    ** (1/pow) for wheel in self.train.wheels])

    def calculate_wheel_rail_contact_force(self, t, du_wheels):

        elastic_wheel_deformation = self.__calculate_elastic_wheel_deformation(t)

        contact_force = np.sign(elastic_wheel_deformation - du_wheels) * \
                        np.nan_to_num((1 / self.herzian_contact_coef * abs((elastic_wheel_deformation - du_wheels))
                        ) ** self.herzian_power)

        return contact_force

    def get_disp_track_at_wheels(self, track_elements, t, u):
        disp_at_wheels = np.zeros(len(track_elements))
        for idx, track_element in enumerate(track_elements):

            global_indices = [track_element.nodes[0].index_dof[1], track_element.nodes[0].index_dof[2],
                              track_element.nodes[1].index_dof[1], track_element.nodes[1].index_dof[2]]

            # get displacement at indices
            if None not in global_indices:
                disp = u[global_indices]
            else:
                # set inactive dofs at 0
                disp = np.zeros(len(global_indices))
                for i, v in enumerate(global_indices):
                    if v is None:
                        disp[i] = 0
                    else:
                        disp[i] = u[v]

            disp_at_wheels[idx] = np.sum(disp * self.y_shape_factors[idx,t,:])

        return disp_at_wheels

    def update_force_vector(self,u, t):
        u_wheels = u[self.train.contact_dofs]
        u_stat = self.static_contact_deformation

        contact_track_elements = self.get_track_element_at_wheels(t)
        disp_at_wheels = self.get_disp_track_at_wheels(contact_track_elements, t, u)
        contact_force = self.calculate_wheel_rail_contact_force(t, u_wheels + u_stat - disp_at_wheels)

        F = copy.deepcopy(self.global_force_vector[:,t])
        F[self.train.contact_dofs] += contact_force

        self.set_wheel_load_on_track(contact_track_elements, -contact_force, t)

        F[:self.track.total_n_dof] = self.track.global_force_vector[:, t].toarray()
        return F


    def calculate_static_load(self):
        self.train.calculate_total_static_load()
        self.track.calculate_total_static_load()

    def initialize_force_vector(self):

        self.train.initialize_force_vector()
        self.track.initialize_force_vector()

    def initialize_global_matrices(self):
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

    def combine_global_matrices(self):
        # add track
        self.global_stiffness_matrix[:self.track.total_n_dof, :self.track.total_n_dof] = self.track.global_stiffness_matrix
        self.global_damping_matrix[:self.track.total_n_dof, :self.track.total_n_dof] = self.track.global_damping_matrix
        self.global_mass_matrix[:self.track.total_n_dof,:self.track.total_n_dof] = self.track.global_mass_matrix
        self.global_force_vector[:self.track.total_n_dof,:] = self.track.global_force_vector

        self.solver.u[:,:self.track.total_n_dof] = self.track.solver.u[:,:]
        self.solver.v[:, :self.track.total_n_dof] = self.track.solver.v[:, :]

        # add train
        self.global_stiffness_matrix[self.track.total_n_dof:self.total_n_dof,
        self.track.total_n_dof:self.total_n_dof] = self.train.global_stiffness_matrix

        self.global_damping_matrix[self.track.total_n_dof:self.total_n_dof,
        self.track.total_n_dof:self.total_n_dof] = self.train.global_damping_matrix

        self.global_mass_matrix[self.track.total_n_dof:self.total_n_dof,
        self.track.total_n_dof:self.total_n_dof] = self.train.global_mass_matrix

        self.global_force_vector[self.track.total_n_dof:self.total_n_dof, :] = self.train.global_force_vector

        self.solver.u[:,self.track.total_n_dof:self.total_n_dof] = self.train.solver.u[:,:]
        self.solver.v[:, self.track.total_n_dof:self.total_n_dof] = self.train.solver.v[:, :]



    def initialise_ndof(self):
        # self.track.initialise_ndof()
        # self.train.calculate_active_n_dof()

        for node in self.train.nodes:
            for idx, index_dof in enumerate(node.index_dof):
                if index_dof is not None:
                    node.index_dof[idx] += self.track.total_n_dof

        # recalculate contact dofs train
        self.train.get_contact_dofs()

        self.total_n_dof = self.track.total_n_dof + self.train.active_n_dof


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

        # run_stages with Zhai solver
        if isinstance(self.solver, ZhaiSolver):
            self.solver.calculate(M, C, K, F, start_time_id, end_time_id)

        # run_stages with Newmark solver
        if isinstance(self.solver, NewmarkSolver):
            self.solver.calculate(M, C, K, F, start_time_id, end_time_id)

        # run_stages with Static solver
        if isinstance(self.solver, StaticSolver):
            self.solver.calculate(K, F, start_time_id, end_time_id)

        # self.assign_results_to_nodes()


    def calculate_initial_displacement_track(self):
        # transfer matrices to compressed sparsed column matrices
        K = sparse.lil_matrix(deepcopy(self.track.global_stiffness_matrix))
        F = sparse.lil_matrix(deepcopy(self.track.global_force_vector[:,:3]))

        ini_solver = StaticSolver()
        ini_solver.initialise(self.track.total_n_dof, self.time[:3])
        ini_solver.calculate(K, F, 0, 2)

        # self.track.solver.u[0,:] = ini_solver.u[2,:]
        pass


    def calculate_initial_displacement_train(self, wheel_displacements):
        # transfer matrices to compressed sparsed column matrices
        K = sparse.lil_matrix(deepcopy(self.train.global_stiffness_matrix))
        F = sparse.lil_matrix(deepcopy(self.train.global_force_vector))

        wheel_dofs = [wheel.nodes[0].index_dof[1] -self.track.total_n_dof for wheel in self.train.wheels]
        ini_solver = StaticSolver()
        ini_solver.initialise(self.train.active_n_dof - len(wheel_dofs), self.time)
        K = utils.delete_from_lil(
            K, row_indices=wheel_dofs, col_indices=wheel_dofs).tocsc()
        F = utils.delete_from_lil(
            F, row_indices=wheel_dofs).tocsc()
        ini_solver.calculate(K, F, 0, 1)

        self.train.solver.initialise(self.train.active_n_dof, self.time)
        # todo take into account initial differential settlements between wheels, for now max displacement of wheel is taken
        mask = np.ones(self.train.solver.u[0, :].shape, bool)
        mask[wheel_dofs] = False
        self.train.solver.u[0, mask] = ini_solver.u[1, :] + max(wheel_displacements)
        self.train.solver.u[0, wheel_dofs] = wheel_displacements

        pass


    def update_stage(self, start_time_id, end_time_id):
        """
        Updates model parts and solver
        :param start_time_id:
        :param end_time_id:
        :return:
        """
        self.solver.update(start_time_id)

    def calculate_initial_state(self):

        self.calculate_initial_displacement_track()

        contact_track_elements = self.get_track_element_at_wheels(0)
        # wheel_distances = [wheel.distances[0] for wheel in self.train.wheels]
        disp_at_wheels = self.get_disp_track_at_wheels(contact_track_elements, 0, self.track.solver.u[0,:])

        self.calculate_initial_displacement_train(disp_at_wheels)
        self.calculate_static_contact_deformation()

        self.combine_global_matrices()

        self.solver.load_func = self.update_force_vector




    def initialize_wheel_loads(self):
        self.wheel_loads = []
        for wheel in self.train.wheels:
            load = add_moving_point_load_to_track(
                self.rail,
                self.time,
                0,
                self.velocities,
                y_load=wheel.total_static_load, start_coords=[wheel.distances[0], 0, 0]
            )
            # todo set y and z start coords


            self.wheel_loads.extend(list(load.values()))
            self.wheel_loads[-1].y_force[:, 0] = 0
            self.wheel_loads[-1].z_moment[:, 0] = 0
        self.track.model_parts.extend(self.wheel_loads)


    def initialise(self):

        self.train.solver = self.solver.__class__()
        self.track.solver = self.solver.__class__()

        self.train.initialise()
        # self.track.initialise()

        self.initialize_wheel_loads()
        self.track.initialise()
        self.initialise_ndof()
        self.initialize_global_matrices()

        self.initialize_track_elements()

        self.calculate_distance_wheels_track_nodes()
        self.calculate_y_shape_factors()

        self.solver.initialise(self.total_n_dof, self.time)

        self.calculate_initial_state()
        self.set_stage_time_ids()


    def set_stage_time_ids(self):
        """
        Find indices of unique time steps
        :return:
        """
        diff = np.diff(self.time)
        new_dt_idxs = sorted(np.unique(diff.round(decimals=15), return_index=True)[1])
        self.stage_time_ids = np.append(new_dt_idxs, len(self.time) - 1)

    def main(self):

        self.initialise()

        # calculate stages
        for i in range(len(self.stage_time_ids) - 1):
            # self.update_stage(0, len(self.time) - 1)
            # self.calculate_stage(0, len(self.time) - 1)

            self.update_stage(self.stage_time_ids[i], self.stage_time_ids[i + 1])
            self.calculate_stage(self.stage_time_ids[i], self.stage_time_ids[i + 1])

        # self.finalise()