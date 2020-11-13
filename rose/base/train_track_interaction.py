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

    # def __calculate_elastic_wheel_deformation(self, du, t):
    #     elastic_wheel_deformation = (
    #             # self.static_wheel_deformation
    #             du
    #             - self.irregularities_at_wheels[:, t]
    #     )
    #
    #
    #     return elastic_wheel_deformation


    def get_position_wheels(self, t):
        position_wheels = [wheel.distances for wheel in self.train.wheels]
        position_wheels_at_t = [position_wheel[t] for position_wheel in position_wheels]
        return position_wheels_at_t


    # def get_displacement_track_at_wheels(self, position_wheels,track_elements, u):
    #     # for idx, wheel_load in enumerate(self.wheel_loads):
    #     #     active_element_idx = wheel_load.active_elements.nonzero()[0][t]
    #     #     track_element = wheel_load.elements[active_element_idx]
    #     disp_at_wheels = np.empty(len(track_elements))
    #     for idx, track_element in enumerate(track_elements):
    #         for model_part in track_element.model_parts:
    #             if isinstance(model_part, TimoshenkoBeamElementModelPart):
    #                 track_model_part = model_part
    #                 dist = distance_np(position_wheels[idx], np.array(track_element.nodes[0].coordinates))
    #
    #                 track_model_part.set_y_shape_functions(dist)
    #                 displacements_element = u[[track_element.nodes[0].index_dof[0],
    #                                             track_element.nodes[0].index_dof[1],
    #                                             track_element.nodes[1].index_dof[0],
    #                                             track_element.nodes[1].index_dof[1]]]
    #
    #                 disp_at_wheels[idx] = sum(displacements_element / track_model_part.y_shape_functions)
    #                 break
    #
    #             return disp_at_wheels

    def set_wheel_load_on_track(self, track_elements, position_wheels, wheel_loads,t):
        for idx, track_element in enumerate(track_elements):
            for model_part in track_element.model_parts:
                if isinstance(model_part, TimoshenkoBeamElementModelPart):
                    track_model_part = model_part
                    #todo calc distance for non horizontal track
                    # dist = distance_np(position_wheels[idx], np.array(track_element.nodes[0].coordinates))
                    dist = distance_np(position_wheels[idx], track_element.nodes[0].coordinates[0])
                    track_model_part.set_y_shape_functions(dist)

                    global_indices = np.array([track_element.nodes[0].index_dof[1], track_element.nodes[0].index_dof[2],
                                               track_element.nodes[1].index_dof[1], track_element.nodes[1].index_dof[2]])

                    mask = []
                    for i in range(len(global_indices)):
                        if global_indices[i] is not None:
                            mask.append(i)

                    force_vector = np.array([track_model_part.y_shape_functions[0] * wheel_loads[idx],
                                                                           track_model_part.y_shape_functions[1] * wheel_loads[idx],
                                                                           track_model_part.y_shape_functions[2] * wheel_loads[idx],
                                                                           track_model_part.y_shape_functions[3] * wheel_loads[idx]])


                    t_idxs = np.ones(len(mask)) * t


                    self.track.global_force_vector[global_indices[mask],t_idxs] = force_vector[mask]
                    pass

    def initialize_track_elements(self):
        self.track_elements =np.empty((len(self.wheel_loads), len(self.time)))
        self.track_elements = []


        for idx, wheel_load in enumerate(self.wheel_loads):
            wheel_load_np = np.array(wheel_load.elements)
            self.track_elements.append(wheel_load_np[wheel_load.active_elements.nonzero()[0]])
        self.track_elements = np.array(self.track_elements)
        # test[self.wheel_loads[0].active_elements.nonzero()[0]]

        # for t in range(len(self.time)):
        #     active_elements = [wheel_load.elements[wheel_load.active_elements.nonzero()[0][t]] for wheel_load in self.wheel_loads]
        #     self.track_elements[t] = active_elements
        pass



    def get_track_element_at_wheels(self, t):

        # track_elements =
        #
        # for idx, wheel_load in enumerate(self.wheel_loads):
        #     self.track_elements[idx,t]
        #     active_element_idx = wheel_load.active_elements.nonzero()[0][t]
        #     track_elements[idx] = wheel_load.elements[active_element_idx]
        return self.track_elements[:,t]


    def __get_wheel_dofs(self):
        return [wheel.nodes[0].index_dof[1] for wheel in self.train.wheels]


    def calculate_active_n_dof(self):
        self.train.calculate_active_n_dof()


    def __calculate_elastic_wheel_deformation(self, t):
        # elastic_wheel_deformation = (
        #         # self.static_wheel_deformation
        #         + self.deformation_wheels
        #         - self.deformation_track_at_wheels
        #         - self.irregularities_at_wheels[:, t]
        # )

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

    def get_disp_track_at_wheels(self, track_elements, position_wheels, u):
        disp_at_wheels = np.zeros(len(position_wheels))
        for idx, track_element in enumerate(track_elements):
            for model_part in track_element.model_parts:
                if isinstance(model_part, TimoshenkoBeamElementModelPart):
                    #todo calc cumulative distance track elements, for now programm works with horizontal track
                    # dist = distance_np(position_wheels[idx], np.array(track_element.nodes[0].coordinates))
                    dist = distance_np(position_wheels[idx], track_element.nodes[0].coordinates[0])
                    model_part.set_y_shape_functions(dist)
                    test = [track_element.nodes[0].index_dof[1], track_element.nodes[0].index_dof[2],
                            track_element.nodes[1].index_dof[1], track_element.nodes[1].index_dof[2]]
                    disp = np.zeros(len(test))
                    if None not in test:
                        disp = u[test]
                    else:
                        for i ,v in enumerate(test):
                            if v is None:
                                disp[i] = 0
                            else:
                                disp[i] = u[v]
                    disp_at_wheels[idx] = np.sum(disp * model_part.y_shape_functions)
                    break
        return disp_at_wheels

    def update_force_vector(self,u, t):
        u_wheels = u[self.train.contact_dofs]
        u_stat = self.static_contact_deformation

        test = self.calculate_static_contact_deformation()
        contact_track_elements = self.get_track_element_at_wheels(t)
        wheel_distances = [wheel.distances[t] for wheel in self.train.wheels]
        disp_at_wheels = self.get_disp_track_at_wheels(contact_track_elements, wheel_distances, u)

        contact_force = self.calculate_wheel_rail_contact_force(t, u_wheels + u_stat - disp_at_wheels)

        F = copy.deepcopy(self.global_force_vector[:,t])
        F[self.train.contact_dofs] += contact_force

        self.set_wheel_load_on_track(contact_track_elements, wheel_distances, -contact_force, t)

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
        wheel_distances = [wheel.distances[0] for wheel in self.train.wheels]
        disp_at_wheels = self.get_disp_track_at_wheels(contact_track_elements, wheel_distances, self.track.solver.u[0,:])

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