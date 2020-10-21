from src.train_model.train_model import TrainModel
from src.global_system import GlobalSystem
from src.model_part import ElementModelPart, TimoshenkoBeamElementModelPart
from src.utils import *

from src.solver import NewmarkSolver, StaticSolver, ZhaiSolver

import numpy as np
from scipy import sparse

class CoupledTrainTrack():
    def __init__(self):
        self.train = None
        self.track = None

        self.time = None

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

    def __calculate_elastic_wheel_deformation(self, du, t):
        elastic_wheel_deformation = (
                # self.static_wheel_deformation
                du
                - self.irregularities_at_wheels[:, t]
        )


        return elastic_wheel_deformation


    def get_position_wheels(self, t):
        position_wheels = [wheel.distances for wheel in self.train.wheels]
        position_wheels_at_t = [position_wheel[t] for position_wheel in position_wheels]
        return position_wheels_at_t


    def get_displacement_track_at_wheels(self, position_wheels,track_elements, u):
        # for idx, wheel_load in enumerate(self.wheel_loads):
        #     active_element_idx = wheel_load.active_elements.nonzero()[0][t]
        #     track_element = wheel_load.elements[active_element_idx]
        disp_at_wheels = np.empty(len(track_elements))
        for idx, track_element in enumerate(track_elements):
            for model_part in track_element.model_parts:
                if isinstance(model_part, TimoshenkoBeamElementModelPart):
                    track_model_part = model_part
                    dist = distance_np(position_wheels[idx], np.array(track_element.nodes[0].coordinates))

                    track_model_part.set_y_shape_functions(dist)
                    displacements_element = u[[track_element.nodes[0].index_dof[0],
                                                track_element.nodes[0].index_dof[1],
                                                track_element.nodes[1].index_dof[0],
                                                track_element.nodes[1].index_dof[1]]]

                    disp_at_wheels[idx] = sum(displacements_element / track_model_part.y_shape_functions)
                    break

                return disp_at_wheels

    def set_wheel_load_on_track(self, track_elements, position_wheels, wheel_loads,t):
        for idx, track_element in enumerate(track_elements):
            for model_part in track_element.model_parts:
                if isinstance(model_part, TimoshenkoBeamElementModelPart):
                    track_model_part = model_part
                    dist = distance_np(position_wheels[idx], np.array(track_element.nodes[0].coordinates))
                    track_model_part.set_y_shape_functions(dist)

                    global_indices = [track_element.nodes[0].index_dof[1], track_element.nodes[0].index_dof[2],
                                      track_element.nodes[1].index_dof[1], track_element.nodes[0].index_dof[2]]
                    self.global_force_vector[global_indices,[t,t,t,t]] = [track_model_part.y_shape_functions[0] * wheel_loads[idx],
                                                                           track_model_part.y_shape_functions[1] * wheel_loads[idx],
                                                                           track_model_part.y_shape_functions[2] * wheel_loads[idx],
                                                                           track_model_part.y_shape_functions[3] * wheel_loads[idx]]


    def get_track_element_at_wheels(self, t):
        track_elements = [None] * len(self.wheel_loads)
        for idx, wheel_load in enumerate(self.wheel_loads):
            active_element_idx = wheel_load.active_elements.nonzero()[0][t]
            track_elements[idx] = wheel_load.elements[active_element_idx].nodes
        return track_elements


    def __get_wheel_dofs(self):
        return [wheel.nodes[0].index_dof[1] for wheel in self.train.wheels]

    def calculate_wheel_rail_contact_force(self, u, t):

        track_elements = self.get_track_element_at_wheels(t)
        position_wheels = self.get_position_wheels(t)
        disp_track_at_wheel = self.get_displacement_track_at_wheels(position_wheels,track_elements, u)

        wheel_dofs = self.__get_wheel_dofs()
        disp_wheel = u[wheel_dofs]

        du = disp_wheel - disp_track_at_wheel
        elastic_wheel_deformation = self.__calculate_elastic_wheel_deformation(du, t)

        contact_force = np.sign(elastic_wheel_deformation)*(
                                1 / self.herzian_contact_coef * abs(elastic_wheel_deformation)
                        ) ** self.herzian_power

        self.global_force_vector[wheel_dofs,t] = contact_force
        self.set_wheel_load_on_track(track_elements, position_wheels, contact_force, t)

        return self.global_force_vector[:,t]


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

        # add train
        self.global_stiffness_matrix[self.track.total_n_dof:self.total_n_dof,
        self.track.total_n_dof:self.total_n_dof] = self.train.global_stiffness_matrix

        self.global_damping_matrix[self.track.total_n_dof:self.total_n_dof,
        self.track.total_n_dof:self.total_n_dof] = self.train.global_damping_matrix

        self.global_mass_matrix[self.track.total_n_dof:self.total_n_dof,
        self.track.total_n_dof:self.total_n_dof] = self.train.global_mass_matrix

        self.global_force_vector[self.track.total_n_dof:self.total_n_dof, :] = self.train.global_force_vector

    def initialise_ndof(self):
        for node in self.train.nodes:
            for idx, index_dof in enumerate(node.index_dof):
                if index_dof is not None:
                    node.index_dof[idx] += self.track.total_n_dof

        self.total_n_dof = self.track.total_n_dof + self.train.total_n_dof


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


    def initialise(self):
        self.track.initialise()
        self.train.initialize()


    def main(self):

        self.initialise()

        # calculate stages
        for i in range(len(self.stage_time_ids) - 1):
            self.update(self.stage_time_ids[i], self.stage_time_ids[i + 1])
            self.calculate_stage(self.stage_time_ids[i], self.stage_time_ids[i + 1])

        self.finalise()